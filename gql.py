"""
GraphQL subset executor for the git forge.
"""

import json
import base64
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Iterator
from dataclasses import dataclass

from gitcore import (
    ObjectStore,
    parse_commit,
    build_commit,
    flatten_tree,
    tree_from_flat,
    build_tree,
    diff_trees,
    is_ancestor,
    ZERO_SHA,
    TYPE_NAME,
)
from store import (
    Store,
    Repo,
    iso_utc,
    git_ident,
    merge_pull,
)


# ============================================================
# GID encoding
# ============================================================

def gid(kind: str, key: str) -> str:
    """Global ID encoding."""
    return base64.b64encode(f"{kind}:{key}".encode()).decode()


def decode_gid(gid_str: str) -> Optional[Tuple[str, str]]:
    """Decode a global ID. Returns (kind, key) or None."""
    try:
        decoded = base64.b64decode(gid_str).decode()
        if ":" not in decoded:
            return None
        kind, key = decoded.split(":", 1)
        return kind, key
    except Exception:
        return None


# ============================================================
# GraphQL Lexer
# ============================================================

@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            self._tokenize_one()
        self.tokens.append(Token("EOF", "", self.line, self.col))
        return self.tokens
    
    def _skip_whitespace(self):
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "#":
                # Comment
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            elif ch in " \t\r\n":
                if ch == "\n":
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1
            else:
                break
    
    def _tokenize_one(self):
        ch = self.text[self.pos]
        start_line, start_col = self.line, self.col
        
        if ch.isalpha() or ch == "_":
            # Name
            val = ch
            self.pos += 1
            self.col += 1
            while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
                val += self.text[self.pos]
                self.pos += 1
                self.col += 1
            self.tokens.append(Token("NAME", val, start_line, start_col))
            return
        
        if ch == '"':
            # String
            val, self.pos = self._read_string()
            self.tokens.append(Token("STRING", val, start_line, start_col))
            return
        
        if ch == "-" or ch.isdigit():
            # Number
            val = ch
            self.pos += 1
            self.col += 1
            is_float = False
            while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == "."):
                if self.text[self.pos] == ".":
                    is_float = True
                val += self.text[self.pos]
                self.pos += 1
                self.col += 1
            token_type = "FLOAT" if is_float else "INT"
            self.tokens.append(Token(token_type, val, start_line, start_col))
            return
        
        # Punctuation
        if self.text[self.pos:self.pos+3] == "...":
            self.tokens.append(Token("SPREAD", "...", start_line, start_col))
            self.pos += 3
            self.col += 3
            return
        
        single_char = {
            "{": "LBRACE", "}": "RBRACE",
            "(": "LPAREN", ")": "RPAREN",
            "[": "LBRACKET", "]": "RBRACKET",
            ":": "COLON", "=": "EQUALS",
            "@": "AT", "$": "DOLLAR", "!": "BANG",
            ",": "COMMA",
        }
        if ch in single_char:
            self.tokens.append(Token(single_char[ch], ch, start_line, start_col))
            self.pos += 1
            self.col += 1
            return
        
        raise ValueError(f"Unexpected character: {ch} at line {self.line}, col {self.col}")
    
    def _read_string(self) -> Tuple[str, int]:
        # Called when at opening quote
        self.pos += 1  # skip opening quote
        self.col += 1
        val = ""
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == '"':
                self.pos += 1
                self.col += 1
                break
            elif ch == "\\":
                self.pos += 1
                self.col += 1
                if self.pos >= len(self.text):
                    break
                esc = self.text[self.pos]
                if esc == "u":
                    self.pos += 1
                    self.col += 1
                    hex_digits = self.text[self.pos:self.pos+4]
                    val += chr(int(hex_digits, 16))
                    self.pos += 4
                    self.col += 4
                elif esc == "n":
                    val += "\n"
                elif esc == "t":
                    val += "\t"
                elif esc == "r":
                    val += "\r"
                elif esc == '"':
                    val += '"'
                elif esc == "\\":
                    val += "\\"
                elif esc == "/":
                    val += "/"
                elif esc == "b":
                    val += "\b"
                elif esc == "f":
                    val += "\f"
                else:
                    val += esc
                self.pos += 1
                self.col += 1
            else:
                val += ch
                self.pos += 1
                self.col += 1
        return val, self.pos


# ============================================================
# GraphQL Parser
# ============================================================

@dataclass
class Argument:
    name: str
    value: Any


@dataclass
class Directive:
    name: str
    args: List[Argument]


@dataclass
class Selection:
    alias: Optional[str]
    name: str
    args: List[Argument]
    directives: List[Directive]
    selection_set: Optional[List["Selection"]]


@dataclass
class FragmentDefinition:
    name: str
    type_condition: str
    selection_set: List[Selection]


@dataclass
class OperationDefinition:
    type: str  # query, mutation
    name: Optional[str]
    variable_definitions: List[Any]  # ignored for now
    directives: List[Directive]
    selection_set: List[Selection]


@dataclass
class Document:
    operations: List[OperationDefinition]
    fragments: Dict[str, FragmentDefinition]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current(self) -> Token:
        return self.tokens[self.pos]
    
    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok
    
    def expect(self, token_type: str) -> Token:
        if self.current().type != token_type:
            raise ValueError(f"Expected {token_type}, got {self.current().type} at line {self.current().line}")
        return self.advance()
    
    def match(self, token_type: str) -> bool:
        if self.current().type == token_type:
            self.advance()
            return True
        return False
    
    def parse(self) -> Document:
        operations = []
        fragments = {}
        
        while self.current().type != "EOF":
            if self.match("NAME"):
                kw = self.tokens[self.pos - 1].value
                if kw == "query" or kw == "mutation":
                    operations.append(self._parse_operation(kw, named=True))
                elif kw == "fragment":
                    frag = self._parse_fragment()
                    fragments[frag.name] = frag
                else:
                    # Could be a query without the 'query' keyword, starting with a field
                    # Put the token back and parse as anonymous query
                    self.pos -= 1
                    operations.append(self._parse_operation("query"))
            elif self.match("LBRACE"):
                # Anonymous shorthand query operation
                self.pos -= 1  # Go back to the LBRACE
                operations.append(self._parse_operation("query"))
            else:
                raise ValueError(f"Unexpected token: {self.current().type}")
        
        return Document(operations, fragments)
    
    def _parse_operation(self, op_type: str, named: bool = False) -> OperationDefinition:
        name = None
        selection_set = None
        
        if self.current().type == "NAME":
            if named:
                # An operation-type keyword (query/mutation) was consumed, so this
                # NAME is the optional operation name, not a field.
                name = self.advance().value
            else:
                # Check if this is an operation name or a field
                next_pos = self.pos + 1
                if next_pos < len(self.tokens):
                    next_token = self.tokens[next_pos]
                    if next_token.type in ("LBRACE", "LPAREN"):
                        # This is a field, not an operation name. Treat as anonymous query.
                        name = None
                        # Parse the selection set starting from this field
                        selection_set = self._parse_selection_set()
                    else:
                        name = self.advance().value
                else:
                    name = self.advance().value
        
        if selection_set is None:
            variable_definitions = []
            if self.match("LPAREN"):
                while not self.match("RPAREN"):
                    if self.current().type == "DOLLAR":
                        self.advance()
                        self.expect("NAME")
                        if self.match("COLON"):
                            self._parse_type()
                        if self.match("EQUALS"):
                            self._parse_value()
                    if not self.match("COMMA"):
                        break
                self.match("RPAREN")
            
            directives = []
            while self.match("AT"):
                directives.append(self._parse_directive())
            
            self.expect("LBRACE")
            selection_set = self._parse_selection_set()
            self.expect("RBRACE")
        
        return OperationDefinition(op_type, name, [], [], selection_set)
    
    def _parse_fragment(self) -> FragmentDefinition:
        name = self.expect("NAME").value
        self.expect("NAME")  # "on"
        type_condition = self.expect("NAME").value
        
        directives = []
        while self.match("AT"):
            directives.append(self._parse_directive())
        
        self.expect("LBRACE")
        selection_set = self._parse_selection_set()
        self.expect("RBRACE")
        
        return FragmentDefinition(name, type_condition, selection_set)
    
    def _parse_selection_set(self) -> List[Selection]:
        selections = []
        while self.current().type != "RBRACE" and self.current().type != "EOF":
            if self.match("SPREAD"):
                # Handle fragment spread or inline fragment
                selections.append(self._parse_fragment_spread())
                continue
            start_pos = self.pos
            selections.append(self._parse_selection())
            if self.pos == start_pos:
                raise ValueError(f"Selection parser made no progress at token {self.current().type}")
        return selections
    
    def _parse_fragment_spread(self) -> Selection:
        """Parse fragment spread (...FragmentName) or inline fragment (... on TypeName)."""
        # We've already consumed the SPREAD token
        if self.match("NAME"):
            # Named fragment spread: ...FragmentName
            fragment_name = self.tokens[self.pos - 1].value
            return Selection(None, f"...{fragment_name}", [], [], None)
        elif self.match("NAME") and self.tokens[self.pos - 1].value == "on":
            # Inline fragment: ... on TypeName
            type_name = self.expect("NAME").value
            directives = []
            while self.match("AT"):
                directives.append(self._parse_directive())
            selection_set = None
            if self.match("LBRACE"):
                selection_set = self._parse_selection_set()
                self.expect("RBRACE")
            return Selection(None, f"... on {type_name}", [], directives, selection_set)
        else:
            raise ValueError("Expected fragment name or 'on' after '...'")
    
    def _parse_selection(self) -> Selection:
        alias = None
        name = ""
        
        if self.current().type != "NAME":
            raise ValueError(f"Expected field name, got {self.current().type} at line {self.current().line}")
        name = self.advance().value
        if self.match("COLON"):
            alias = name
            name = self.expect("NAME").value
        
        args = []
        if self.match("LPAREN"):
            while True:
                if self.match("RPAREN"):
                    break
                arg_name = self.expect("NAME").value
                self.expect("COLON")
                arg_value = self._parse_value()
                args.append(Argument(arg_name, arg_value))
                if not self.match("COMMA"):
                    self.expect("RPAREN")
                    break
        
        directives = []
        while self.match("AT"):
            directives.append(self._parse_directive())
        
        selection_set = None
        if self.match("LBRACE"):
            selection_set = self._parse_selection_set()
            self.expect("RBRACE")
        
        return Selection(alias, name, args, directives, selection_set)
    
    def _parse_directive(self) -> Directive:
        name = self.expect("NAME").value
        args = []
        if self.match("LPAREN"):
            while not self.match("RPAREN"):
                arg_name = self.expect("NAME").value
                self.expect("COLON")
                arg_value = self._parse_value()
                args.append(Argument(arg_name, arg_value))
                self.match("COMMA")
        return Directive(name, args)
    
    def _parse_type(self):
        # Simplified type parsing
        if self.match("NAME"):
            pass
        if self.match("BANG"):
            pass
        if self.match("LBRACKET"):
            self._parse_type()
            self.expect("RBRACKET")
            if self.match("BANG"):
                pass
    
    def _parse_value(self) -> Any:
        tok = self.current()
        if tok.type == "STRING":
            self.advance()
            return tok.value
        elif tok.type == "INT":
            self.advance()
            return int(tok.value)
        elif tok.type == "FLOAT":
            self.advance()
            return float(tok.value)
        elif tok.type == "NAME":
            val = self.advance().value
            if val == "true":
                return True
            elif val == "false":
                return False
            elif val == "null":
                return None
            return val  # enum
        elif tok.type == "DOLLAR":
            self.advance()
            name = self.expect("NAME").value
            return {"$var": name}
        elif tok.type == "LBRACKET":
            self.advance()
            items = []
            while True:
                if self.match("RBRACKET"):
                    break
                items.append(self._parse_value())
                if not self.match("COMMA"):
                    # Expect RBRACKET next
                    self.expect("RBRACKET")
                    break
            return items
        elif tok.type == "LBRACE":
            self.advance()
            obj = {}
            while True:
                if self.match("RBRACE"):
                    break
                key = self.expect("NAME").value
                self.expect("COLON")
                val = self._parse_value()
                obj[key] = val
                if not self.match("COMMA"):
                    # Next must be RBRACE
                    self.expect("RBRACE")
                    break
            return obj
        raise ValueError(f"Unexpected value token: {tok.type}")


# ============================================================
# Schema and Type Definitions
# ============================================================

SCALAR_TYPES = {"String", "Int", "Boolean", "ID", "Float"}

ENUM_TYPES = {
    "IssueState": ["OPEN", "CLOSED"],
    "PullRequestState": ["OPEN", "CLOSED", "MERGED"],
    "PullRequestMergeMethod": ["MERGE", "SQUASH", "REBASE"],
    "OrderDirection": ["ASC", "DESC"],
    "IssueOrderField": ["CREATED_AT", "UPDATED_AT", "COMMENTS"],
    "PullRequestOrderField": ["CREATED_AT", "UPDATED_AT"],
    "SearchType": ["ISSUE", "REPOSITORY", "USER", "DISCUSSION"],
    "MergeableState": ["MERGEABLE", "CONFLICTING", "UNKNOWN"],
}

INPUT_OBJECT_TYPES = {
    "CreateIssueInput": {
        "repositoryId": "ID",
        "title": "String",
        "body": "String",
        "clientMutationId": "String",
    },
    "CloseIssueInput": {
        "id": "ID",
        "clientMutationId": "String",
    },
    "CreatePullRequestInput": {
        "repositoryId": "ID",
        "baseRefName": "String",
        "headRefName": "String",
        "title": "String",
        "body": "String",
        "clientMutationId": "String",
    },
    "MergePullRequestInput": {
        "pullRequestId": "ID",
        "method": "PullRequestMergeMethod",
        "clientMutationId": "String",
    },
    "AddCommentInput": {
        "subjectId": "ID",
        "body": "String",
        "clientMutationId": "String",
    },
}

OBJECT_TYPES = {
    "Query": {
        "viewer": "User",
        "repository": "Repository",
        "node": "Node",
        "__type": "__Type",
        "__schema": "__Schema",
    },
    "Mutation": {
        "createIssue": "CreateIssuePayload",
        "closeIssue": "CloseIssuePayload",
        "createPullRequest": "CreatePullRequestPayload",
        "mergePullRequest": "MergePullRequestPayload",
        "addComment": "AddCommentPayload",
    },
    "User": {
        "id": "ID",
        "login": "String",
        "name": "String",
        "email": "String",
        "url": "String",
    },
    "Repository": {
        "id": "ID",
        "databaseId": "Int",
        "name": "String",
        "nameWithOwner": "String",
        "description": "String",
        "isPrivate": "Boolean",
        "owner": "User",
        "defaultBranchRef": "Ref",
        "viewerPermission": "String",
        "hasIssuesEnabled": "Boolean",
        "hasWikiEnabled": "Boolean",
        "url": "String",
        "createdAt": "String",
        "updatedAt": "String",
        "issue": "Issue",
        "issues": "IssueConnection",
        "pullRequest": "PullRequest",
        "pullRequests": "PullRequestConnection",
    },
    "Ref": {
        "name": "String",
        "prefix": "String",
        "target": "GitObject",
    },
    "GitObject": {
        "id": "ID",
        "oid": "String",
        "abbreviatedOid": "String",
    },
    "Issue": {
        "id": "ID",
        "databaseId": "Int",
        "number": "Int",
        "title": "String",
        "body": "String",
        "state": "IssueState",
        "author": "User",
        "repository": "Repository",
        "labels": "LabelConnection",
        "comments": "IssueCommentConnection",
        "createdAt": "String",
        "updatedAt": "String",
        "closedAt": "String",
        "url": "String",
        "locked": "Boolean",
    },
    "Label": {
        "id": "ID",
        "name": "String",
        "color": "String",
    },
    "LabelConnection": {
        "nodes": "[Label]",
        "pageInfo": "PageInfo",
        "totalCount": "Int",
    },
    "IssueComment": {
        "id": "ID",
        "body": "String",
        "author": "User",
        "createdAt": "String",
        "updatedAt": "String",
    },
    "IssueCommentConnection": {
        "nodes": "[IssueComment]",
        "pageInfo": "PageInfo",
        "totalCount": "Int",
    },
    "IssueCommentEdge": {
        "node": "IssueComment",
        "cursor": "String",
    },
    "PullRequest": {
        "id": "ID",
        "databaseId": "Int",
        "number": "Int",
        "title": "String",
        "body": "String",
        "state": "PullRequestState",
        "merged": "Boolean",
        "mergedAt": "String",
        "mergeCommit": "GitObject",
        "mergeable": "MergeableState",
        "headRefName": "String",
        "baseRefName": "String",
        "headRef": "Ref",
        "baseRef": "Ref",
        "author": "User",
        "repository": "Repository",
        "files": "PullRequestChangedFileConnection",
        "additions": "Int",
        "deletions": "Int",
        "changedFiles": "Int",
        "createdAt": "String",
        "updatedAt": "String",
        "url": "String",
    },
    "PullRequestChangedFile": {
        "path": "String",
        "additions": "Int",
        "deletions": "Int",
    },
    "PullRequestChangedFileConnection": {
        "nodes": "[PullRequestChangedFile]",
        "pageInfo": "PageInfo",
        "totalCount": "Int",
    },
    "PageInfo": {
        "hasNextPage": "Boolean",
        "hasPreviousPage": "Boolean",
        "startCursor": "String",
        "endCursor": "String",
    },
    "CreateIssuePayload": {
        "issue": "Issue",
        "clientMutationId": "String",
    },
    "CloseIssuePayload": {
        "issue": "Issue",
        "clientMutationId": "String",
    },
    "CreatePullRequestPayload": {
        "pullRequest": "PullRequest",
        "clientMutationId": "String",
    },
    "MergePullRequestPayload": {
        "pullRequest": "PullRequest",
        "clientMutationId": "String",
    },
    "AddCommentPayload": {
        "commentEdge": "IssueCommentEdge",
        "clientMutationId": "String",
    },
}

# Connection element type mapping
CONNECTION_ELEMENT = {
    "IssueConnection": "Issue",
    "IssueCommentConnection": "IssueComment",
    "PullRequestConnection": "PullRequest",
    "PullRequestChangedFileConnection": "PullRequestChangedFile",
    "LabelConnection": "Label",
}

# Payload fields
PAYLOAD_FIELDS = {
    "CreateIssuePayload": {"issue": "Issue", "clientMutationId": "String"},
    "CloseIssuePayload": {"issue": "Issue", "clientMutationId": "String"},
    "CreatePullRequestPayload": {"pullRequest": "PullRequest", "clientMutationId": "String"},
    "MergePullRequestPayload": {"pullRequest": "PullRequest", "clientMutationId": "String"},
    "AddCommentPayload": {"commentEdge": "IssueCommentEdge", "clientMutationId": "String"},
}


# ============================================================
# Validation
# ============================================================

def validate_selection(
    type_name: str,
    selection_set: List[Selection],
    fragments: Dict[str, FragmentDefinition],
    path: str = "",
) -> Optional[str]:
    """Validate a selection set against a type. Returns error message or None."""
    if type_name in SCALAR_TYPES or type_name in ENUM_TYPES:
        return None  # No selection set on scalars
    
    if type_name not in OBJECT_TYPES:
        # Might be an interface/union - skip validation
        return None
    
    fields = OBJECT_TYPES.get(type_name, {})
    
    i = 0
    n = len(selection_set)
    while i < n:
        sel = selection_set[i]
        i += 1
        if sel.name == "__typename":
            continue
        if sel.name.startswith("__"):
            continue
        # Inline fragment marker "...on": the NEXT sibling is the fragment body,
        # named after the type condition; skip it too (not a real field).
        if sel.name == "...on" or sel.name == "... on":
            if i < n:
                i += 1  # skip the type-named fragment sibling
            continue
        # Skip fragment spreads and named inline fragments (they start with ...)
        if sel.name.startswith("..."):
            continue
        
        field_type = fields.get(sel.name)
        if field_type is None:
            return f"Field '{sel.name}' doesn't exist on type '{type_name}'"
        
        # Unwrap list and non-null
        base_type = field_type
        while base_type.startswith("[") or base_type.endswith("!"):
            if base_type.startswith("["):
                base_type = base_type[1:-1]
            if base_type.endswith("!"):
                base_type = base_type[:-1]
        
        if sel.selection_set:
            err = validate_selection(base_type, sel.selection_set, fragments, f"{path}.{sel.name}")
            if err:
                return err
        
        # Check fragment spreads
        for frag_name in [d.name for d in sel.directives if d.name == "inline" and sel.name == "..."]:
            pass  # Not implemented
    
    return None


def validate_operation(op: OperationDefinition, fragments: Dict[str, FragmentDefinition]) -> Optional[str]:
    """Validate a single operation."""
    # Determine root type
    root_type = "Query" if op.type == "query" else "Mutation"
    return validate_selection(root_type, op.selection_set, fragments)


# ============================================================
# Execution Context
# ============================================================

@dataclass
class ExecContext:
    store: Store
    login: str
    variables: Dict[str, Any]
    fragments: Dict[str, FragmentDefinition]


# Field type mapping for recursion
FIELD_TYPE_MAP = {
    "Query": {
        "viewer": "User",
        "repository": "Repository",
        "node": "Node",
    },
    "Mutation": {
        "createIssue": "CreateIssuePayload",
        "closeIssue": "CloseIssuePayload",
        "createPullRequest": "CreatePullRequestPayload",
        "mergePullRequest": "MergePullRequestPayload",
        "addComment": "AddCommentPayload",
    },
    "Repository": {
        "owner": "User",
        "defaultBranchRef": "Ref",
        "issue": "Issue",
        "issues": "IssueConnection",
        "pullRequest": "PullRequest",
        "pullRequests": "PullRequestConnection",
    },
    "Issue": {
        "author": "User",
        "repository": "Repository",
        "labels": "LabelConnection",
        "comments": "IssueCommentConnection",
    },
    "PullRequest": {
        "author": "User",
        "repository": "Repository",
        "mergeCommit": "GitObject",
        "headRef": "Ref",
        "baseRef": "Ref",
        "files": "PullRequestChangedFileConnection",
    },
    "Ref": {
        "target": "GitObject",
    },
    "LabelConnection": {"nodes": "Label", "edges": "Label"},
    "IssueConnection": {"nodes": "Issue", "edges": "Issue"},
    "IssueCommentConnection": {"nodes": "IssueComment", "edges": "IssueComment"},
    "PullRequestConnection": {"nodes": "PullRequest", "edges": "PullRequest"},
    "PullRequestChangedFileConnection": {"nodes": "PullRequestChangedFile", "edges": "PullRequestChangedFile"},
    "CreateIssuePayload": {"issue": "Issue"},
    "CloseIssuePayload": {"issue": "Issue"},
    "CreatePullRequestPayload": {"pullRequest": "PullRequest"},
    "MergePullRequestPayload": {"pullRequest": "PullRequest"},
    "AddCommentPayload": {"commentEdge": "IssueCommentEdge"},
    "IssueCommentEdge": {"node": "IssueComment"},
}


def get_field_type(parent_type: str, field_name: str) -> Optional[str]:
    """Get the type of a field for recursion."""
    if parent_type in FIELD_TYPE_MAP:
        return FIELD_TYPE_MAP[parent_type].get(field_name)
    return None


def execute(context: ExecContext, type_name: str, payload: Any, selection_set: List[Selection], depth: int = 0) -> Dict[str, Any]:
    """Execute a selection set."""
    if depth > 50:
        return {"__error": "max depth exceeded"}
    result = {}
    
    i = 0
    n = len(selection_set)
    while i < n:
        sel = selection_set[i]
        i += 1
        
        if sel.name == "__typename":
            result[sel.alias or "__typename"] = type_name
            continue
        
        if sel.name.startswith("__"):
            if sel.name == "__type":
                # Resolve arguments from the selection, like resolve_field does
                sel_args = {arg.name: resolve_value(context, arg.value) for arg in sel.args}
                type_name_arg = sel_args.get("name")
                if type_name_arg:
                    result[sel.alias or sel.name] = type_object(type_name_arg)
                else:
                    result[sel.alias or sel.name] = None
            elif sel.name == "__schema":
                result[sel.alias or sel.name] = schema_object()
            continue
        
        # Inline fragment marker "...on": the NEXT sibling is the fragment body,
        # named after the type condition. If its name matches the concrete type
        # of the current payload, execute its selection set and merge; otherwise
        # it contributes nothing. The marker itself contributes nothing.
        if sel.name == "...on" or sel.name == "... on":
            if i < n:
                frag = selection_set[i]
                concrete = _concrete_type_from_payload(payload, type_name)
                if frag.name == concrete and frag.selection_set:
                    inline_result = execute(context, concrete, payload, frag.selection_set, depth + 1)
                    for k, v in inline_result.items():
                        result[k] = v
                i += 1  # consume the fragment sibling
            continue
        
        # Named fragment spread: ...FragmentName
        if sel.name.startswith("..."):
            fragment_name = sel.name[3:]  # Remove "..."
            if fragment_name in context.fragments:
                fragment = context.fragments[fragment_name]
                fragment_result = execute(context, fragment.type_condition, payload, fragment.selection_set, depth + 1)
                for k, v in fragment_result.items():
                    result[k] = v
            continue
        
        # Resolve field
        value = resolve_field(context, type_name, payload, sel)
        
        # Recurse into sub-selection if present and value is not None and not a list
        # Lists indicate the sub-selection has already been applied (e.g., connection nodes)
        if sel.selection_set and value is not None and not isinstance(value, list):
            field_type = get_field_type(type_name, sel.name)
            if field_type:
                # If the field resolves to the abstract "Node" type, determine the concrete type
                # from the payload value so inline fragments and plain fields resolve correctly.
                if field_type == "Node":
                    field_type = _concrete_type_from_payload(value, field_type)
                value = execute(context, field_type, value, sel.selection_set, depth + 1)
        
        key = sel.alias or sel.name
        result[key] = value
    
    return result


def _concrete_type_from_payload(payload: Any, default_type: str) -> str:
    """Determine the concrete GraphQL type from a Node payload value."""
    if isinstance(payload, Repo):
        return "Repository"
    if isinstance(payload, tuple) and len(payload) == 2:
        # (repo, issue) or (repo, pull)
        if isinstance(payload[1], dict):
            if "number" in payload[1] and "head_ref" in payload[1]:
                return "PullRequest"
            if "number" in payload[1]:
                return "Issue"
    return default_type


def resolve_field(context: ExecContext, type_name: str, payload: Any, sel: Selection) -> Any:
    args = {arg.name: resolve_value(context, arg.value) for arg in sel.args}
    
    # Handle connections
    if type_name in CONNECTION_ELEMENT:
        return resolve_connection(context, type_name, payload, sel, args)
    
    # Handle payload objects
    if type_name in PAYLOAD_FIELDS:
        return resolve_payload(context, type_name, payload, sel, args)
    
    # Object type handlers
    handlers = {
        "Query": resolve_query,
        "Mutation": resolve_mutation,
        "User": resolve_user,
        "Repository": resolve_repository,
        "Issue": resolve_issue,
        "PullRequest": resolve_pull_request,
        "Ref": resolve_ref,
        "GitObject": resolve_git_object,
        "Label": resolve_label,
        "IssueComment": resolve_issue_comment,
        "PullRequestChangedFile": resolve_changed_file,
    }
    
    handler = handlers.get(type_name)
    if handler:
        return handler(context, payload, sel, args)
    
    return None


def resolve_value(context: ExecContext, value: Any) -> Any:
    """Resolve a value (handle variables)."""
    if isinstance(value, dict) and "$var" in value:
        return context.variables.get(value["$var"])
    if isinstance(value, list):
        return [resolve_value(context, v) for v in value]
    if isinstance(value, dict):
        return {k: resolve_value(context, v) for k, v in value.items()}
    return value


def resolve_query(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    if sel.name == "viewer":
        return context.login
    elif sel.name == "repository":
        owner = args.get("owner")
        name = args.get("name")
        if owner and name:
            return context.store.get_repo(owner, name)
        return None
    elif sel.name == "node":
        id_val = args.get("id")
        if id_val:
            decoded = decode_gid(id_val)
            if decoded:
                kind, key = decoded
                if kind == "Repository":
                    for repo in context.store.repos.values():
                        if str(repo.db_id) == key:
                            return repo
                elif kind in ("Issue", "PullRequest"):
                    parts = key.split(":")
                    if len(parts) == 2:
                        db_id, num = parts
                        for repo in context.store.repos.values():
                            if str(repo.db_id) == db_id:
                                if kind == "Issue":
                                    if int(num) in repo.issues:
                                        return (repo, repo.issues[int(num)])
                                    return None
                                else:
                                    if int(num) in repo.pulls:
                                        return (repo, repo.pulls[int(num)])
                                    return None
        return None
    return None


def resolve_mutation(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    input_val = args.get("input", {})
    
    if sel.name == "createIssue":
        repo_id = input_val.get("repositoryId")
        repo = find_repo_by_gid(context.store, repo_id)
        if not repo:
            return {"issue": None, "clientMutationId": input_val.get("clientMutationId")}
        
        title = input_val.get("title", "").strip()
        if not title:
            return {"issue": None, "clientMutationId": input_val.get("clientMutationId")}
        
        ts = context.store.clock.tick()
        number = repo.new_number()
        issue = {
            "number": number,
            "db_id": context.store.new_issue_db(),
            "title": title,
            "body": input_val.get("body", "") or "",
            "state": "open",
            "labels": [],
            "user": context.login,
            "created_at": ts,
            "updated_at": ts,
            "closed_at": None,
            "comments": [],
        }
        repo.issues[number] = issue
        return {
            "issue": (repo, issue),
            "clientMutationId": input_val.get("clientMutationId"),
        }
    
    elif sel.name == "closeIssue":
        issue_id = input_val.get("issueId")
        if issue_id is None:
            issue_id = input_val.get("id")
        repo, issue = find_issue_by_gid(context.store, issue_id)
        if repo and issue:
            issue["state"] = "closed"
            issue["updated_at"] = context.store.clock.tick()
            issue["closed_at"] = issue["updated_at"]
        return {
            "issue": (repo, issue) if repo and issue else None,
            "clientMutationId": input_val.get("clientMutationId"),
        }
    
    elif sel.name == "createPullRequest":
        repo_id = input_val.get("repositoryId")
        repo = find_repo_by_gid(context.store, repo_id)
        if not repo:
            return {"pullRequest": None, "clientMutationId": input_val.get("clientMutationId")}
        
        head_ref = input_val.get("headRefName")
        base_ref = input_val.get("baseRefName")
        if not head_ref or not base_ref:
            return {"pullRequest": None, "clientMutationId": input_val.get("clientMutationId")}
        
        head_sha = repo.tip(f"refs/heads/{head_ref}")
        base_sha = repo.tip(f"refs/heads/{base_ref}")
        if not head_sha or not base_sha:
            return {"pullRequest": None, "clientMutationId": input_val.get("clientMutationId")}
        
        title = input_val.get("title", "").strip()
        if not title:
            return {"pullRequest": None, "clientMutationId": input_val.get("clientMutationId")}
        
        ts = context.store.clock.tick()
        number = repo.new_number()
        pull = {
            "number": number,
            "db_id": context.store.new_pull_db(),
            "title": title,
            "body": input_val.get("body", "") or "",
            "state": "open",
            "head_ref": head_ref,
            "base_ref": base_ref,
            "head_sha": head_sha,
            "base_sha": base_sha,
            "merged": False,
            "merged_at": None,
            "merge_commit_sha": None,
            "user": context.login,
            "created_at": ts,
            "updated_at": ts,
        }
        repo.pulls[number] = pull
        return {
            "pullRequest": (repo, pull),
            "clientMutationId": input_val.get("clientMutationId"),
        }
    
    elif sel.name == "mergePullRequest":
        pull_id = input_val.get("pullRequestId")
        repo, pull = find_pull_by_gid(context.store, pull_id)
        if not repo or not pull:
            return {"pullRequest": None, "clientMutationId": input_val.get("clientMutationId")}
        
        ok, result = merge_pull(context.store, repo, pull, context.login)
        if not ok:
            return {"pullRequest": None, "clientMutationId": input_val.get("clientMutationId")}
        
        return {
            "pullRequest": (repo, pull),
            "clientMutationId": input_val.get("clientMutationId"),
        }
    
    elif sel.name == "addComment":
        subject_id = input_val.get("subjectId")
        repo, issue = find_issue_by_gid(context.store, subject_id)
        if not repo or not issue:
            return {"commentEdge": None, "clientMutationId": input_val.get("clientMutationId")}
        
        ts = context.store.clock.tick()
        comment = {
            "id": context.store.new_comment_id(),
            "body": input_val.get("body", ""),
            "user": context.login,
            "created_at": ts,
            "updated_at": ts,
        }
        issue.setdefault("comments", []).append(comment)
        return {
            "commentEdge": {
                "node": (repo, comment),
                "cursor": base64.b64encode(f"c:{comment['id']}".encode()).decode(),
            },
            "clientMutationId": input_val.get("clientMutationId"),
        }
    
    return None


def find_repo_by_gid(store: Store, gid_str: str) -> Optional[Repo]:
    if not gid_str:
        return None
    decoded = decode_gid(gid_str)
    if not decoded:
        return None
    kind, key = decoded
    if kind == "Repository":
        for repo in store.repos.values():
            if str(repo.db_id) == key:
                return repo
    return None


def find_issue_by_gid(store: Store, gid_str: str) -> Tuple[Optional[Repo], Optional[Dict]]:
    if not gid_str:
        return None, None
    decoded = decode_gid(gid_str)
    if not decoded:
        return None, None
    kind, key = decoded
    if kind != "Issue":
        return None, None
    parts = key.split(":")
    if len(parts) != 2:
        return None, None
    db_id, num = parts
    for repo in store.repos.values():
        if str(repo.db_id) == db_id:
            return repo, repo.issues.get(int(num))
    return None, None


def find_pull_by_gid(store: Store, gid_str: str) -> Tuple[Optional[Repo], Optional[Dict]]:
    if not gid_str:
        return None, None
    decoded = decode_gid(gid_str)
    if not decoded:
        return None, None
    kind, key = decoded
    if kind != "PullRequest":
        return None, None
    parts = key.split(":")
    if len(parts) != 2:
        return None, None
    db_id, num = parts
    for repo in store.repos.values():
        if str(repo.db_id) == db_id:
            return repo, repo.pulls.get(int(num))
    return None, None


def resolve_user(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    login = payload
    if sel.name == "id":
        return gid("User", login)
    elif sel.name == "login":
        return login
    elif sel.name == "name":
        return login
    elif sel.name == "email":
        return f"{login}@example.invalid"
    elif sel.name == "url":
        return f"/{login}"
    return None


def resolve_repository(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    repo: Repo = payload
    if sel.name == "id":
        return gid("Repository", str(repo.db_id))
    elif sel.name == "databaseId":
        return repo.db_id
    elif sel.name == "name":
        return repo.name
    elif sel.name == "nameWithOwner":
        return repo.full_name
    elif sel.name == "description":
        return repo.description or ""
    elif sel.name == "isPrivate":
        return repo.private
    elif sel.name == "owner":
        return repo.owner
    elif sel.name == "defaultBranchRef":
        tip = repo.tip(f"refs/heads/{repo.default_branch}")
        if tip:
            return (repo.default_branch, tip)
        return None
    elif sel.name == "viewerPermission":
        return "ADMIN"
    elif sel.name == "hasIssuesEnabled":
        return True
    elif sel.name == "hasWikiEnabled":
        return False
    elif sel.name == "url":
        return f"/{repo.full_name}"
    elif sel.name == "createdAt":
        return iso_utc(repo.created_at)
    elif sel.name == "updatedAt":
        return iso_utc(repo.updated_at)
    elif sel.name == "issue":
        num = args.get("number")
        if num:
            return (repo, repo.issues.get(int(num)))
        return None
    elif sel.name == "issues":
        return resolve_issues_connection(context, repo, args)
    elif sel.name == "pullRequest":
        num = args.get("number")
        if num:
            return (repo, repo.pulls.get(int(num)))
        return None
    elif sel.name == "pullRequests":
        return resolve_pull_requests_connection(context, repo, args)
    return None


def resolve_issues_connection(context: ExecContext, repo: Repo, args: Dict) -> Dict:
    states = args.get("states", [])
    order_by = args.get("orderBy", {})
    field = order_by.get("field", "CREATED_AT")
    direction = order_by.get("direction", "DESC")
    
    # Filter
    issues = list(repo.issues.values())
    if states:
        filtered = []
        for state in states:
            if state == "OPEN":
                filtered.extend([i for i in issues if i["state"] == "open"])
            elif state == "CLOSED":
                filtered.extend([i for i in issues if i["state"] == "closed"])
        issues = filtered
    
    # Sort
    if field == "CREATED_AT":
        issues.sort(key=lambda x: (x["created_at"], x["number"]))
    elif field == "UPDATED_AT":
        issues.sort(key=lambda x: (x["updated_at"], x["number"]))
    elif field == "COMMENTS":
        issues.sort(key=lambda x: (len(x["comments"]), x["number"]))
    
    if direction == "DESC":
        issues.reverse()
    
    total = len(issues)
    
    # Pagination
    after = args.get("after")
    first = args.get("first", 10)
    
    start = 0
    if after:
        try:
            cursor_val = base64.b64decode(after).decode()
            if cursor_val.startswith("c:"):
                cursor_num = int(cursor_val[2:])
                for i, iss in enumerate(issues):
                    if iss["number"] == cursor_num:
                        start = i + 1
                        break
        except Exception:
            pass
    
    end = min(start + first, total)
    page_items = issues[start:end]
    
    # Build edges and pageInfo
    edges = []
    for iss in page_items:
        cursor = base64.b64encode(f"c:{iss['number']}".encode()).decode()
        edges.append({
            "node": (repo, iss),
            "cursor": cursor,
        })
    
    page_info = {
        "hasNextPage": end < total,
        "hasPreviousPage": start > 0,
        "startCursor": edges[0]["cursor"] if edges else None,
        "endCursor": edges[-1]["cursor"] if edges else None,
    }
    
    return {
        "nodes": [(repo, iss) for iss in page_items],
        "edges": edges,
        "pageInfo": page_info,
        "totalCount": total,
    }


def resolve_pull_requests_connection(context: ExecContext, repo: Repo, args: Dict) -> Dict:
    states = args.get("states", [])
    order_by = args.get("orderBy", {})
    field = order_by.get("field", "CREATED_AT")
    direction = order_by.get("direction", "DESC")
    
    pulls = list(repo.pulls.values())
    if states:
        filtered = []
        for state in states:
            if state == "OPEN":
                filtered.extend([p for p in pulls if p["state"] == "open" and not p.get("merged", False)])
            elif state == "CLOSED":
                filtered.extend([p for p in pulls if p["state"] == "closed" and not p.get("merged", False)])
            elif state == "MERGED":
                filtered.extend([p for p in pulls if p.get("merged", False)])
        pulls = filtered
    
    if field == "CREATED_AT":
        pulls.sort(key=lambda x: (x["created_at"], x["number"]))
    elif field == "UPDATED_AT":
        pulls.sort(key=lambda x: (x["updated_at"], x["number"]))
    
    if direction == "DESC":
        pulls.reverse()
    
    total = len(pulls)
    
    after = args.get("after")
    first = args.get("first", 10)
    
    start = 0
    if after:
        try:
            cursor_val = base64.b64decode(after).decode()
            if cursor_val.startswith("c:"):
                cursor_num = int(cursor_val[2:])
                for i, pr in enumerate(pulls):
                    if pr["number"] == cursor_num:
                        start = i + 1
                        break
        except Exception:
            pass
    
    end = min(start + first, total)
    page_items = pulls[start:end]
    
    edges = []
    for pr in page_items:
        cursor = base64.b64encode(f"c:{pr['number']}".encode()).decode()
        edges.append({
            "node": (repo, pr),
            "cursor": cursor,
        })
    
    page_info = {
        "hasNextPage": end < total,
        "hasPreviousPage": start > 0,
        "startCursor": edges[0]["cursor"] if edges else None,
        "endCursor": edges[-1]["cursor"] if edges else None,
    }
    
    return {
        "nodes": [(repo, pr) for pr in page_items],
        "edges": edges,
        "pageInfo": page_info,
        "totalCount": total,
    }


def resolve_connection(context: ExecContext, type_name: str, payload: Dict, sel: Selection, args: Dict) -> Any:
    element_type = CONNECTION_ELEMENT.get(type_name)
    if not element_type:
        return None
    
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    page_info = payload.get("pageInfo", {})
    total = payload.get("totalCount", 0)
    
    if sel.name == "nodes":
        if sel.selection_set:
            return [execute(context, element_type, node, sel.selection_set) for node in nodes]
        return nodes
    elif sel.name == "edges":
        if sel.selection_set:
            return [execute(context, element_type, {"node": e["node"], "cursor": e["cursor"]}, sel.selection_set) for e in edges]
        return edges
    elif sel.name == "pageInfo":
        return page_info
    elif sel.name == "totalCount":
        return total
    return None


def resolve_changed_file(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    if isinstance(payload, dict):
        return payload.get(sel.name)
    return None


def resolve_payload(context: ExecContext, type_name: str, payload: Dict, sel: Selection, args: Dict) -> Any:
    if sel.name == "clientMutationId":
        return payload.get("clientMutationId")
    
    # Other payload fields - return the raw value, let execute handle recursion
    field = sel.name
    if field in payload:
        return payload[field]
    return None


def resolve_issue(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    # Handle both tuple and list payloads
    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        repo, issue = payload[0], payload[1]
    elif isinstance(payload, dict):
        # Debug: payload is a dict, try to extract repo and issue
        # This happens when the tuple is converted to dict somewhere
        if 'repo' in payload and 'issue' in payload:
            repo, issue = payload['repo'], payload['issue']
        elif len(payload) >= 2:
            items = list(payload.items())
            repo, issue = items[0][1], items[1][1]
        else:
            return f"DEBUG: payload type=dict, keys={list(payload.keys())}"
    else:
        return f"DEBUG: payload type={type(payload).__name__}, len={len(payload) if hasattr(payload, '__len__') else 'N/A'}"
    if sel.name == "id":
        return gid("Issue", f"{repo.db_id}:{issue['number']}")
    elif sel.name == "databaseId":
        return issue["db_id"]
    elif sel.name == "number":
        return issue["number"]
    elif sel.name == "title":
        return issue["title"]
    elif sel.name == "body":
        return issue["body"] or ""
    elif sel.name == "state":
        return "OPEN" if issue["state"] == "open" else "CLOSED"
    elif sel.name == "author":
        return issue["user"]
    elif sel.name == "repository":
        return repo
    elif sel.name == "labels":
        label_names = [n for n in issue["labels"] if n in repo.labels]
        label_objs = [repo.labels[n] for n in label_names]
        return {
            "nodes": [(repo, l) for l in label_objs],
            "edges": [{"node": (repo, l), "cursor": base64.b64encode(f"c:{l['id']}".encode()).decode()} for l in label_objs],
            "pageInfo": {"hasNextPage": False, "hasPreviousPage": False, "startCursor": None, "endCursor": None},
            "totalCount": len(label_objs),
        }
    elif sel.name == "comments":
        comments = issue.get("comments", [])
        return {
            "nodes": [(repo, c) for c in comments],
            "edges": [{"node": (repo, c), "cursor": base64.b64encode(f"c:{c['id']}".encode()).decode()} for c in comments],
            "pageInfo": {"hasNextPage": False, "hasPreviousPage": False, "startCursor": None, "endCursor": None},
            "totalCount": len(comments),
        }
    elif sel.name == "createdAt":
        return iso_utc(issue["created_at"])
    elif sel.name == "updatedAt":
        return iso_utc(issue["updated_at"])
    elif sel.name == "closedAt":
        if issue.get("closed_at"):
            return iso_utc(issue["closed_at"])
        return None
    elif sel.name == "url":
        return f"/{repo.full_name}/issues/{issue['number']}"
    elif sel.name == "locked":
        return False
    return None


def resolve_pull_request(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    repo, pull = payload
    if sel.name == "id":
        return gid("PullRequest", f"{repo.db_id}:{pull['number']}")
    elif sel.name == "databaseId":
        return pull["db_id"]
    elif sel.name == "number":
        return pull["number"]
    elif sel.name == "title":
        return pull["title"]
    elif sel.name == "body":
        return pull["body"] or ""
    elif sel.name == "state":
        if pull.get("merged", False):
            return "MERGED"
        elif pull["state"] == "closed":
            return "CLOSED"
        else:
            return "OPEN"
    elif sel.name == "merged":
        return pull.get("merged", False)
    elif sel.name == "mergedAt":
        if pull.get("merged_at"):
            return iso_utc(pull["merged_at"])
        return None
    elif sel.name == "mergeCommit":
        merge_sha = pull.get("merge_commit_sha")
        if merge_sha:
            return merge_sha
        return None
    elif sel.name == "mergeable":
        if pull.get("merged", False):
            return "MERGEABLE"
        # Check if there are conflicts
        base_sha = pull.get("base_sha")
        head_sha = pull.get("head_sha")
        if base_sha and head_sha and repo.objects.has(base_sha) and repo.objects.has(head_sha):
            _, base_data = repo.objects.get(base_sha)
            base_parsed = parse_commit(base_data)
            base_tree = base_parsed["tree"]
            
            _, head_data = repo.objects.get(head_sha)
            head_parsed = parse_commit(head_data)
            head_tree = head_parsed["tree"]
            
            diff = diff_trees(repo.objects, base_tree, head_tree)
            # Check if any file changed on both sides
            for path, (old_sha, new_sha) in diff.items():
                if old_sha and new_sha and old_sha != new_sha:
                    # Check if both sides have changes
                    # Simplified: if both old and new exist and differ, could be conflict
                    pass
        return "MERGEABLE"
    elif sel.name == "headRefName":
        return pull["head_ref"]
    elif sel.name == "baseRefName":
        return pull["base_ref"]
    elif sel.name == "headRef":
        tip = repo.tip(f"refs/heads/{pull['head_ref']}")
        if tip:
            return (pull["head_ref"], tip)
        return None
    elif sel.name == "baseRef":
        tip = repo.tip(f"refs/heads/{pull['base_ref']}")
        if tip:
            return (pull["base_ref"], tip)
        return None
    elif sel.name == "author":
        return pull["user"]
    elif sel.name == "repository":
        return repo
    elif sel.name == "files":
        return resolve_pr_files(context, repo, pull)
    elif sel.name == "additions":
        files = resolve_pr_files(context, repo, pull)
        return sum(f.get("additions", 0) for f in files.get("nodes", []))
    elif sel.name == "deletions":
        files = resolve_pr_files(context, repo, pull)
        return sum(f.get("deletions", 0) for f in files.get("nodes", []))
    elif sel.name == "changedFiles":
        files = resolve_pr_files(context, repo, pull)
        return len(files.get("nodes", []))
    elif sel.name == "createdAt":
        return iso_utc(pull["created_at"])
    elif sel.name == "updatedAt":
        return iso_utc(pull["updated_at"])
    elif sel.name == "url":
        return f"/{repo.full_name}/pulls/{pull['number']}"
    return None


def resolve_pr_files(context: ExecContext, repo: Repo, pull: Dict) -> Dict:
    base_sha = pull.get("base_sha")
    head_sha = pull.get("head_sha")
    if not base_sha or not head_sha:
        return {"nodes": [], "edges": [], "pageInfo": {}, "totalCount": 0}
    
    if not repo.objects.has(base_sha) or not repo.objects.has(head_sha):
        return {"nodes": [], "edges": [], "pageInfo": {}, "totalCount": 0}
    
    _, base_data = repo.objects.get(base_sha)
    base_parsed = parse_commit(base_data)
    base_tree = base_parsed["tree"]
    
    _, head_data = repo.objects.get(head_sha)
    head_parsed = parse_commit(head_data)
    head_tree = head_parsed["tree"]
    
    diff = diff_trees(repo.objects, base_tree, head_tree)
    
    files = []
    for path, (old_sha, new_sha) in sorted(diff.items()):
        if old_sha is None:
            # Added
            additions = 0
            if new_sha and repo.objects.has(new_sha):
                _, blob = repo.objects.get(new_sha)
                additions = len(blob.decode("utf-8", errors="replace").splitlines())
            files.append({
                "path": path,
                "additions": additions,
                "deletions": 0,
            })
        elif new_sha is None:
            # Removed
            deletions = 0
            if old_sha and repo.objects.has(old_sha):
                _, blob = repo.objects.get(old_sha)
                deletions = len(blob.decode("utf-8", errors="replace").splitlines())
            files.append({
                "path": path,
                "additions": 0,
                "deletions": deletions,
            })
        else:
            # Modified
            additions = 0
            deletions = 0
            if old_sha and repo.objects.has(old_sha):
                _, old_blob = repo.objects.get(old_sha)
                old_lines = old_blob.decode("utf-8", errors="replace").splitlines()
            else:
                old_lines = []
            if new_sha and repo.objects.has(new_sha):
                _, new_blob = repo.objects.get(new_sha)
                new_lines = new_blob.decode("utf-8", errors="replace").splitlines()
            else:
                new_lines = []
            
            import difflib
            matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "insert":
                    additions += j2 - j1
                elif tag == "delete":
                    deletions += i2 - i1
                elif tag == "replace":
                    additions += j2 - j1
                    deletions += i2 - i1
            
            files.append({
                "path": path,
                "additions": additions,
                "deletions": deletions,
            })
    
    return {
        "nodes": files,
        "edges": [{"node": f, "cursor": base64.b64encode(f"c:{i}".encode()).decode()} for i, f in enumerate(files)],
        "pageInfo": {"hasNextPage": False, "hasPreviousPage": False, "startCursor": None, "endCursor": None},
        "totalCount": len(files),
    }


def resolve_ref(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    name, sha = payload
    if sel.name == "name":
        return name
    elif sel.name == "prefix":
        return "refs/heads/"
    elif sel.name == "target":
        return sha
    return None


def resolve_git_object(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    sha = payload
    if sel.name == "id":
        return gid("Commit", sha)
    elif sel.name == "oid":
        return sha
    elif sel.name == "abbreviatedOid":
        return sha[:7]
    return None


def resolve_label(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    repo, label = payload
    if sel.name == "id":
        return gid("Label", f"{repo.db_id}:{label['id']}")
    elif sel.name == "name":
        return label["name"]
    elif sel.name == "color":
        return label["color"]
    return None


def resolve_issue_comment(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    repo, comment = payload
    if sel.name == "id":
        return gid("IssueComment", str(comment["id"]))
    elif sel.name == "body":
        return comment["body"]
    elif sel.name == "author":
        return comment["user"]
    elif sel.name == "createdAt":
        return iso_utc(comment["created_at"])
    elif sel.name == "updatedAt":
        return iso_utc(comment["updated_at"])
    return None


def resolve_issue_comment_edge(context: ExecContext, payload: Any, sel: Selection, args: Dict) -> Any:
    if sel.name == "node":
        return payload.get("node")
    elif sel.name == "cursor":
        return payload.get("cursor")
    return None


# ============================================================
# Introspection
# ============================================================

def schema_object() -> Dict:
    types = []
    for name in OBJECT_TYPES:
        types.append({"name": name, "kind": "OBJECT"})
    for name in ENUM_TYPES:
        types.append({"name": name, "kind": "ENUM"})
    for name in INPUT_OBJECT_TYPES:
        types.append({"name": name, "kind": "INPUT_OBJECT"})
    for name in SCALAR_TYPES:
        types.append({"name": name, "kind": "SCALAR"})
    
    return {
        "queryType": {"name": "Query"},
        "mutationType": {"name": "Mutation"},
        "subscriptionType": None,
        "types": types,
        "directives": [],
    }


def type_object(name: str) -> Dict:
    if name in SCALAR_TYPES:
        return {"name": name, "kind": "SCALAR", "description": None, "fields": None, "enumValues": None, "inputFields": None, "possibleTypes": None, "ofType": None, "specifiedByURL": None}
    
    if name in ENUM_TYPES:
        return {
            "name": name,
            "kind": "ENUM",
            "description": None,
            "fields": None,
            "enumValues": [{"name": v, "description": None, "isDeprecated": False, "deprecationReason": None} for v in ENUM_TYPES[name]],
            "inputFields": None,
            "possibleTypes": None,
            "ofType": None,
            "specifiedByURL": None,
        }
    
    if name in INPUT_OBJECT_TYPES:
        return {
            "name": name,
            "kind": "INPUT_OBJECT",
            "description": None,
            "fields": None,
            "enumValues": None,
            "inputFields": [
                {"name": fn, "description": None, "type": {"name": ft, "kind": "SCALAR" if ft in SCALAR_TYPES else "ENUM", "ofType": None}, "defaultValue": None}
                for fn, ft in INPUT_OBJECT_TYPES[name].items()
            ],
            "possibleTypes": None,
            "ofType": None,
            "specifiedByURL": None,
        }
    
    if name in OBJECT_TYPES:
        fields = []
        for fname, ftype in OBJECT_TYPES[name].items():
            fields.append({
                "name": fname,
                "description": None,
                "args": [],
                "type": {"name": ftype, "kind": "OBJECT" if ftype in OBJECT_TYPES else "SCALAR" if ftype in SCALAR_TYPES else "ENUM" if ftype in ENUM_TYPES else "INPUT_OBJECT" if ftype in INPUT_OBJECT_TYPES else "SCALAR", "ofType": None},
                "isDeprecated": False,
                "deprecationReason": None,
            })
        return {
            "name": name,
            "kind": "OBJECT",
            "description": None,
            "fields": fields,
            "enumValues": None,
            "inputFields": None,
            "possibleTypes": None,
            "ofType": None,
            "specifiedByURL": None,
        }
    
    return {"name": name, "kind": "OBJECT", "description": None, "fields": None, "enumValues": None, "inputFields": None, "possibleTypes": None, "ofType": None, "specifiedByURL": None}


def execute_type_field(context: ExecContext, type_name: str, field_name: str, args: Dict) -> Any:
    if field_name == "__type":
        type_name_arg = args.get("name")
        if type_name_arg:
            return type_object(type_name_arg)
        return None
    elif field_name == "__schema":
        return schema_object()
    return None


# ============================================================
# Main Handle Function
# ============================================================

def handle(store: Store, login: str, body: bytes) -> Tuple[int, Any]:
    """Main GraphQL entry point."""
    # Parse JSON body
    try:
        request = json.loads(body.decode("utf-8")) if body else {}
    except json.JSONDecodeError:
        return 400, {"errors": [{"message": "invalid request body"}]}
    
    # Handle batch requests (list)
    if isinstance(request, list):
        results = []
        for item in request:
            status, result = handle_single(store, login, item)
            results.append(result)
        return 200, results
    
    return handle_single(store, login, request)


def handle_single(store: Store, login: str, request: Dict) -> Tuple[int, Any]:
    """Handle a single GraphQL request."""
    query = request.get("query", "")
    variables = request.get("variables", {})
    operation_name = request.get("operationName")
    
    # Parse query
    try:
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        document = parser.parse()
    except Exception as e:
        return 400, {"errors": [{"message": str(e)}]}
    
    # Validate operations
    for op in document.operations:
        err = validate_operation(op, document.fragments)
        if err:
            return 200, {"data": None, "errors": [{"message": err}]}
    
    # Select operation(s) to execute
    ops_to_run = []
    if operation_name:
        for op in document.operations:
            if op.name == operation_name:
                ops_to_run.append(op)
                break
    elif len(document.operations) == 1:
        ops_to_run = document.operations
    else:
        ops_to_run = document.operations
    
    if not ops_to_run:
        return 200, {"data": None, "errors": [{"message": "No operations to execute"}]}
    
    context = ExecContext(store, login, variables, document.fragments)
    
    # If multiple operations, return list
    if len(ops_to_run) > 1:
        results = []
        for op in ops_to_run:
            result = execute_operation(context, op)
            results.append(result)
        return 200, results
    
    # Single operation
    result = execute_operation(context, ops_to_run[0])
    return 200, result


def execute_operation(context: ExecContext, op: OperationDefinition) -> Dict:
    """Execute a single operation."""
    root_type = "Query" if op.type == "query" else "Mutation"
    data = execute(context, root_type, None, op.selection_set)
    return {"data": data}


# ============================================================
# Resolver dispatch with special types
# ============================================================

# Patch resolve_field to handle special types
_original_resolve_field = resolve_field

def resolve_field(context: ExecContext, type_name: str, payload: Any, sel: Selection) -> Any:
    # Handle __type and __schema on Query
    if type_name == "Query" and sel.name in ("__type", "__schema"):
        args = {arg.name: resolve_value(context, arg.value) for arg in sel.args}
        return execute_type_field(context, type_name, sel.name, args)
    
    # Handle IssueCommentEdge
    if type_name == "IssueCommentEdge":
        return resolve_issue_comment_edge(context, payload, sel, {})
    
    return _original_resolve_field(context, type_name, payload, sel)


# Make it available
resolve_field = resolve_field