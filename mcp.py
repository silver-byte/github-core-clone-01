"""
MCP (Model Context Protocol) endpoint over JSON-RPC 2.0.
Shares the same store as REST/GraphQL.
"""

import json
from typing import Any, Dict, List, Tuple, Optional, Union

from store import Store, Repo, iso_utc, git_ident
from gitcore import ObjectStore, parse_commit, build_commit, is_ancestor, ZERO_SHA
from webapi import _resolve_ref_name
import sys


def _dumps(obj: Any) -> bytes:
    """Deterministic JSON encoding."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


# Module constant TOOLS - built ONCE with exact six tools
TOOLS = [
    {
        "name": "create_repo",
        "description": "Create a new repository owned by the authenticated user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "private": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_repo",
        "description": "Get a repository by owner and name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_issue",
        "description": "Create an issue in a repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["owner", "repo", "title"],
        },
    },
    {
        "name": "list_issues",
        "description": "List issues in a repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_pull_request",
        "description": "Create a pull request from a head branch to a base branch.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "head": {"type": "string"},
                "base": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["owner", "repo", "title", "head", "base"],
        },
    },
    {
        "name": "merge_pull_request",
        "description": "Merge a pull request (squash) into its base branch.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pull_number": {"type": "integer"},
            },
            "required": ["owner", "repo", "pull_number"],
        },
    },
]


# Supported protocol versions
SUPPORTED_VERSIONS = [
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
]

DEFAULT_VERSION = "2025-11-25"


def _jsonrpc_error(id_val: Any, code: int, message: str) -> bytes:
    """Create JSON-RPC error response."""
    return _dumps({
        "jsonrpc": "2.0",
        "id": id_val,
        "error": {"code": code, "message": message},
    })


def _jsonrpc_success(id_val: Any, result: Any) -> bytes:
    """Create JSON-RPC success response."""
    return _dumps({
        "jsonrpc": "2.0",
        "id": id_val,
        "result": result,
    })


def _tool_result(text: str, is_error: bool = False) -> Dict[str, Any]:
    """Create tool result shape."""
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def _resolve_tool_name(name: str) -> Optional[str]:
    """Resolve tool name with fuzzy matching. Try longer names first."""
    lowered = name.lower()
    # Sort tools by name length descending for longest match first
    for tool in sorted(TOOLS, key=lambda t: len(t["name"]), reverse=True):
        tool_name = tool["name"]
        if lowered == tool_name or lowered.endswith("_" + tool_name) or lowered.endswith(tool_name):
            return tool_name
    return None


def _error_result(message: str) -> Dict[str, Any]:
    """Create error tool result."""
    return _tool_result(message, is_error=True)


def _success_result(data: Any) -> Dict[str, Any]:
    """Create success tool result with JSON text."""
    return _tool_result(_dumps(data).decode(), is_error=False)


def _get_repo(store: Store, owner: str, repo_name: str) -> Optional[Repo]:
    """Get repo by owner/name."""
    return store.get_repo(owner, repo_name)


def handle_tool_create_repo(store: Store, login: str, args: Dict) -> Dict[str, Any]:
    """Handle create_repo tool."""
    name = args.get("name", "").strip()
    if not name:
        return _error_result("missing required argument: name")
    description = args.get("description", "") or ""
    private = bool(args.get("private", False))
    try:
        repo = store.create_repo(login, name, description, private)
    except ValueError:
        return _error_result("repository already exists")
    return _success_result({
        "id": repo.db_id,
        "name": repo.name,
        "full_name": repo.full_name,
        "private": repo.private,
        "default_branch": repo.default_branch,
        "owner": {"login": repo.owner},
    })


def handle_tool_get_repo(store: Store, login: str, args: Dict) -> Dict[str, Any]:
    """Handle get_repo tool."""
    owner = args.get("owner", "").strip()
    repo_name = args.get("repo", "").strip()
    if not owner or not repo_name:
        return _error_result("missing required argument: owner or repo")
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return _error_result("Not Found")
    return _success_result({
        "id": repo.db_id,
        "name": repo.name,
        "full_name": repo.full_name,
        "private": repo.private,
        "default_branch": repo.default_branch,
        "owner": {"login": repo.owner},
    })


def handle_tool_create_issue(store: Store, login: str, args: Dict) -> Dict[str, Any]:
    """Handle create_issue tool."""
    owner = args.get("owner", "").strip()
    repo_name = args.get("repo", "").strip()
    title = args.get("title", "").strip()
    body = args.get("body", "") or ""
    if not owner or not repo_name:
        return _error_result("missing required argument: owner or repo")
    if not title:
        return _error_result("title is required")
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return _error_result("Not Found")
    ts = store.clock.tick()
    number = repo.new_number()
    issue = {
        "number": number,
        "db_id": store.new_issue_db(),
        "title": title,
        "body": body,
        "state": "open",
        "labels": [],
        "user": login,
        "created_at": ts,
        "updated_at": ts,
        "closed_at": None,
        "comments": [],
    }
    repo.issues[number] = issue
    return _success_result({
        "id": issue["db_id"],
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "state": issue["state"],
        "labels": [],
        "user": {"login": issue["user"]},
    })


def handle_tool_list_issues(store: Store, login: str, args: Dict) -> Dict[str, Any]:
    """Handle list_issues tool."""
    owner = args.get("owner", "").strip()
    repo_name = args.get("repo", "").strip()
    state = args.get("state", "open")
    if not owner or not repo_name:
        return _error_result("missing required argument: owner or repo")
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return _error_result("Not Found")
    issues_list = list(repo.issues.values())
    if state == "open":
        issues_list = [i for i in issues_list if i["state"] == "open"]
    elif state == "closed":
        issues_list = [i for i in issues_list if i["state"] == "closed"]
    elif state == "all":
        pass
    issues_list.sort(key=lambda x: x["number"])
    return _success_result([{
        "id": i["db_id"],
        "number": i["number"],
        "title": i["title"],
        "body": i["body"],
        "state": i["state"],
        "labels": i["labels"],
        "user": {"login": i["user"]},
    } for i in issues_list])


def handle_tool_create_pull_request(store: Store, login: str, args: Dict) -> Dict[str, Any]:
    """Handle create_pull_request tool."""
    owner = args.get("owner", "").strip()
    repo_name = args.get("repo", "").strip()
    title = args.get("title", "").strip()
    head = args.get("head", "").strip()
    base = args.get("base", "").strip()
    body = args.get("body", "") or ""
    if not owner or not repo_name or not title or not head or not base:
        return _error_result("missing required argument: owner, repo, title, head, or base")
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return _error_result("Not Found")
    head_sha = repo.tip(f"refs/heads/{head}")
    base_sha = repo.tip(f"refs/heads/{base}")
    if not head_sha or not base_sha:
        return _error_result("head or base branch not found")
    ts = store.clock.tick()
    number = repo.new_number()
    pull = {
        "number": number,
        "db_id": store.new_pull_db(),
        "title": title,
        "body": body,
        "state": "open",
        "head_ref": head,
        "base_ref": base,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "merged": False,
        "merged_at": None,
        "merge_commit_sha": None,
        "user": login,
        "created_at": ts,
        "updated_at": ts,
    }
    repo.pulls[number] = pull
    return _success_result({
        "id": pull["db_id"],
        "number": pull["number"],
        "title": pull["title"],
        "body": pull["body"],
        "state": pull["state"],
        "head": {"ref": pull["head_ref"], "sha": pull["head_sha"]},
        "base": {"ref": pull["base_ref"], "sha": pull["base_sha"]},
    })


def handle_tool_merge_pull_request(store: Store, login: str, args: Dict) -> Dict[str, Any]:
    """Handle merge_pull_request tool."""
    owner = args.get("owner", "").strip()
    repo_name = args.get("repo", "").strip()
    pull_number = args.get("pull_number")
    if not owner or not repo_name or pull_number is None:
        return _error_result("missing required argument: owner, repo, or pull_number")
    try:
        pull_number = int(pull_number)
    except (ValueError, TypeError):
        return _error_result("pull_number must be an integer")
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return _error_result("Not Found")
    pull = repo.pulls.get(pull_number)
    if not pull:
        return _error_result("pull request not found")
    ok, result = __import__("store").merge_pull(store, repo, pull, login)
    if not ok:
        return _error_result(str(result))
    merge_sha = result
    return _success_result({
        "number": pull["number"],
        "merged": True,
        "sha": merge_sha,
        "state": "closed",
    })


def handle_tools_call(store: Store, login: str, tool_name: str, arguments: Dict) -> Dict[str, Any]:
    """Route to appropriate tool handler."""
    resolved = _resolve_tool_name(tool_name)
    if not resolved:
        return _error_result(f"Tool {tool_name} not found")
    
    handlers = {
        "create_repo": handle_tool_create_repo,
        "get_repo": handle_tool_get_repo,
        "create_issue": handle_tool_create_issue,
        "list_issues": handle_tool_list_issues,
        "create_pull_request": handle_tool_create_pull_request,
        "merge_pull_request": handle_tool_merge_pull_request,
    }
    handler = handlers.get(resolved)
    if not handler:
        return _error_result(f"Tool {tool_name} not found")
    return handler(store, login, arguments)


def handle(
    store: Store,
    login: str,
    headers: Dict[str, str],
    body: bytes,
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Main MCP handler."""
    # Parse JSON body
    try:
        if not body:
            request = {}
        else:
            request = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        error_body = _jsonrpc_error(None, -32700, "Parse error")
        return 400, [("Content-Type", "application/json")], error_body
    
    # Handle batch requests (list)
    if isinstance(request, list):
        responses = []
        all_extra_headers: List[Tuple[str, str]] = []
        for item in request:
            resp, extra_h = _handle_single_request(store, login, item)
            if resp is not None:
                responses.append(json.loads(resp.decode("utf-8")))
                all_extra_headers.extend(extra_h)
        if not responses:
            return 202, [("Content-Type", "application/json")], b""
        return 200, [("Content-Type", "application/json")] + all_extra_headers, _dumps(responses)
    
    # Single request
    resp, extra_h = _handle_single_request(store, login, request)
    if resp is None:
        return 202, [("Content-Type", "application/json")], b""
    return 200, [("Content-Type", "application/json")] + extra_h, resp


def _handle_single_request(store: Store, login: str, request: Dict) -> Tuple[Optional[bytes], List[Tuple[str, str]]]:
    """Handle a single JSON-RPC request. Returns (response_bytes, extra_headers) or (None, []) for notifications."""
    # Check for id
    id_val = request.get("id")
    is_notification = "id" not in request
    
    method = request.get("method", "")
    params = request.get("params", {})
    extra_headers: List[Tuple[str, str]] = []
    
    # Handle methods
    if method == "server/discover":
        result = {
            "supportedVersions": SUPPORTED_VERSIONS,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "forge-mcp", "version": "1.0.0"},
            "instructions": "GitHub-compatible forge tools",
        }
    elif method == "initialize":
        proto = params.get("protocolVersion")
        if proto not in SUPPORTED_VERSIONS:
            proto = DEFAULT_VERSION
        result = {
            "protocolVersion": proto,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "forge-mcp", "version": "1.0.0"},
        }
        # Add session header
        extra_headers = [("Mcp-Session-Id", "forge-session-1")]
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        tool_result = handle_tools_call(store, login, name, arguments)
        if tool_result.get("isError"):
            result = tool_result
        else:
            result = tool_result
    else:
        # Unknown method
        if is_notification:
            return None, []
        return _jsonrpc_error(id_val, -32601, "Method not found"), []
    
    if is_notification:
        return None, []
    return _jsonrpc_success(id_val, result), extra_headers