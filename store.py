"""
In-memory data store for the git forge.
Imports ObjectStore from gitcore.
"""

import os
import time
import hashlib
import difflib
import threading
from typing import Dict, List, Optional, Any, Tuple, Union

from gitcore import (
    ObjectStore,
    parse_commit,
    build_commit,
    flatten_tree,
    tree_from_flat,
    build_tree,
    is_ancestor,
)


BASE_EPOCH = 1735689600  # 2025-01-01T00:00:00Z


def iso_utc(ts: int) -> str:
    """Format epoch seconds as YYYY-MM-DDTHH:MM:SSZ using time.gmtime."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def git_ident(login: str, ts: int) -> str:
    """Return Git author/committer ident string."""
    return f"{login} <{login}@example.invalid> {ts} +0000"


class Clock:
    """Logical clock providing strictly increasing timestamps."""

    def __init__(self) -> None:
        self.n = 0

    def tick(self) -> int:
        self.n += 1
        return BASE_EPOCH + self.n


class Repo:
    """In-memory repository representation."""

    def __init__(
        self,
        db_id: int,
        owner: str,
        name: str,
        description: str,
        private: bool,
        created_ts: int,
    ) -> None:
        self.db_id = db_id
        self.owner = owner
        self.name = name
        self.description = description
        self.private = private
        self.default_branch = "main"
        self.created_at = created_ts
        self.updated_at = created_ts
        self.objects = ObjectStore()
        self.refs: Dict[str, str] = {}
        self.labels: Dict[str, Dict[str, Any]] = {}
        self.issues: Dict[int, Dict[str, Any]] = {}
        self.pulls: Dict[int, Dict[str, Any]] = {}
        self.next_number = 1
        self.next_label_id = 1

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def head_ref(self) -> str:
        return f"refs/heads/{self.default_branch}"

    def tip(self, refname: str) -> Optional[str]:
        return self.refs.get(refname)

    def set_ref(self, refname: str, sha: str) -> None:
        self.refs[refname] = sha

    def delete_ref(self, refname: str) -> bool:
        if refname in self.refs:
            del self.refs[refname]
            return True
        return False

    def new_number(self) -> int:
        num = self.next_number
        self.next_number += 1
        return num


class Store:
    """Global in-memory store for all repositories."""

    def __init__(self, env_token: Optional[str]) -> None:
        self.clock = Clock()
        self.env_token = env_token
        self.viewer_login = "adaiversen"
        self.repos: Dict[tuple[str, str], Repo] = {}
        self._foreign: Dict[str, str] = {}
        self.next_repo_id = 1
        self.next_issue_db = 1
        self.next_pull_db = 1
        self.next_comment_id = 1
        self.lock = threading.RLock()

    def login_for_token(self, token: Optional[str]) -> Optional[str]:
        if token is None:
            return None
        if self.env_token is not None and token == self.env_token:
            return self.viewer_login
        if token not in self._foreign:
            self._foreign[token] = "ext-" + hashlib.sha1(token.encode()).hexdigest()[:10]
        return self._foreign[token]

    def is_env_token(self, token: Optional[str]) -> bool:
        return self.env_token is not None and token == self.env_token

    def learn_viewer(self, owner: Optional[str]) -> None:
        if owner and owner != self.viewer_login and not self.repos:
            self.viewer_login = owner

    def create_repo(
        self, owner: str, name: str, description: str = "", private: bool = False
    ) -> Repo:
        key = (owner, name)
        if key in self.repos:
            raise ValueError("exists")
        db_id = self.next_repo_id
        self.next_repo_id += 1
        created_ts = self.clock.tick()
        repo = Repo(db_id, owner, name, description, private, created_ts)
        self.repos[key] = repo
        return repo

    def get_repo(self, owner: str, name: str) -> Optional[Repo]:
        return self.repos.get((owner, name))

    def delete_repo(self, owner: str, name: str) -> bool:
        key = (owner, name)
        if key in self.repos:
            del self.repos[key]
            return True
        return False

    def list_repos(self, owner: str) -> List[Repo]:
        return sorted(
            [r for (o, _), r in self.repos.items() if o == owner],
            key=lambda r: r.db_id,
        )

    def new_issue_db(self) -> int:
        db_id = self.next_issue_db
        self.next_issue_db += 1
        return db_id

    def new_pull_db(self) -> int:
        db_id = self.next_pull_db
        self.next_pull_db += 1
        return db_id

    def new_comment_id(self) -> int:
        cid = self.next_comment_id
        self.next_comment_id += 1
        return cid


def new_store_from_env() -> Store:
    """Create Store from MAKEFUN_SHOP_TOKEN environment variable."""
    token = os.environ.get("MAKEFUN_SHOP_TOKEN")
    return Store(token)


def merge_pull(store: Store, repo: Repo, pull: Dict[str, Any], login: str) -> Tuple[bool, Union[str, str]]:
    """Squash-merge pull into its base branch. Returns (ok, payload)."""
    from gitcore import (
        ObjectStore,
        parse_commit,
        build_commit,
        flatten_tree,
        tree_from_flat,
        build_tree,
        diff_trees,
        is_ancestor,
    )
    import difflib
    
    head_ref = f"refs/heads/{pull['head_ref']}"
    base_ref = f"refs/heads/{pull['base_ref']}"
    
    head_tip = repo.tip(head_ref)
    base_tip = repo.tip(base_ref)
    
    if not head_tip or not base_tip:
        return False, "branch missing"
    
    if pull.get("merged", False) or pull["state"] == "closed":
        return False, "not mergeable"
    
    if is_ancestor(repo.objects, head_tip, base_tip):
        return False, "already up to date"
    
    # Find merge base: ancestors of base_tip
    base_ancestors = set()
    stack = [base_tip]
    while stack:
        sha = stack.pop()
        if sha in base_ancestors or not repo.objects.has(sha):
            continue
        base_ancestors.add(sha)
        obj_type, data = repo.objects.get(sha)
        if obj_type != "commit":
            continue
        parsed = parse_commit(data)
        for p in parsed["parents"]:
            if p not in base_ancestors:
                stack.append(p)
    
    # BFS from head_tip by generation
    merge_base = None
    queue = [head_tip]
    visited = set()
    while queue:
        sha = queue.pop(0)
        if sha in visited or not repo.objects.has(sha):
            continue
        visited.add(sha)
        if sha in base_ancestors:
            merge_base = sha
            break
        obj_type, data = repo.objects.get(sha)
        if obj_type != "commit":
            continue
        parsed = parse_commit(data)
        for p in parsed["parents"]:
            if p not in visited:
                queue.append(p)
    
    # Get trees
    def get_tree(sha: str) -> str:
        if not sha or not repo.objects.has(sha):
            return ""
        obj_type, data = repo.objects.get(sha)
        if obj_type != "commit":
            return ""
        parsed = parse_commit(data)
        return parsed["tree"]
    
    mb_tree = get_tree(merge_base) if merge_base else ""
    b_tree = get_tree(base_tip)
    h_tree = get_tree(head_tip)
    
    # Flat maps
    mb_flat = flatten_tree(repo.objects, mb_tree) if mb_tree else {}
    b_flat = flatten_tree(repo.objects, b_tree) if b_tree else {}
    h_flat = flatten_tree(repo.objects, h_tree) if h_tree else {}
    
    all_paths = set(mb_flat.keys()) | set(b_flat.keys()) | set(h_flat.keys())
    result_files = {}
    
    for path in all_paths:
        mb_sha = mb_flat.get(path, (None, None))[1]
        b_sha = b_flat.get(path, (None, None))[1]
        h_sha = h_flat.get(path, (None, None))[1]
        
        if b_sha == h_sha:
            result_files[path] = b_sha
        elif mb_sha == b_sha:
            result_files[path] = h_sha
        elif mb_sha == h_sha:
            result_files[path] = b_sha
        else:
            # Conflict - try line-level merge
            if not mb_sha or not b_sha or not h_sha:
                return False, "conflict"
            
            mb_blob = repo.objects.get_data(mb_sha).decode("utf-8", errors="replace")
            b_blob = repo.objects.get_data(b_sha).decode("utf-8", errors="replace")
            h_blob = repo.objects.get_data(h_sha).decode("utf-8", errors="replace")
            
            mb_lines = mb_blob.splitlines()
            b_lines = b_blob.splitlines()
            h_lines = h_blob.splitlines()
            
            # Get changes as spans (i1, i2, replacement_lines)
            def get_spans(from_lines, to_lines):
                matcher = difflib.SequenceMatcher(None, from_lines, to_lines)
                spans = []
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != "equal":
                        spans.append((i1, i2, to_lines[j1:j2]))
                return spans
            
            b_spans = get_spans(mb_lines, b_lines)
            h_spans = get_spans(mb_lines, h_lines)
            
            # Check for conflicts
            conflict = False
            for bi1, bi2, b_repl in b_spans:
                for hi1, hi2, h_repl in h_spans:
                    if not (bi2 <= hi1 or hi2 <= bi1):  # overlap
                        if b_repl != h_repl:
                            conflict = True
                            break
                if conflict:
                    break
            
            if conflict:
                return False, "conflict"
            
            # Apply both sets of changes
            result_lines = list(mb_lines)
            offset = 0
            all_spans = sorted(b_spans + h_spans, key=lambda x: x[0])
            for i1, i2, repl in all_spans:
                adj_i1 = i1 + offset
                adj_i2 = i2 + offset
                result_lines[adj_i1:adj_i2] = repl
                offset += len(repl) - (i2 - i1)
            
            result_blob = ("\n".join(result_lines) + ("\n" if result_lines else "")).encode("utf-8")
            result_sha = repo.objects.add("blob", result_blob)
            result_files[path] = result_sha
    
    # Build result tree
    if result_files:
        result_tree_sha = tree_from_flat(repo.objects, result_files)
    else:
        result_tree_sha = repo.objects.add("tree", build_tree([]))
    
    # Create merge commit
    ts1 = store.clock.tick()
    ident = git_ident(login, ts1)
    merge_msg = pull["title"]
    if pull.get("body"):
        merge_msg += "\n\n" + pull["body"]
    merge_commit = build_commit(result_tree_sha, [base_tip], ident, ident, merge_msg)
    merge_sha = repo.objects.add("commit", merge_commit)
    
    # Update base branch
    repo.set_ref(base_ref, merge_sha)
    repo.updated_at = store.clock.tick()
    
    # Update pull
    ts2 = store.clock.tick()
    pull["state"] = "closed"
    pull["merged"] = True
    pull["merged_at"] = ts2
    pull["merge_commit_sha"] = merge_sha
    pull["base_sha"] = base_tip
    pull["updated_at"] = ts2
    
    return True, merge_sha