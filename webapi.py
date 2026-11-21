"""
HTTP routing layer for the git forge.
Imports: gitcore, gitproto, store
"""

import json
import base64
import itertools
import difflib
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

from gitcore import (
    ObjectStore,
    parse_commit,
    build_commit,
    tree_list,
    tree_lookup,
    tree_update,
    tree_from_flat,
    flatten_tree,
    diff_trees,
    is_ancestor,
    ZERO_SHA,
    TYPE_NAME,
    PackError,
)
from gitproto import (
    advertise,
    upload_pack_v2,
    upload_pack_v0,
    receive_pack,
    FLUSH,
    PacketError,
)
from store import Store, Repo, iso_utc, git_ident, BASE_EPOCH


# Global request counter
_request_counter = itertools.count(1)


def _next_req_id() -> str:
    return f"req-{next(_request_counter)}"


def err(status: int, message: str) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Return error JSON response."""
    body = json.dumps({"message": message}, separators=(",", ":"))
    return status, [("Content-Type", "application/json; charset=utf-8")], body.encode()


def gid(kind: str, key: str) -> str:
    """Global ID encoding."""
    return base64.b64encode(f"{kind}:{key}".encode()).decode()


def user_json(login: str) -> Dict[str, Any]:
    """User JSON object."""
    return {
        "login": login,
        "id": 1000,
        "type": "User",
        "site_admin": False,
    }


def label_json(label: Dict[str, Any]) -> Dict[str, Any]:
    """Label JSON object."""
    return {
        "id": label["id"],
        "name": label["name"],
        "color": label["color"],
        "description": label["description"],
        "default": False,
    }


def comment_json(c: Dict[str, Any]) -> Dict[str, Any]:
    """Comment JSON object."""
    return {
        "id": c["id"],
        "body": c["body"],
        "user": user_json(c["user"]),
        "created_at": iso_utc(c["created_at"]),
        "updated_at": iso_utc(c["updated_at"]),
    }


def ident_parts(line: str, default_ts: int) -> Dict[str, str]:
    """Parse 'Name <email> ts +0000' into parts."""
    try:
        if " <" in line and "> " in line:
            name = line[:line.index(" <")]
            rest = line[line.index(" <") + 2:]
            email = rest[:rest.index("> ")]
            ts_str = rest[rest.index("> ") + 2:].strip()
            ts = int(ts_str.split()[0])
            return {"name": name, "email": email, "date": iso_utc(ts)}
    except Exception:
        pass
    return {"name": line, "email": "", "date": iso_utc(default_ts)}


def repo_json(repo: Repo, login: Optional[str] = None) -> Dict[str, Any]:
    """Repository JSON object."""
    open_issues = sum(1 for i in repo.issues.values() if i["state"] == "open")
    open_pulls = sum(1 for p in repo.pulls.values() if p["state"] == "open")
    return {
        "id": repo.db_id,
        "node_id": gid("Repository", str(repo.db_id)),
        "name": repo.name,
        "full_name": repo.full_name,
        "private": repo.private,
        "owner": user_json(repo.owner),
        "description": repo.description,
        "default_branch": repo.default_branch,
        "created_at": iso_utc(repo.created_at),
        "updated_at": iso_utc(repo.updated_at),
        "pushed_at": iso_utc(repo.updated_at),
        "html_url": f"/{repo.owner}/{repo.name}",
        "clone_url": f"/{repo.owner}/{repo.name}.git",
        "git_url": f"/{repo.owner}/{repo.name}.git",
        "fork": False,
        "size": 0,
        "stargazers_count": 0,
        "watchers_count": 0,
        "language": None,
        "has_issues": True,
        "has_projects": False,
        "has_wiki": False,
        "archived": False,
        "disabled": False,
        "open_issues_count": open_issues + open_pulls,
        "license": None,
        "visibility": "private" if repo.private else "public",
    }


def commit_json(store: Store, repo: Repo, csha: str) -> Dict[str, Any]:
    """Commit JSON object."""
    obj_type, data = repo.objects.get(csha)
    parsed = parse_commit(data)
    default_ts = store.clock.n + BASE_EPOCH
    return {
        "sha": csha,
        "node_id": gid("Commit", csha),
        "commit": {
            "author": ident_parts(parsed["author"], default_ts),
            "committer": ident_parts(parsed["committer"], default_ts),
            "message": parsed["message"],
            "tree": {"sha": parsed["tree"]},
        },
        "parents": [{"sha": p} for p in parsed["parents"]],
    }


def issue_json(repo: Repo, issue: Dict[str, Any]) -> Dict[str, Any]:
    """Issue JSON object."""
    labels = [label_json(repo.labels[n]) for n in issue["labels"] if n in repo.labels]
    closed_at = iso_utc(issue["closed_at"]) if issue.get("closed_at") else None
    return {
        "id": issue["db_id"],
        "node_id": gid("Issue", f"{repo.db_id}:{issue['number']}"),
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "state": issue["state"],
        "user": user_json(issue["user"]),
        "labels": labels,
        "assignees": [],
        "comments": len(issue["comments"]),
        "created_at": iso_utc(issue["created_at"]),
        "updated_at": iso_utc(issue["updated_at"]),
        "closed_at": closed_at,
        "html_url": f"/{repo.owner}/{repo.name}/issues/{issue['number']}",
    }


def pull_json(repo: Repo, pull: Dict[str, Any]) -> Dict[str, Any]:
    """Pull request JSON object."""
    merged_at = iso_utc(pull["merged_at"]) if pull.get("merged_at") else None
    return {
        "id": pull["db_id"],
        "node_id": gid("PullRequest", f"{repo.db_id}:{pull['number']}"),
        "number": pull["number"],
        "title": pull["title"],
        "body": pull["body"],
        "state": pull["state"],
        "draft": False,
        "merged": pull.get("merged", False),
        "merged_at": merged_at,
        "merge_commit_sha": pull.get("merge_commit_sha"),
        "user": user_json(pull["user"]),
        "head": {
            "label": f"{repo.owner}:{pull['head_ref']}",
            "ref": pull["head_ref"],
            "sha": pull["head_sha"],
        },
        "base": {
            "label": f"{repo.owner}:{pull['base_ref']}",
            "ref": pull["base_ref"],
            "sha": pull["base_sha"],
        },
        "created_at": iso_utc(pull["created_at"]),
        "updated_at": iso_utc(pull["updated_at"]),
        "html_url": f"/{repo.owner}/{repo.name}/pulls/{pull['number']}",
    }


def parse_json_body(body: bytes) -> Dict[str, Any]:
    """Parse JSON request body."""
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        raise ValueError("Problems parsing JSON")


def _parse_auth(headers: Dict[str, str]) -> Optional[str]:
    """Extract token from Authorization header."""
    for k, v in headers.items():
        if k.lower() == "authorization":
            auth = v
            if auth.startswith("Bearer "):
                return auth[7:].strip()
            if auth.startswith("token "):
                return auth[6:].strip()
    return None


def _get_header(headers: Dict[str, str], name: str, default: str = "") -> str:
    """Get header value by name (case-insensitive)."""
    name_lower = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lower:
            return v
    return default


def _resolve_ref_name(repo: Repo, ref: str) -> str:
    """Resolve branch name or SHA to a commit SHA."""
    # Try as branch
    if ref.startswith("refs/heads/"):
        tip = repo.tip(ref)
        if tip:
            return tip
    elif ref.startswith("refs/tags/"):
        tip = repo.tip(ref)
        if tip:
            return tip
    # Try as branch name
    tip = repo.tip(f"refs/heads/{ref}")
    if tip:
        return tip
    # Try as tag
    tip = repo.tip(f"refs/tags/{ref}")
    if tip:
        return tip
    # Assume it's a SHA if 40 hex
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
        if repo.objects.has(ref):
            return ref
    return ZERO_SHA


def _resolve_commit_sha(repo: Repo, ref: Optional[str]) -> str:
    """Resolve a ref (branch name or SHA) to a commit SHA."""
    if not ref:
        ref = repo.default_branch
    sha = _resolve_ref_name(repo, ref)
    if sha == ZERO_SHA:
        # Try default branch
        sha = repo.tip(f"refs/heads/{repo.default_branch}") or ZERO_SHA
    return sha


def _walk_commits_paged(repo: Repo, start_sha: str, page: int, per_page: int) -> List[str]:
    """Walk commits from start_sha newest-first, return page slice."""
    if not repo.objects.has(start_sha):
        return []
    
    commits: List[str] = []
    visited: set = set()
    queue: List[str] = [start_sha]
    
    while queue and len(commits) < page * per_page:
        sha = queue.pop(0)
        if sha in visited:
            continue
        if not repo.objects.has(sha):
            continue
        obj_type, data = repo.objects.get(sha)
        if obj_type != "commit":
            continue
        visited.add(sha)
        commits.append(sha)
        parsed = parse_commit(data)
        for p in parsed["parents"]:
            if p not in visited:
                queue.append(p)
    
    start = (page - 1) * per_page
    end = start + per_page
    return commits[start:end]


def _build_link_header(path: str, page: int, per_page: int, has_next: bool) -> str:
    """Build Link header for pagination."""
    links = []
    if has_next:
        links.append(f'<{path}?per_page={per_page}&page={page + 1}>; rel="next"')
    # last page - approximate
    links.append(f'<{path}?per_page={per_page}&page={page + 1}>; rel="last"')
    return ", ".join(links)


def repo_for_git(path: str) -> Optional[Tuple[str, str]]:
    """Extract owner, repo from git path like /owner/repo.git/..."""
    if not path.startswith("/"):
        return None
    parts = path[1:].split("/")
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo_name = parts[1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return owner, repo_name


def _handle_git_endpoints(
    store: Store,
    repo_key: Optional[Tuple[str, str]],
    method: str,
    path: str,
    query: Dict[str, List[str]],
    headers: List[Tuple[str, str]],
    body: bytes,
    base_headers: List[Tuple[str, str]],
    login: Optional[str],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle git smart-HTTP endpoints."""
    if not repo_key:
        return err(404, "Not Found")
    
    owner, repo_name = repo_key
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return 404, base_headers + [("Content-Type", "text/plain; charset=utf-8")], b"not found\n"
    
    # Learn viewer for git endpoints
    if login and store.is_env_token(_parse_auth(headers)):
        store.learn_viewer(owner)
    
    service_param = query.get("service", [None])[0]
    git_protocol = _get_header(headers, "git-protocol", "")
    version2 = "version=2" in git_protocol
    
    if path.endswith("/info/refs") and method == "GET":
        if service_param not in ("git-upload-pack", "git-receive-pack"):
            return err(400, "Bad Request")
        content_type = f"application/x-{service_param}-advertisement"
        adv = advertise(repo, service_param, version2)
        return 200, base_headers + [("Content-Type", content_type)], adv
    
    if path.endswith("/git-upload-pack") and method == "POST":
        if version2:
            try:
                result = upload_pack_v2(repo, body)
            except PacketError:
                return 400, base_headers + [("Content-Type", "text/plain")], b"Bad pkt-line\n"
        else:
            try:
                result = upload_pack_v0(repo, body)
            except PacketError:
                return 400, base_headers + [("Content-Type", "text/plain")], b"Bad pkt-line\n"
        return 200, base_headers + [("Content-Type", "application/x-git-upload-pack-result")], result
    
    if path.endswith("/git-receive-pack") and method == "POST":
        try:
            result = receive_pack(repo, body)
        except PacketError:
            return 400, base_headers + [("Content-Type", "text/plain")], b"Bad pkt-line\n"
        return 200, base_headers + [("Content-Type", "application/x-git-receive-pack-result")], result
    
    return err(404, "Not Found")


def dispatch(
    store: Store,
    method: str,
    path: str,
    query: Dict[str, List[str]],
    headers: List[Tuple[str, str]],
    body: bytes,
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Main HTTP dispatch function."""
    req_id = _next_req_id()
    base_headers = [("X-GitHub-Request-Id", req_id)]
    
    # Handle /healthz
    if path == "/healthz" and method == "GET":
        resp_body = json.dumps({"ok": True}, separators=(",", ":"))
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], resp_body.encode()
    
    # Parse auth
    token = _parse_auth(headers)
    login = store.login_for_token(token) if token else None
    
    # MCP endpoint (before git endpoints since it's not under /api)
    if path.startswith("/mcp"):
        return _handle_mcp(store, method, path, query, headers, body, base_headers, login, token)
    
    # Git endpoints (not under /api)
    if not path.startswith("/api/"):
        return _handle_git_endpoints(store, repo_for_git(path), method, path, query, headers, body, base_headers, login)
    
    # API routes
    if path.startswith("/api/v3"):
        return _handle_api_v3(store, method, path, query, headers, body, base_headers, login, token)
    
    if path.startswith("/api/graphql"):
        return _handle_graphql(store, method, path, query, headers, body, base_headers, login, token)
    
    return err(404, "Not Found")


def _handle_api_v3(
    store: Store,
    method: str,
    path: str,
    query: Dict[str, List[str]],
    headers: List[Tuple[str, str]],
    body: bytes,
    base_headers: List[Tuple[str, str]],
    login: Optional[str],
    token: Optional[str],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle /api/v3 routes."""
    if login is None:
        return err(401, "Requires authentication")
    
    # Learn viewer
    store.learn_viewer(login)
    
    # Strip /api/v3 prefix
    api_path = path[7:]  # remove "/api/v3"
    if not api_path:
        api_path = "/"
    
    # Parse JSON body for mutating methods
    json_body: Dict[str, Any] = {}
    if method in ("POST", "PATCH", "PUT", "DELETE") and body:
        try:
            json_body = parse_json_body(body)
        except ValueError:
            return err(400, "Problems parsing JSON")
    
    # Helper to get query param
    def q(name: str, default: str = "") -> str:
        vals = query.get(name, [])
        return vals[0] if vals else default
    
    def q_int(name: str, default: int) -> int:
        try:
            return int(q(name, str(default)))
        except ValueError:
            return default
    
    # Root
    if api_path in ("", "/"):
        return 200, base_headers + [
            ("Content-Type", "application/json; charset=utf-8"),
            ("X-OAuth-Scopes", "repo"),
        ], b"{}"
    
    if api_path == "/meta":
        resp = json.dumps({"installed_version": "forge-1.0"}, separators=(",", ":"))
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], resp.encode()
    
    if api_path == "/user":
        resp = json.dumps(user_json(login), separators=(",", ":"))
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], resp.encode()
    
    # /user/repos
    if api_path == "/user/repos" and method == "GET":
        repos = store.list_repos(login)
        page = max(1, q_int("page", 1))
        per_page = min(100, max(1, q_int("per_page", 30)))
        start = (page - 1) * per_page
        end = start + per_page
        page_repos = repos[start:end]
        has_next = end < len(repos)
        resp_body = json.dumps([repo_json(r) for r in page_repos], separators=(",", ":"))
        hdrs = base_headers + [("Content-Type", "application/json; charset=utf-8")]
        if has_next:
            hdrs.append(("Link", _build_link_header("/api/v3/user/repos", page, per_page, True)))
        return 200, hdrs, resp_body.encode()
    
    if api_path == "/user/repos" and method == "POST":
        name = json_body.get("name", "").strip()
        if not name or "/" in name or any(c.isspace() for c in name):
            return err(422, "Repository creation failed.")
        if store.get_repo(login, name):
            return err(422, "Repository creation failed.")
        description = json_body.get("description", "") or ""
        private = bool(json_body.get("private", False))
        repo = store.create_repo(login, name, description, private)
        if "default_branch" in json_body:
            repo.default_branch = json_body["default_branch"]
        return 201, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(repo_json(repo), separators=(",", ":")).encode()
    
    # /repos/{owner}/{repo}
    if api_path.startswith("/repos/"):
        return _handle_repo_routes(store, api_path, method, query, json_body, base_headers, login)
    
    return err(404, "Not Found")


def _handle_repo_routes(
    store: Store,
    api_path: str,
    method: str,
    query: Dict[str, List[str]],
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
    login: str,
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle /repos/{owner}/{repo}/* routes."""
    # Parse owner/repo from path
    parts = api_path.split("/")
    if len(parts) < 4:
        return err(404, "Not Found")
    owner = parts[2]
    repo_name = parts[3]
    repo = store.get_repo(owner, repo_name)
    if not repo:
        return err(404, "Not Found")
    
    # Learn viewer
    store.learn_viewer(owner)
    
    sub_path = "/" + "/".join(parts[4:]) if len(parts) > 4 else ""
    
    def q(name: str, default: str = "") -> str:
        vals = query.get(name, [])
        return vals[0] if vals else default
    
    def q_int(name: str, default: int) -> int:
        try:
            return int(q(name, str(default)))
        except ValueError:
            return default
    
    # GET/PATCH/DELETE /repos/{owner}/{repo}
    if sub_path in ("", "/") and method == "GET":
        repo.updated_at = store.clock.tick()
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(repo_json(repo), separators=(",", ":")).encode()
    
    if sub_path in ("", "/") and method == "PATCH":
        if "description" in json_body:
            repo.description = json_body["description"] or ""
        if "name" in json_body:
            new_name = json_body["name"].strip()
            if new_name and new_name != repo.name:
                if not store.get_repo(owner, new_name):
                    # Rekey in store
                    del store.repos[(owner, repo.name)]
                    repo.name = new_name
                    store.repos[(owner, new_name)] = repo
        if "private" in json_body:
            repo.private = bool(json_body["private"])
        if "default_branch" in json_body:
            repo.default_branch = json_body["default_branch"]
        repo.updated_at = store.clock.tick()
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(repo_json(repo), separators=(",", ":")).encode()
    
    if sub_path in ("", "/") and method == "DELETE":
        store.delete_repo(owner, repo_name)
        return 204, base_headers + [("Content-Type", "application/json; charset=utf-8")], b""
    
    # /branches
    if sub_path == "/branches" and method == "GET":
        branches = []
        for refname, sha in sorted(repo.refs.items()):
            if refname.startswith("refs/heads/"):
                branches.append({
                    "name": refname[11:],
                    "commit": {"sha": sha},
                    "protected": False,
                })
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(branches, separators=(",", ":")).encode()
    
    if sub_path.startswith("/branches/") and method == "GET":
        branch_name = sub_path[10:]
        refname = f"refs/heads/{branch_name}"
        sha = repo.tip(refname)
        if not sha:
            return err(404, "Not Found")
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
            "name": branch_name,
            "commit": {"sha": sha},
            "protected": False,
        }, separators=(",", ":")).encode()
    
    # /commits
    if sub_path == "/commits" and method == "GET":
        sha_param = q("sha")
        if sha_param:
            start_sha = _resolve_ref_name(repo, sha_param)
        else:
            start_sha = repo.tip(f"refs/heads/{repo.default_branch}") or ZERO_SHA
        
        if start_sha == ZERO_SHA:
            return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], b"[]"
        
        page = max(1, q_int("page", 1))
        per_page = min(100, max(1, q_int("per_page", 30)))
        
        commits_list = _walk_commits_paged(repo, start_sha, page, per_page)
        resp_body = json.dumps([commit_json(store, repo, c) for c in commits_list], separators=(",", ":"))
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], resp_body.encode()
    
    if sub_path.startswith("/commits/") and method == "GET":
        sha = sub_path[9:]
        if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
            return err(404, "Not Found")
        if not repo.objects.has(sha):
            return err(404, "Not Found")
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(commit_json(store, repo, sha), separators=(",", ":")).encode()
    
    # /git/refs
    if sub_path == "/git/refs" and method == "POST":
        ref = json_body.get("ref", "")
        sha = json_body.get("sha", "")
        if not ref.startswith("refs/") or not repo.objects.has(sha):
            return err(422, "Invalid ref/sha")
        if ref in repo.refs:
            return err(422, "Reference already exists")
        repo.set_ref(ref, sha)
        obj_type, _ = repo.objects.get(sha)
        return 201, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
            "ref": ref,
            "node_id": gid("Ref", f"{repo.db_id}:{ref}"),
            "object": {"sha": sha, "type": obj_type},
        }, separators=(",", ":")).encode()
    
    if sub_path.startswith("/git/ref/") and method == "GET":
        refname = "refs/" + sub_path[9:]
        sha = repo.tip(refname)
        if not sha:
            return err(404, "Not Found")
        obj_type, _ = repo.objects.get(sha)
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
            "ref": refname,
            "node_id": gid("Ref", f"{repo.db_id}:{refname}"),
            "object": {"sha": sha, "type": obj_type},
        }, separators=(",", ":")).encode()
    
    # /contents/{rest}
    if sub_path.startswith("/contents/") and method == "GET":
        return _handle_contents_get(repo, sub_path[10:], q("ref"), base_headers)
    
    if sub_path.startswith("/contents/") and method == "PUT":
        return _handle_contents_put(store, repo, login, sub_path[10:], json_body, base_headers)
    
    if sub_path.startswith("/contents/") and method == "DELETE":
        return _handle_contents_delete(store, repo, login, sub_path[10:], json_body, base_headers)
    
    # /labels
    if sub_path == "/labels" and method == "POST":
        name = json_body.get("name", "").strip()
        if not name:
            return err(422, "Validation Failed")
        if name in repo.labels:
            return err(422, "Validation Failed")
        label_id = repo.next_label_id
        repo.next_label_id += 1
        label = {
            "id": label_id,
            "name": name,
            "color": json_body.get("color", "ededed"),
            "description": json_body.get("description", ""),
        }
        repo.labels[name] = label
        return 201, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(label_json(label), separators=(",", ":")).encode()
    
    if sub_path == "/labels" and method == "GET":
        labels = [label_json(l) for l in sorted(repo.labels.values(), key=lambda x: x["id"])]
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(labels, separators=(",", ":")).encode()
    
    # /issues
    if sub_path == "/issues" and method == "POST":
        return _handle_issues_post(store, repo, login, json_body, base_headers)
    
    if sub_path == "/issues" and method == "GET":
        return _handle_issues_get(repo, query, base_headers)
    
    if sub_path.startswith("/issues/") and not sub_path.startswith("/issues//"):
        return _handle_issue_detail(store, repo, login, sub_path, method, query, json_body, base_headers)
    
    # /pulls
    if sub_path == "/pulls" and method == "POST":
        return _handle_pulls_post(store, repo, login, json_body, base_headers)
    
    if sub_path == "/pulls" and method == "GET":
        return _handle_pulls_get(repo, query, base_headers)
    
    if sub_path.startswith("/pulls/") and not sub_path.startswith("/pulls//"):
        return _handle_pull_detail(store, repo, login, sub_path, method, query, json_body, base_headers)
    
    return err(404, "Not Found")


def _handle_contents_get(
    repo: Repo,
    rest: str,
    ref: Optional[str],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle GET /contents/{rest}."""
    sha = _resolve_commit_sha(repo, ref)
    if sha == ZERO_SHA:
        return err(404, "Not Found")
    
    obj_type, data = repo.objects.get(sha)
    if obj_type != "commit":
        return err(404, "Not Found")
    parsed = parse_commit(data)
    tree_sha = parsed["tree"]
    
    path = unquote(rest).strip("/")
    
    if not path:
        # Root listing
        entries = tree_list(repo.objects, tree_sha, "")
        if entries is None:
            return err(404, "Not Found")
        result = []
        for mode, name, entry_sha in sorted(entries, key=lambda x: x[1]):
            if mode == "40000":
                result.append({
                    "name": name,
                    "path": name,
                    "sha": entry_sha,
                    "size": 0,
                    "type": "dir",
                })
            else:
                _, blob_data = repo.objects.get(entry_sha)
                result.append({
                    "name": name,
                    "path": name,
                    "sha": entry_sha,
                    "size": len(blob_data),
                    "type": "file",
                })
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(result, separators=(",", ":")).encode()
    
    # Lookup specific path
    lookup = tree_lookup(repo.objects, tree_sha, path)
    if not lookup:
        return err(404, "Not Found")
    mode, entry_sha = lookup
    if mode == "40000":
        # Directory listing
        entries = tree_list(repo.objects, entry_sha, "")
        if entries is None:
            return err(404, "Not Found")
        result = []
        for m, name, e_sha in sorted(entries, key=lambda x: x[1]):
            if m == "40000":
                result.append({
                    "name": name,
                    "path": f"{path}/{name}",
                    "sha": e_sha,
                    "size": 0,
                    "type": "dir",
                })
            else:
                _, blob_data = repo.objects.get(e_sha)
                result.append({
                    "name": name,
                    "path": f"{path}/{name}",
                    "sha": e_sha,
                    "size": len(blob_data),
                    "type": "file",
                })
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(result, separators=(",", ":")).encode()
    else:
        # File
        _, blob_data = repo.objects.get(entry_sha)
        content_b64 = base64.b64encode(blob_data).decode()
        return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
            "name": path.split("/")[-1],
            "path": path,
            "sha": entry_sha,
            "size": len(blob_data),
            "type": "file",
            "encoding": "base64",
            "content": content_b64,
        }, separators=(",", ":")).encode()


def _handle_contents_put(
    store: Store,
    repo: Repo,
    login: str,
    rest: str,
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle PUT /contents/{rest}."""
    path = unquote(rest).strip("/")
    if not path:
        return err(422, "Invalid path")
    
    content_b64 = json_body.get("content")
    if content_b64 is None:
        return err(422, "Missing content")
    try:
        content = base64.b64decode(content_b64)
    except Exception:
        return err(422, "Invalid base64")
    
    message = json_body.get("message") or f"Update {path}"
    branch = json_body.get("branch", repo.default_branch)
    given_sha = json_body.get("sha")
    
    branch_ref = f"refs/heads/{branch}"
    tip_sha = repo.tip(branch_ref)
    
    # Get current tree
    if tip_sha:
        _, tip_data = repo.objects.get(tip_sha)
        parsed = parse_commit(tip_data)
        old_tree = parsed["tree"]
    else:
        old_tree = None
    
    # Check existing blob
    existing_sha = None
    if old_tree:
        lookup = tree_lookup(repo.objects, old_tree, path)
        if lookup:
            existing_mode, existing_sha = lookup
    
    if existing_sha:
        if given_sha and given_sha != existing_sha:
            return err(409, "Conflict: the sha does not match")
    elif given_sha:
        return err(409, "Conflict: the sha does not match")
    
    # Add new blob
    blob_sha = repo.objects.add("blob", content)
    
    # Update tree
    new_tree = tree_update(repo.objects, old_tree, path, blob_sha)
    
    # Create commit
    ts = store.clock.tick()
    ident = git_ident(login, ts)
    parents = [tip_sha] if tip_sha else []
    commit_data = build_commit(new_tree, parents, ident, ident, message)
    commit_sha = repo.objects.add("commit", commit_data)
    
    # Update ref
    repo.set_ref(branch_ref, commit_sha)
    repo.updated_at = store.clock.tick()
    
    # Response
    file_entry = {
        "name": path.split("/")[-1],
        "path": path,
        "sha": blob_sha,
        "size": len(content),
        "type": "file",
    }
    status = 200 if existing_sha else 201
    return status, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
        "content": file_entry,
        "commit": {"sha": commit_sha},
    }, separators=(",", ":")).encode()


def _handle_contents_delete(
    store: Store,
    repo: Repo,
    login: str,
    rest: str,
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle DELETE /contents/{rest}."""
    path = unquote(rest).strip("/")
    if not path:
        return err(422, "Invalid path")
    
    message = json_body.get("message") or f"Delete {path}"
    branch = json_body.get("branch", repo.default_branch)
    given_sha = json_body.get("sha")
    
    branch_ref = f"refs/heads/{branch}"
    tip_sha = repo.tip(branch_ref)
    
    if not tip_sha:
        return err(404, "Not Found")
    
    _, tip_data = repo.objects.get(tip_sha)
    parsed = parse_commit(tip_data)
    old_tree = parsed["tree"]
    
    # Check existing blob
    lookup = tree_lookup(repo.objects, old_tree, path)
    if not lookup:
        return err(404, "Not Found")
    existing_mode, existing_sha = lookup
    
    if given_sha != existing_sha:
        return err(409, "Conflict: the sha does not match")
    
    # Update tree (delete)
    new_tree = tree_update(repo.objects, old_tree, path, None)
    
    # Create commit
    ts = store.clock.tick()
    ident = git_ident(login, ts)
    commit_data = build_commit(new_tree, [tip_sha], ident, ident, message)
    commit_sha = repo.objects.add("commit", commit_data)
    
    # Update ref
    repo.set_ref(branch_ref, commit_sha)
    repo.updated_at = store.clock.tick()
    
    return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
        "content": None,
        "commit": {"sha": commit_sha},
    }, separators=(",", ":")).encode()


def _handle_issues_post(
    store: Store,
    repo: Repo,
    login: str,
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle POST /issues."""
    title = json_body.get("title", "").strip()
    if not title:
        return err(422, "Validation Failed")
    
    labels = json_body.get("labels", [])
    for l in labels:
        if l not in repo.labels:
            return err(422, "Validation Failed")
    
    ts = store.clock.tick()
    number = repo.new_number()
    issue = {
        "number": number,
        "db_id": store.new_issue_db(),
        "title": title,
        "body": json_body.get("body", "") or "",
        "state": "open",
        "labels": labels,
        "user": login,
        "created_at": ts,
        "updated_at": ts,
        "closed_at": None,
        "comments": [],
    }
    repo.issues[number] = issue
    return 201, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(issue_json(repo, issue), separators=(",", ":")).encode()


def _handle_issues_get(
    repo: Repo,
    query: Dict[str, List[str]],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle GET /issues."""
    state = query.get("state", ["open"])[0]
    sort = query.get("sort", ["created"])[0]
    direction = query.get("direction", ["desc"])[0]
    
    issues_list = list(repo.issues.values())
    if state == "open":
        issues_list = [i for i in issues_list if i["state"] == "open"]
    elif state == "closed":
        issues_list = [i for i in issues_list if i["state"] == "closed"]
    elif state == "all":
        pass
    else:
        issues_list = [i for i in issues_list if i["state"] == "open"]
    
    if sort == "created":
        issues_list.sort(key=lambda x: (x["created_at"], x["number"]))
    elif sort == "updated":
        issues_list.sort(key=lambda x: (x["updated_at"], x["number"]))
    elif sort == "comments":
        issues_list.sort(key=lambda x: (len(x["comments"]), x["number"]))
    
    if direction == "desc":
        issues_list.reverse()
    
    return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps([issue_json(repo, i) for i in issues_list], separators=(",", ":")).encode()


def _handle_issue_detail(
    store: Store,
    repo: Repo,
    login: str,
    sub_path: str,
    method: str,
    query: Dict[str, List[str]],
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle /issues/{number} and /issues/{number}/comments."""
    parts = sub_path.split("/")
    try:
        number = int(parts[2])
    except (IndexError, ValueError):
        return err(404, "Not Found")
    
    issue = repo.issues.get(number)
    if not issue:
        return err(404, "Not Found")
    
    if len(parts) == 3:
        # /issues/{number}
        if method == "GET":
            return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(issue_json(repo, issue), separators=(",", ":")).encode()
        if method == "PATCH":
            if "state" in json_body:
                new_state = json_body["state"]
                if new_state not in ("open", "closed"):
                    return err(422, "Validation Failed")
                issue["state"] = new_state
                if new_state == "closed" and issue["closed_at"] is None:
                    issue["closed_at"] = store.clock.tick()
            if "title" in json_body:
                issue["title"] = json_body["title"]
            if "body" in json_body:
                issue["body"] = json_body["body"] or ""
            if "labels" in json_body:
                labels = json_body["labels"]
                for l in labels:
                    if l not in repo.labels:
                        return err(422, "Validation Failed")
                issue["labels"] = labels
            issue["updated_at"] = store.clock.tick()
            return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(issue_json(repo, issue), separators=(",", ":")).encode()
        return err(404, "Not Found")
    
    if len(parts) == 4 and parts[3] == "comments":
        # /issues/{number}/comments
        if method == "GET":
            comments = [comment_json(c) for c in issue["comments"]]
            return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(comments, separators=(",", ":")).encode()
        if method == "POST":
            body = json_body.get("body", "")
            comment = {
                "id": store.new_comment_id(),
                "body": body,
                "user": login,
                "created_at": store.clock.tick(),
                "updated_at": store.clock.tick(),
            }
            issue["comments"].append(comment)
            issue["updated_at"] = store.clock.tick()
            return 201, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(comment_json(comment), separators=(",", ":")).encode()
    
    return err(404, "Not Found")


def _handle_pulls_post(
    store: Store,
    repo: Repo,
    login: str,
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle POST /pulls."""
    title = json_body.get("title", "").strip()
    if not title:
        return err(422, "Validation Failed")
    
    head = json_body.get("head", "").strip()
    base = json_body.get("base", "").strip()
    if not head or not base:
        return err(422, "Validation Failed")
    
    # Extract branch name (after last :)
    head_ref = head.split(":")[-1]
    base_ref = base.split(":")[-1]
    
    head_sha = repo.tip(f"refs/heads/{head_ref}")
    base_sha = repo.tip(f"refs/heads/{base_ref}")
    
    if not head_sha or not base_sha:
        return err(422, "Validation Failed")
    
    ts = store.clock.tick()
    number = repo.new_number()
    pull = {
        "number": number,
        "db_id": store.new_pull_db(),
        "title": title,
        "body": json_body.get("body", "") or "",
        "state": "open",
        "head_ref": head_ref,
        "base_ref": base_ref,
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
    return 201, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(pull_json(repo, pull), separators=(",", ":")).encode()


def _handle_pulls_get(
    repo: Repo,
    query: Dict[str, List[str]],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle GET /pulls."""
    state = query.get("state", ["open"])[0]
    
    pulls_list = list(repo.pulls.values())
    if state == "open":
        pulls_list = [p for p in pulls_list if p["state"] == "open"]
    elif state == "closed":
        pulls_list = [p for p in pulls_list if p["state"] == "closed"]
    elif state == "all":
        pass
    else:
        pulls_list = [p for p in pulls_list if p["state"] == "open"]
    
    pulls_list.sort(key=lambda x: x["number"], reverse=True)
    
    return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps([pull_json(repo, p) for p in pulls_list], separators=(",", ":")).encode()


def _handle_pull_detail(
    store: Store,
    repo: Repo,
    login: str,
    sub_path: str,
    method: str,
    query: Dict[str, List[str]],
    json_body: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle /pulls/{number}, /pulls/{number}/files, /pulls/{number}/merge."""
    parts = sub_path.split("/")
    try:
        number = int(parts[2])
    except (IndexError, ValueError):
        return err(404, "Not Found")
    
    pull = repo.pulls.get(number)
    if not pull:
        return err(404, "Not Found")
    
    if len(parts) == 3:
        # /pulls/{number}
        if method == "GET":
            return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(pull_json(repo, pull), separators=(",", ":")).encode()
        if method == "PATCH":
            if "state" in json_body:
                new_state = json_body["state"]
                if new_state not in ("open", "closed"):
                    return err(422, "Validation Failed")
                pull["state"] = new_state
            pull["updated_at"] = store.clock.tick()
            return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(pull_json(repo, pull), separators=(",", ":")).encode()
        return err(404, "Not Found")
    
    if len(parts) == 4 and parts[3] == "files":
        # /pulls/{number}/files
        return _handle_pull_files(repo, pull, base_headers)
    
    if len(parts) == 4 and parts[3] == "merge":
        # /pulls/{number}/merge
        if method != "PUT":
            return err(405, "Method Not Allowed")
        return _handle_pull_merge(store, repo, login, pull, base_headers)
    
    return err(404, "Not Found")


def _handle_pull_files(
    repo: Repo,
    pull: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle GET /pulls/{number}/files."""
    base_sha = pull["base_sha"]
    head_sha = pull["head_sha"]
    
    if not repo.objects.has(base_sha) or not repo.objects.has(head_sha):
        return err(404, "Not Found")
    
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
            status = "added"
            additions = 0
            if new_sha and repo.objects.has(new_sha):
                _, blob = repo.objects.get(new_sha)
                lines = blob.decode("utf-8", errors="replace").splitlines()
                additions = len(lines)
            files.append({
                "filename": path,
                "status": status,
                "additions": additions,
                "deletions": 0,
                "changes": additions,
                "sha": new_sha,
            })
        elif new_sha is None:
            # Removed
            status = "removed"
            deletions = 0
            if old_sha and repo.objects.has(old_sha):
                _, blob = repo.objects.get(old_sha)
                lines = blob.decode("utf-8", errors="replace").splitlines()
                deletions = len(lines)
            files.append({
                "filename": path,
                "status": status,
                "additions": 0,
                "deletions": deletions,
                "changes": deletions,
                "sha": old_sha,
            })
        else:
            # Modified
            status = "modified"
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
                "filename": path,
                "status": status,
                "additions": additions,
                "deletions": deletions,
                "changes": additions + deletions,
                "sha": new_sha,
            })
    
    return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps(files, separators=(",", ":")).encode()


def _handle_pull_merge(
    store: Store,
    repo: Repo,
    login: str,
    pull: Dict[str, Any],
    base_headers: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle PUT /pulls/{number}/merge."""
    if pull.get("merged", False) or pull["state"] == "closed":
        return err(405, "Pull Request is not mergeable")
    
    head_ref = f"refs/heads/{pull['head_ref']}"
    base_ref = f"refs/heads/{pull['base_ref']}"
    
    head_tip = repo.tip(head_ref)
    base_tip = repo.tip(base_ref)
    
    if not head_tip or not base_tip:
        return err(405, "Pull Request is not mergeable")
    
    if is_ancestor(repo.objects, head_tip, base_tip):
        return err(405, "Pull Request is not mergeable")
    
    # Find merge base: BFS from head_tip, first commit in base ancestors
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
    
    if not merge_base:
        return err(405, "Pull Request is not mergeable")
    
    # Three-way merge
    mb_flat = flatten_tree(repo.objects, merge_base) if repo.objects.has(merge_base) else {}
    b_flat = flatten_tree(repo.objects, repo.objects.get(base_tip)[1])
    h_flat = flatten_tree(repo.objects, repo.objects.get(head_tip)[1])
    
    # Get merge base tree
    if repo.objects.has(merge_base):
        _, mb_data = repo.objects.get(merge_base)
        mb_parsed = parse_commit(mb_data)
        mb_tree = mb_parsed["tree"]
        mb_flat = flatten_tree(repo.objects, mb_tree)
    else:
        mb_flat = {}
    
    if repo.objects.has(base_tip):
        _, b_data = repo.objects.get(base_tip)
        b_parsed = parse_commit(b_data)
        b_tree = b_parsed["tree"]
        b_flat = flatten_tree(repo.objects, b_tree)
    else:
        b_flat = {}
    
    if repo.objects.has(head_tip):
        _, h_data = repo.objects.get(head_tip)
        h_parsed = parse_commit(h_data)
        h_tree = h_parsed["tree"]
        h_flat = flatten_tree(repo.objects, h_tree)
    else:
        h_flat = {}
    
    # Merge
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
                return err(405, "Pull Request is not mergeable")
            
            mb_blob = repo.objects.get_data(mb_sha).decode("utf-8", errors="replace")
            b_blob = repo.objects.get_data(b_sha).decode("utf-8", errors="replace")
            h_blob = repo.objects.get_data(h_sha).decode("utf-8", errors="replace")
            
            mb_lines = mb_blob.splitlines()
            b_lines = b_blob.splitlines()
            h_lines = h_blob.splitlines()
            
            # Get changes as spans
            b_matcher = difflib.SequenceMatcher(None, mb_lines, b_lines)
            h_matcher = difflib.SequenceMatcher(None, mb_lines, h_lines)
            
            b_spans = []
            for tag, i1, i2, j1, j2 in b_matcher.get_opcodes():
                if tag != "equal":
                    b_spans.append((i1, i2, b_lines[j1:j2]))
            
            h_spans = []
            for tag, i1, i2, j1, j2 in h_matcher.get_opcodes():
                if tag != "equal":
                    h_spans.append((i1, i2, h_lines[j1:j2]))
            
            # Check for overlapping spans with different replacements
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
                return err(405, "Pull Request is not mergeable")
            
            # Apply both sets of changes
            result_lines = list(mb_lines)
            offset = 0
            all_spans = sorted(b_spans + h_spans, key=lambda x: x[0])
            for i1, i2, repl in all_spans:
                # Apply at adjusted position
                adj_i1 = i1 + offset
                adj_i2 = i2 + offset
                result_lines[adj_i1:adj_i2] = repl
                offset += len(repl) - (i2 - i1)
            
            result_blob = "\n".join(result_lines).encode("utf-8")
            result_sha = repo.objects.add("blob", result_blob)
            result_files[path] = result_sha
    
    # Build result tree
    result_tree_sha = tree_from_flat(repo.objects, result_files)
    
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
    pull["updated_at"] = ts2
    
    return 200, base_headers + [("Content-Type", "application/json; charset=utf-8")], json.dumps({
        "sha": merge_sha,
        "merged": True,
        "message": "Pull Request successfully merged",
    }, separators=(",", ":")).encode()


def _handle_graphql(
    store: Store,
    method: str,
    path: str,
    query: Dict[str, List[str]],
    headers: List[Tuple[str, str]],
    body: bytes,
    base_headers: List[Tuple[str, str]],
    login: Optional[str],
    token: Optional[str],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle /api/graphql."""
    if login is None:
        return err(401, "Requires authentication")
    # Lazy import
    try:
        import gql
        status, obj = gql.handle(store, login, body)
        resp_body = json.dumps(obj, separators=(",", ":"))
        return status, base_headers + [("Content-Type", "application/json; charset=utf-8")], resp_body.encode()
    except ImportError:
        return err(501, "GraphQL not implemented")


def _handle_mcp(
    store: Store,
    method: str,
    path: str,
    query: Dict[str, List[str]],
    headers: Dict[str, str],
    body: bytes,
    base_headers: List[Tuple[str, str]],
    login: Optional[str],
    token: Optional[str],
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    """Handle /mcp."""
    if not any(k.lower() == "authorization" for k in headers):
        return err(401, "Requires authentication")
    if method != "POST":
        return err(405, "Method Not Allowed")
    # Lazy import
    try:
        import mcp
        status, extra_headers, resp_body = mcp.handle(store, login, headers, body)
        all_headers = base_headers + [("Content-Type", "application/json; charset=utf-8")] + extra_headers
        return status, all_headers, resp_body
    except ImportError:
        return err(501, "MCP not implemented")