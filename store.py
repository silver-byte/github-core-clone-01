"""
In-memory data store for the git forge.
Imports ObjectStore from gitcore.
"""

import os
import time
import hashlib
from typing import Dict, List, Optional, Any

from gitcore import ObjectStore


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