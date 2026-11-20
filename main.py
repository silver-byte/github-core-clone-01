"""
Entrypoint for the git forge server.
"""

import os
import sys
import json
import time
import socket
import threading
import urllib.request
import urllib.error
import urllib.parse
import http.server
from typing import Any, Dict, List, Tuple, Optional

from store import new_store_from_env
import webapi


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "forge/1.0"

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
            )
        except OSError:
            pass

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _handle(self) -> None:
        try:
            length_str = self.headers.get("Content-Length", "0")
            try:
                length = int(length_str)
            except ValueError:
                length = 0
            body = self.rfile.read(length) if length else b""

            split = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(split.path)
            query = urllib.parse.parse_qs(split.query)
            headers = {k.lower(): v for k, v in self.headers.items()}

            with self.server.forge_lock:
                status, resp_headers, resp_body = webapi.dispatch(
                    self.server.forge_store,
                    self.command,
                    path,
                    query,
                    headers,
                    body,
                )

            self.send_response(status)
            for k, v in resp_headers:
                if k.lower() == "content-length":
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            if self.command != "HEAD" and resp_body:
                self.wfile.write(resp_body)
        except Exception:
            # If response hasn't started, send 500
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("X-GitHub-Request-Id", "req-err")
                err_body = json.dumps({"message": "internal error"}, separators=(",", ":")).encode()
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(err_body)
            except Exception:
                pass


def _post_register(env_url: str, port: int) -> bool:
    """Register port with environment service."""
    url = f"{env_url}/register"
    data = json.dumps({"port": port}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = time.time() + 150
    backoff = 0.25
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 409):
                    return True
                if resp.status in (400, 405):
                    return False
        except Exception:
            pass
        time.sleep(backoff)
        backoff = min(backoff * 2, 2.0)
    return False


def _wait_done(env_url: str, server: http.server.ThreadingHTTPServer) -> None:
    """Wait for environment to signal done."""
    url = f"{env_url}/done"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read()
                obj = json.loads(data.decode("utf-8"))
                if isinstance(obj, dict) and obj.get("done") is True:
                    return
        except Exception:
            pass
        time.sleep(1)


def main() -> int:
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not env_url:
        print("MAKEFUN_ENV_URL is not set", file=sys.stderr)
        return 1
    env_url = env_url.rstrip("/")

    store = new_store_from_env()
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
    server.daemon_threads = True
    server.forge_store = store
    server.forge_lock = getattr(store, "lock", None) or threading.RLock()

    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if not _post_register(env_url, port):
        server.shutdown()
        return 1

    _wait_done(env_url, server)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())