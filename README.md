# github-core-clone-01 - a GitHub-compatible git forge

A single-process Python 3.13 standard-library forge implementing git smart HTTP, REST v3, GraphQL and MCP surfaces over one in-memory store.

## Running

```bash
export MAKEFUN_ENV_URL=http://localhost:8080
export MAKEFUN_SHOP_TOKEN=your-token
python3 main.py
```

The process registers its port with the environment service and exits when the environment reports done.

## Architecture

| Module | Responsibility |
|--------|----------------|
| `main.py` | Lifecycle entrypoint, HTTP server, environment registration |
| `webapi.py` | Routing + REST v3 surface, git smart-HTTP endpoints |
| `gitproto.py` | Git smart-HTTP wire protocol (pkt-line, v0/v2, fetch, receive-pack) |
| `gitcore.py` | Git object/pack engine (hash, object store, pack read/write, reachability) |
| `gql.py` | GraphQL subset executor (queries, mutations, introspection) |
| `mcp.py` | MCP JSON-RPC tools endpoint sharing the store |
| `store.py` | In-memory data store with logical clock |

## Surfaces

- **Git smart HTTP**: `git-upload-pack` / `git-receive-pack` v0 + protocol v2 with `ls-refs`, `fetch`, and delta support
- **REST API v3**: `/api/v3` — repos, contents, issues, pulls, labels, git refs
- **GraphQL**: `/api/graphql` — queries, mutations, introspection (subset)
- **MCP**: `/mcp` — JSON-RPC 2.0 tools endpoint (create_repo, get_repo, create_issue, list_issues, create_pull_request, merge_pull_request)

## License

MIT — see [LICENSE](LICENSE).