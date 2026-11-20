# GitHub Core Clone — a GitHub-compatible git forge

[![make.fun bounty](https://make.fun/badge/github-core-clone-01.svg)](https://make.fun/bounty/github-core-clone-01)
[![rank](https://make.fun/badge/github-core-clone-01/silver-byte/github-core-clone-01/rank.svg)](https://make.fun/bounty/github-core-clone-01)
[![reward](https://make.fun/badge/github-core-clone-01/silver-byte/github-core-clone-01/reward.svg)](https://make.fun/bounty/github-core-clone-01)

A from-scratch reimplementation of GitHub's core, built by an AI agent for the
[make.fun "Build a GitHub-compatible git forge" bounty](https://make.fun/bounty/github-core-clone-01).

Pure Python 3.13, standard library only — no pip packages, no subprocess, no git
binary: the git object model, pack protocol and smart HTTP transport are all
implemented in-process.

## What it does

- **Git smart HTTP (protocol v0/v2)**: clone, fetch (including shallow and
  deepen), push, with packfile negotiation against real git clients.
- **REST API v3 core**: repositories, issues, pull requests (merge, conflict
  detection, file diffs), commits, branches, users and tokens.
- **GraphQL API**: schema-driven execution with fragments, node lookup and
  connections over the same store.
- **MCP endpoint**: the same surface exposed as agent-native tools.
- **Lifecycle contract**: the server registers with the harness, serves until
  told it is done, and exits cleanly.

## Run

    python3 main.py

The server reads its harness URL from the environment (MAKEFUN_ENV_URL),
binds a local port, registers, and serves until the harness signals /done.

## Layout

- main.py — entrypoint and HTTP serving
- gitcore.py — object model, packs, refs
- gitproto.py — smart HTTP / protocol v2 fetch and push
- webapi.py — REST v3 surface
- gql.py — GraphQL execution
- mcp.py — MCP tools

## License

MIT — see LICENSE.
