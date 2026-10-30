# Contributing to the Telys vLLM Connector

Thank you for your interest in contributing. This repository holds the
**open-source connector** that plugs vLLM into Telys for restart-persistent,
cross-engine KV/prefix caching. It is meant to be forked, read, and improved by
anyone.

The Telys engine itself (the signed runtime the connector talks to) is closed
and distributed separately — `telys login` fetches it at install time. Nothing
in this repository grants access to it, and nothing you contribute here can
change licensed capacity or engine-side enforcement. See [SECURITY.md](./SECURITY.md)
for the trust boundary this implies for security reports.

## What you can contribute

| Area | Status | Notes |
|------|--------|-------|
| `telys_vllm/` | ✅ Open | Connector logic, codecs, GPU/injection helpers, store adapters |
| `tests/` | ✅ Open | Unit and integration tests |
| `bench/` | ✅ Open | Benchmarks, telemetry, and result summaries |
| Examples / docs | ✅ Open | Corrections, new guides |
| The Telys engine runtime | 🔒 Closed | Not in this repository — see above |

## Getting started

```bash
git clone https://github.com/thyn-ai/telys-vllm.git
cd telys-vllm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,vllm]"
pytest
```

The connector's tests exercise the public integration surface without requiring
a live Telys license, but full end-to-end prefix-cache runs need the signed
Telys runtime installed (`telys login`).

## Development workflow

### Branch naming
- `feat/short-description` — new feature
- `fix/short-description` — bug fix
- `docs/short-description` — documentation only
- `bench/short-description` — benchmark or telemetry change

### Commit messages
We follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(connector): add batched KV injection helper
fix(store): handle empty slab fallback in BlobStore mode
docs(readme): document multi-replica warm-start
```

### Pull request checklist
- [ ] New behavior is covered by tests (unit or benchmark)
- [ ] Public interfaces are documented in docstrings or the README
- [ ] No hardcoded credentials or secrets
- [ ] `pytest` and `ruff check telys_vllm tests bench` pass locally

All required checks must pass, including on forked-repository pull requests —
CI runs with no secrets and no elevated permissions, so it's safe to run
automatically on every PR.

## Recognizing contributors

This project follows the [all-contributors](https://allcontributors.org)
specification: everyone who contributes — code, docs, bug reports, reviews,
or any other [contribution type](https://allcontributors.org/docs/en/emoji-key) —
is recognized in the [README](./README.md#contributors). Maintainers add
contributors by commenting `@all-contributors please add @user for code`
(replacing `code` with the relevant contribution type) on an issue or pull
request, and the bot opens a pull request updating the contributors table.

## Licensing

By submitting a pull request you agree that your contribution is licensed
under the project's [Apache-2.0 license](./LICENSE) (inbound=outbound,
[GitHub Terms of Service §D.6](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license)).

## Reporting issues

- **Security vulnerabilities** → see [SECURITY.md](./SECURITY.md) (do NOT
  open a public issue)
- **Bugs** → [GitHub Issues](https://github.com/thyn-ai/telys-vllm/issues)
  with the `bug` label
- **Feature requests** → GitHub Issues with the `enhancement` label
- **Questions** → GitHub Issues with the `question` label, or
  https://discord.gg/w8NDsph9an

## Community

- Discord: https://discord.gg/w8NDsph9an
- Docs: https://docs.telys.ai
- Email: community@algenta.ai
