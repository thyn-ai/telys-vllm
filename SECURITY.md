# Security Policy

This repository contains the open-source **Telys vLLM connector**
(`telys_vllm`). We take its security seriously and appreciate responsible
disclosure from the community.

## Supported versions

Security fixes land on `main` and in the latest published release of the
package (`telys-vllm` on PyPI).

| Channel | Supported |
| --- | --- |
| Latest release / `main` | :white_check_mark: |
| Older tagged releases | Best-effort; please upgrade to the latest |

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for
security problems.** Public disclosure before a fix is available puts other
users at risk.

Report privately through either channel:

1. **GitHub Security Advisories** (preferred) — open a private report from
   this repository's **Security → Report a vulnerability** tab.
2. **Email** — `security@algenta.ai`.

Please include, where possible: a description of the issue and its impact,
the affected component (connector, codec, store adapter, GPU/injection path,
or benchmark harness), steps to reproduce or a proof of concept, and the
package version you tested.

## What to expect

- Acknowledgement within 3 business days.
- An initial assessment and severity triage within 7 business days.
- Regular updates as we work on a fix, and credit in the published advisory
  (unless you prefer to remain anonymous).
- Coordinated disclosure: we agree on a timeline with you and publish a
  GitHub Security Advisory once a fix is available.

## Scope

**In scope** — this repository's own code, including:

- Connector-side handling of KV-cache payloads (serialization, deserialization,
  buffer sizing, injection into vLLM paged caches)
- Transport and auth handling between the connector and a Telys runtime
- The local client, slab/BlobStore adapters, and telemetry surface
- Deserialization, injection, or other memory/logic-safety bugs in connector code
- Insecure defaults in the package

**Out of scope for this repository** (redirect privately to
`security@algenta.ai`, same as above, rather than filing here):

- The closed Telys on-device runtime engine and its own internal enforcement
- The private license-issuance service
- Any private activation or relay infrastructure

We also want to be upfront about the trust model: the connector is designed to
be **assumed untrusted** — a report showing that the connector's own checks can
be bypassed by modifying the connector is informational, not a vulnerability,
unless it also demonstrates that the signed Telys runtime's independent
enforcement was bypassed. The runtime, not this connector, is the sole
authority over licensed capacity and engine-side policy.

**Also out of scope:** third-party dependencies (report those upstream; we
still want to hear how they affect Telys), and social-engineering, physical,
or denial-of-service testing against any hosted environment.

Thank you for helping keep Telys and its users safe.
