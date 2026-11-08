# Final fixed-source validation

Overall status: **FAIL**

PR head: `6cbfe82af264f87bc78da067b8ad7a5254a8a6e1`
Run ID: `final-correctness-20261108-01` for both fresh profiles.

Lifecycle probe: **PASS**; cold 1081/1081 and aligned 1104/1104 expected blocks saved.

## Profile 16384: PASS

| Arm | Phase | Total s | TTFT mean | TTFT p50 | TTFT p95 |
|---|---|---:|---:|---:|---:|
| A | cold | 4.691666842 | 2.812189579 | 2.812189579 | 2.812189579 |
| A | warm | 72.390316963 | 35.659237671 | 35.097700953 | 64.394146144 |
| B | cold | 4.743439436 | 2.866423845 | 2.866423845 | 2.866423845 |
| B | warm | 26.377587080 | 17.935920501 | 17.506097674 | 20.797282410 |
| C | cold | 7.416734457 | 4.711503744 | 4.711503744 | 4.711503744 |
| C | warm | 42.246800184 | 28.955096138 | 30.584477782 | 36.810592473 |

Ordinary correctness: A=pass,B=pass,C=pass; independent output equality/worker checks: True.

cold verify: status=pass; expected=1081; verified=1081; missing=0; resolved shapes=[[16, 4, 256]].
warm verify: status=pass; expected=24333; verified=24333; missing=0; resolved shapes=[[16, 4, 256]].

Worker/engine IDs:
- cold: pid=63662 engine_id=b362e890-c813-455e-be9d-ccb03c10216a lookups=1
- warm: pid=64177 engine_id=619df941-324e-4596-b988-238e87ef987a lookups=20
- run1: pid=66038 engine_id=98b23a8e-ca24-419a-9487-45ce9d6c1292 lookups=2
- run2: pid=66477 engine_id=5445702b-1b42-4f09-9a01-2884cc5222b0 lookups=2

## Profile 4096: PASS

| Arm | Phase | Total s | TTFT mean | TTFT p50 | TTFT p95 |
|---|---|---:|---:|---:|---:|
| A | cold | 2.372623444 | 0.584506512 | 0.584506512 | 0.584506512 |
| A | warm | 20.540916920 | 10.332960808 | 10.267210603 | 17.662422812 |
| B | cold | 2.384196281 | 0.595099926 | 0.595099926 | 0.595099926 |
| B | warm | 13.027897358 | 7.611747134 | 7.279286742 | 9.954432440 |
| C | cold | 3.132462263 | 1.173944235 | 1.173944235 | 1.173944235 |
| C | warm | 21.778853893 | 14.401298046 | 15.382671595 | 18.271852839 |

Ordinary correctness: A=pass,B=pass,C=pass; independent output equality/worker checks: True.

Dynamic persistence: cold 273/273; warm verified expectation 8056; warm new 2676/2676; total store 2949/2949.
Warm prompt/generated accounting: {"generated_blocks": 80, "new_blocks": 2676, "persisted_blocks": 8056, "prompt_boundary_blocks": 7976, "reused_blocks": 5380}

Worker/engine IDs:
- cold: pid=69115 engine_id=1bc16453-c860-4ed7-b586-d35f3c023500 lookups=1
- warm: pid=69551 engine_id=9517ce98-3d7d-418e-95c5-a12ec1b7c085 lookups=20
- run1: pid=71340 engine_id=feffcef6-ca21-4aa3-a4f6-c725252f2c2d lookups=2
- run2: pid=71763 engine_id=678e20d7-72ef-4a23-af9c-b591e555be6a lookups=2

## Focused compatibility outcomes
- Dual-codec compatibility: FAIL (legacy_to_fast_decode and legacy_decode both leak UnicodeDecodeError; see focused/lifecycle-result.json).
- Overlap payload visibility: PASS, but deletion leaves stale data in the other client cache (first_get_present=true, first_contains=true).
- v2 meta_blob: FAIL (payload is empty and metadata contains default/null fields rather than the nested blob).
- Reopened-reset and resumed/divergence checks: PASS.

8192 was not run. No endpoint, scale-to-zero, release, commit, push, or external post was performed. GPU final state: `NVIDIA A40, 0 MiB, 46068 MiB, 0 %`; compute apps: ``.
