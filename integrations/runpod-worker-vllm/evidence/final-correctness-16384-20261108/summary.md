| Arm | Phase | Total s | Tok/s | TTFT mean | TTFT p50 | TTFT p95 | First warm TTFT | Later warm p50 | External hit evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (no cache) | cold | 4.692 | 13.641 | 2.812 | 2.812 | 2.812 | unreported | unreported | external hit rate=unreported |
| A (no cache) | warm | 72.390 | 17.682 | 35.659 | 35.098 | 64.394 | 4.953 | 36.504 | external hit rate=unreported |
| B (filesystem) | cold | 4.743 | 13.492 | 2.866 | 2.866 | 2.866 | unreported | unreported | vLLM external hit rate=0.0% |
| B (filesystem) | warm | 26.378 | 48.526 | 17.936 | 17.506 | 20.797 | 14.517 | 17.543 | vLLM external hit rate=88.8% |
| C (Telys) | cold | 7.417 | 8.629 | 4.712 | 4.712 | 4.712 | unreported | unreported | vLLM external hit rate=0.0%; Telys last lookup hit_blocks=0,hit_tokens=0 |
| C (Telys) | warm | 42.247 | 30.298 | 28.955 | 30.584 | 36.811 | 16.648 | 30.603 | vLLM external hit rate=88.8%; Telys last lookup hit_blocks=1077,hit_tokens=17232 |

| Arm | Correctness | Fresh-process warm output vs no-cache reference |
|---|---|---|
| A (no cache) | pass | reference |
| B (filesystem) | pass | match |
| C (Telys) | pass | match |
