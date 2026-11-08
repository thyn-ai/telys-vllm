| Arm | Phase | Total s | Tok/s | TTFT mean | TTFT p50 | TTFT p95 | First warm TTFT | Later warm p50 | External hit evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (no cache) | cold | 2.373 | 26.974 | 0.585 | 0.585 | 0.585 | unreported | unreported | external hit rate=unreported |
| A (no cache) | warm | 20.541 | 62.315 | 10.333 | 10.267 | 17.662 | 1.592 | 10.285 | external hit rate=unreported |
| B (filesystem) | cold | 2.384 | 26.843 | 0.595 | 0.595 | 0.595 | unreported | unreported | vLLM external hit rate=0.0% |
| B (filesystem) | warm | 13.028 | 98.251 | 7.612 | 7.279 | 9.954 | 5.400 | 7.555 | vLLM external hit rate=67.4% |
| C (Telys) | cold | 3.132 | 20.431 | 1.174 | 1.174 | 1.174 | unreported | unreported | vLLM external hit rate=0.0%; Telys last lookup hit_blocks=0,hit_tokens=0 |
| C (Telys) | warm | 21.779 | 58.773 | 14.401 | 15.383 | 18.272 | 6.792 | 17.167 | vLLM external hit rate=67.4%; Telys last lookup hit_blocks=269,hit_tokens=4304 |

| Arm | Correctness | Fresh-process warm output vs no-cache reference |
|---|---|---|
| A (no cache) | pass | reference |
| B (filesystem) | pass | match |
| C (Telys) | pass | match |
