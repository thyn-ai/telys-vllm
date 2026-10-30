## Telemetry summary (262 records, 10 phases)

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| wait_for_save | 82 | 28325.313 | 0.619 | 345.431 | 2041.399 | `/root/bench/telys-vllm/telys_vllm/connector.py:584` |
| save_collect | 60 | 22056.645 | 0.007 | 367.611 | 1533.821 | `/root/bench/telys-vllm/telys_vllm/connector.py:609` |
| save_encode_persist | 16 | 5801.062 | 296.361 | 362.566 | 759.482 | `/root/bench/telys-vllm/telys_vllm/connector.py:621` |
| save_encode_chunk | 20 | 5685.253 | 293.692 | 284.263 | 451.620 | `/root/bench/telys-vllm/telys_vllm/connector.py:634` |
| store_persist | 20 | 4214.848 | 217.730 | 210.742 | 382.994 | `/root/bench/telys-vllm/telys_vllm/store.py:468` |
| telys_startup | 2 | 12.209 | 6.104 | 6.104 | 12.046 | `/root/bench/telys-vllm/telys_vllm/connector.py:156` |
| lookup | 20 | 4.802 | 0.233 | 0.240 | 0.597 | `/root/bench/telys-vllm/telys_vllm/connector.py:233` |
| connector_store_init | 2 | 0.921 | 0.460 | 0.460 | 0.493 | `/root/bench/telys-vllm/telys_vllm/connector.py:172` |
| save_chunk_async | 20 | 0.887 | 0.045 | 0.044 | 0.056 | `/root/bench/telys-vllm/telys_vllm/connector.py:638` |
| store_lookup_prefix | 20 | 0.386 | 0.019 | 0.019 | 0.041 | `/root/bench/telys-vllm/telys_vllm/store.py:172` |

### Top 3 contributors by total time
- **wait_for_save**: 28325.313 ms (42.9%)
- **save_collect**: 22056.645 ms (33.4%)
- **save_encode_persist**: 5801.062 ms (8.8%)

### Dominant phase
- phase: `wait_for_save`
- total: 28325.313 ms
- median: 0.619 ms
- dominant caller: `/root/bench/telys-vllm/telys_vllm/connector.py:584`
