## Telemetry summary (1124542 records, 17 phases)

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 4 | 96691.550 | 14405.363 | 24172.888 | 66736.870 | `/root/bench/telys-vllm/telys_vllm/connector.py:404` |
| inject_block | 2828 | 44112.025 | 14.874 | 15.598 | 19.446 | `/root/bench/telys-vllm/telys_vllm/connector.py:473` |
| load_decoded_blocks | 101 | 35866.125 | 343.577 | 355.110 | 413.945 | `/root/bench/telys-vllm/telys_vllm/connector.py:412` |
| store_read_chunk | 101 | 34141.412 | 327.403 | 338.034 | 395.424 | `/root/bench/telys-vllm/telys_vllm/store.py:276` |
| tensor_from_blob | 1118012 | 16726.819 | 0.014 | 0.015 | 0.018 | `/root/bench/telys-vllm/telys_vllm/connector.py:479` |
| store_decode_chunk | 101 | 13551.071 | 125.004 | 134.169 | 173.993 | `/root/bench/telys-vllm/telys_vllm/store.py:307` |
| inject_kv_block | 2828 | 7779.874 | 2.500 | 2.751 | 3.571 | `/root/bench/telys-vllm/telys_vllm/connector.py:485` |
| wait_for_save | 132 | 1340.880 | 0.569 | 10.158 | 0.926 | `/root/bench/telys-vllm/telys_vllm/connector.py:584` |
| save_collect | 88 | 1160.324 | 0.006 | 13.186 | 0.009 | `/root/bench/telys-vllm/telys_vllm/connector.py:609` |
| store_persist | 4 | 131.231 | 27.511 | 32.808 | 73.355 | `/root/bench/telys-vllm/telys_vllm/store.py:468` |
| save_encode_persist | 1 | 120.549 | 120.549 | 120.549 | 120.549 | `/root/bench/telys-vllm/telys_vllm/connector.py:621` |
| save_encode_chunk | 4 | 119.043 | 29.859 | 29.761 | 55.246 | `/root/bench/telys-vllm/telys_vllm/connector.py:634` |
| lookup | 165 | 36.505 | 0.036 | 0.221 | 0.482 | `/root/bench/telys-vllm/telys_vllm/connector.py:233` |
| telys_startup | 2 | 13.227 | 6.613 | 6.613 | 12.932 | `/root/bench/telys-vllm/telys_vllm/connector.py:156` |
| store_lookup_prefix | 165 | 1.764 | 0.005 | 0.011 | 0.023 | `/root/bench/telys-vllm/telys_vllm/store.py:172` |
| connector_store_init | 2 | 0.952 | 0.476 | 0.476 | 0.536 | `/root/bench/telys-vllm/telys_vllm/connector.py:172` |
| save_chunk_async | 4 | 0.145 | 0.030 | 0.036 | 0.061 | `/root/bench/telys-vllm/telys_vllm/connector.py:638` |

### Top 3 contributors by total time
- **start_load_kv**: 96691.550 ms (38.4%)
- **inject_block**: 44112.025 ms (17.5%)
- **load_decoded_blocks**: 35866.125 ms (14.2%)

### Dominant phase
- phase: `start_load_kv`
- total: 96691.550 ms
- median: 14405.363 ms
- dominant caller: `/root/bench/telys-vllm/telys_vllm/connector.py:404`
