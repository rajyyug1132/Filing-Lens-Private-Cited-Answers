| System | Easy acc. | Hard acc. | ECE ↓ (Easy / Hard) | Citation page match | p50 latency | Verifier calls |
|---|---|---|---|---|---|---|
| Feature gate only | 0.864 | 0.524 | 0.192 / 0.156 | 0.279 (top-ranked page) | 0.42 ms | 0% |
| Verifier only (fp16_llamacpp) | 0.509 | 0.495 | 0.476 / 0.496 | 0.500 | 7447 ms | 100% |
| Cascade: gate → verifier (fp16_llamacpp) | 0.518 | 0.495 | 0.392 / 0.404 | 0.500 | 7202 ms | 52% |
| Verifier only (gguf_q4) | 0.527 | 0.495 | 0.468 / 0.474 | 0.500 | 7156 ms | 100% |
| Cascade: gate → verifier (gguf_q4) | 0.536 | 0.495 | 0.410 / 0.391 | 0.500 | 6939 ms | 52% |

Test split only: 160 instances (company-grouped): 55 answer, 55 Easy, 50 Hard. Only 27 of the 55 answer cases have a gold page in the snippets, which caps citation page match and answer accuracy. Gate latency: features + LR on Node v22.22.0, 4 vCPU container, no GPU; verifier latency: one verifier call on the Kaggle T4 build named in the row. Retrieval (shared by all rows) is not included.
