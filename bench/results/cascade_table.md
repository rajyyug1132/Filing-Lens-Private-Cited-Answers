| System | Easy acc. | Hard acc. | ECE ↓ (Easy / Hard) | Citation page match | p50 latency | Verifier calls |
|---|---|---|---|---|---|---|
| Feature gate only | 0.864 | 0.518 | 0.191 / 0.157 | 0.279 (top-ranked page) | 0.41 ms | 0% |
| Verifier only (Qwen2.5-3B, DSPy-compiled) | pending (Kaggle) | pending (Kaggle) | pending | pending | pending | 100% |
| Cascade: gate → verifier | pending (Kaggle) | pending (Kaggle) | pending | pending | pending | 53% |

Test split only: 165 instances (company-grouped; 55 per set). Gate latency: features + LR on Node v22.22.0, 4 vCPU container, no GPU; verifier latency: one verifier call on the Kaggle T4 build named in the row. Retrieval (shared by all rows) is not included.
