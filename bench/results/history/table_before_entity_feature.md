| Gate | Accuracy | ECE ↓ | AUROC | Abstain precision | Abstain recall | Answer rate | Decision latency p50 | Marginal cost / query |
|---|---|---|---|---|---|---|---|---|
| No gate (always answer) | 0.500 | 0.500 | — | — | 0.000 | 1.000 | 0 ms | $0 (on-device) |
| Retrieval score threshold (uncalibrated) | 0.780 | 0.204 | 0.873 | 0.804 | 0.740 | 0.540 | <0.01 ms | $0 (on-device) |
| LR gate, no temperature | 0.820 | 0.042 | 0.890 | 0.829 | 0.807 | 0.513 | 0.26 ms | $0 (on-device) |
| Calibrated LR gate (LR + temperature, shipped) | 0.820 | 0.046 | 0.885 | 0.829 | 0.807 | 0.513 | 0.26 ms | $0 (on-device) |
| LLM-router (3B self-check) | pending (Kaggle) | | | | | | | |
| Jev | [ASK: what is Jev + can it run locally] | | | | | | | |

Retrieval recall of the gold page on the 150 answer cases: @1 0.247 · @2 0.340 · @4 0.453 · @8 0.613 · @16 0.740

n=300 instances (150 gold-answer, 150 gold-abstain), 5-fold CV grouped by company, top-k=4. Shared retrieval cost per question (query embed + hybrid search): p50 9.50 ms, p95 21.77 ms on Node v22.22.0, 4 vCPU container, no GPU.
