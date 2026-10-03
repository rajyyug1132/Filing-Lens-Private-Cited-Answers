| Gate | Easy acc. | Easy ECE ↓ | Easy abstain P / R | Hard acc. | Hard ECE ↓ | Hard abstain P / R | Decision latency p50 |
|---|---|---|---|---|---|---|---|
| No gate (always answer) | 0.500 | 0.500 | — / 0.000 | 0.500 | 0.500 | — / 0.000 | 0 ms |
| Retrieval score threshold (uncalibrated) | 0.623 | 0.204 | 0.570 / 1.000 | 0.523 | 0.085 | 0.515 / 0.800 | <0.01 ms |
| LR gate, no temperature | 0.867 | 0.185 | 0.813 / 0.953 | 0.517 | 0.144 | 0.535 / 0.253 | 0.41 ms |
| Calibrated LR gate (LR + temperature), shipped (v2 features) | 0.867 | 0.169 | 0.813 / 0.953 | 0.517 | 0.159 | 0.535 / 0.253 | 0.41 ms |

Hard accuracy with v1 features was 0.513 (< 0.75), so the gate was retrained once with two "does the top passage answer this kind of question" features. Before → after: Easy acc 0.870 → 0.867, Easy ECE 0.166 → 0.169; Hard acc 0.513 → 0.517, Hard ECE 0.162 → 0.159.

Recall of the gold page (150 answer cases): @1 0.233 · @2 0.333 · @4 0.440 · @8 0.633 · @12 0.700 · @16 0.733

Prompt tokens before the cap (system + question + k chunks, Qwen tokenizer proxy): k=4: p50 1193, p95 1689, max 2100 · k=8: p50 2253, p95 3075, max 3586 · k=12: p50 3261, p95 4362, max 4964

With the 3000-token cap (drop lowest-ranked chunks; 0.411 tokens/char estimate): k=4: recall 0.433, mean 4.0 chunks, max 1843 tokens · k=8: recall 0.567, mean 6.4 chunks, max 2750 tokens · k=12: recall 0.567, mean 6.4 chunks, max 2750 tokens

**Shipped k = 8** (no k reached recall 0.6 under the 3000-token cap; k with the highest capped recall).

Sentence snippets (what the verifier reads, ≤2,200 chars): gold page present in 52.7% of answer cases; verifier prompt with 2 worst-case demos: p95 2459, max 2674 tokens.

Each set: 150 answer + 150 abstain instances; one gate trained on all 450 with 5-fold CV grouped by company; Easy and Hard scored separately. Retrieval (query embed + hybrid search) p50 9.46 ms, p95 23.06 ms on Node v22.22.0, 4 vCPU container, no GPU.
