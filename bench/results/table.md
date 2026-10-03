| Gate | Easy acc. | Easy ECE ↓ | Easy abstain P / R | Hard acc. | Hard ECE ↓ | Hard abstain P / R | Decision latency p50 |
|---|---|---|---|---|---|---|---|
| No gate (always answer) | 0.500 | 0.500 | — / 0.000 | 0.517 | 0.483 | — / 0.000 | 0 ms |
| Retrieval score threshold (uncalibrated) | 0.620 | 0.248 | 0.569 / 0.993 | 0.528 | 0.076 | 0.507 / 0.829 | <0.01 ms |
| LR gate, no temperature | 0.870 | 0.161 | 0.828 / 0.933 | 0.524 | 0.139 | 0.517 / 0.221 | 0.42 ms |
| Calibrated LR gate (LR + temperature), shipped (v2 features) | 0.870 | 0.149 | 0.828 / 0.933 | 0.524 | 0.151 | 0.517 / 0.221 | 0.42 ms |

Hard accuracy with v1 features was 0.528 (< 0.75), so the gate was retrained once with two "does the top passage answer this kind of question" features. Before → after: Easy acc 0.853 → 0.870, Easy ECE 0.152 → 0.149; Hard acc 0.528 → 0.524, Hard ECE 0.147 → 0.151.

Recall of the gold page (150 answer cases): @1 0.233 · @2 0.333 · @4 0.440 · @8 0.633 · @12 0.700 · @16 0.733

Prompt tokens before the cap (system + question + k chunks, Qwen tokenizer proxy): k=4: p50 1193, p95 1689, max 2100 · k=8: p50 2253, p95 3075, max 3586 · k=12: p50 3261, p95 4362, max 4964

With the 3000-token cap (drop lowest-ranked chunks; 0.411 tokens/char estimate): k=4: recall 0.433, mean 4.0 chunks, max 1843 tokens · k=8: recall 0.567, mean 6.4 chunks, max 2750 tokens · k=12: recall 0.567, mean 6.4 chunks, max 2750 tokens

**Shipped k = 8** (no k reached recall 0.6 under the 3000-token cap; k with the highest capped recall).

Sentence snippets (what the verifier reads, ≤2,200 chars): gold page present in 52.7% of answer cases; verifier prompt with 2 worst-case demos: p95 2295, max 2405 tokens.

Hard set: annotated gold pages plus 252 other pages that still contained every gold number were removed; 10 questions were excluded because the retrieved context still contained the answer (4 by numbers, 6 by answer terms), leaving 140. Easy abstains use another company's 10-K from the same CV fold, so no filing crosses folds.

Sets: 150 answer, 150 Easy, 140 Hard; one gate trained on all of them with 5-fold CV grouped by company; Easy and Hard scored separately. Retrieval (query embed + hybrid search) p50 9.47 ms, p95 22.41 ms on Node v22.22.0, 4 vCPU container, no GPU.
