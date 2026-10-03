"""Cascade bench: feature gate only | DSPy verifier only | cascade (gate -> verifier).

Inputs:  bench/results/gen_inputs.jsonl     cloud bench: 150 questions x {own_doc, off_doc, gold_removed}, company-grouped
                                            split, sentence snippets, out-of-fold gate probability p_gate_oof
         bench/results/results.json         cloud bench: gate latency
         bench/results/cascade_results.jsonl  Kaggle (bench/kaggle_gen.ipynb): verifier verdict, evidence page, P(ANSWER)
                                            from logprobs, cited answer, latency, per build (vllm_fp16, gguf_q4)
Outputs: bench/results/cascade_table.md, bench/results/cascade_metrics.json

All rows are scored on the TEST split only (companies never seen by the gate fold or the DSPy compile).
- Easy acc: decisions on own_doc (should ANSWER) + off_doc (should ABSTAIN)
- Hard acc: decisions on own_doc + gold_removed (should ABSTAIN). Never merged with Easy.
- ECE: 10-bin ECE of the system's P(answer), reported per set (Easy / Hard)
- Citation page match: answered gold-answer cases whose evidence page is a gold page. The gate proposes no page, so
  its row uses the top-ranked retrieved page (labelled).
- p50 latency: per decision. Gate: Node, this machine. Verifier rows: the Kaggle T4 build named in the row.
- Verifier calls %: share of test instances that reach the 3B verifier.
Cascade confidence = p_gate when the gate abstains, else the verifier's P(ANSWER).
"""
import json
import re
import statistics
from pathlib import Path

R = Path("bench/results")


def ece(p, y, bins=10):
    tot, n = 0.0, len(p)
    for b in range(bins):
        idx = [i for i, v in enumerate(p) if min(bins - 1, int(v * bins)) == b]
        if idx:
            tot += len(idx) / n * abs(sum(y[i] for i in idx) / len(idx) - sum(p[i] for i in idx) / len(idx))
    return tot


def nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", s or "") if re.search(r"\d", x)]


def f1(a, b):
    ta, tb = re.findall(r"[a-z0-9]+", (a or "").lower()), re.findall(r"[a-z0-9]+", (b or "").lower())
    common = sum(min(ta.count(t), tb.count(t)) for t in set(ta))
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def grade(gold, ans):
    """Auto-grade (approximation of FinanceBench's human grading): short numeric gold -> a number within 1%;
    otherwise token-F1 >= 0.4."""
    g = nums(gold)
    if g and len(gold) < 40:
        return any(abs(x - g[0]) <= 0.01 * max(1.0, abs(g[0])) for x in nums(ans))
    return f1(gold, ans) >= 0.4


def p50(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def system_scores(decisions, confs, pages, inputs, lat, calls):
    """decisions/confs/pages keyed by instance id over the test split."""
    out = {}
    for label, neg in (("easy", "off_doc"), ("hard", "gold_removed")):
        ids = [i for i, g in inputs.items() if g["kind"] in ("own_doc", neg)]
        y = [1 if inputs[i]["kind"] == "own_doc" else 0 for i in ids]
        d = [decisions[i] for i in ids]
        out[f"{label}_acc"] = sum(int(a == b) for a, b in zip(d, y)) / len(ids)
        out[f"{label}_ece"] = ece([confs[i] for i in ids], y)
        out[f"{label}_n"] = len(ids)
    answered_own = [i for i, g in inputs.items() if g["kind"] == "own_doc" and decisions[i] == 1]
    out["citation_page_match"] = (sum(int(pages.get(i) in inputs[i]["gold_pages"]) for i in answered_own) / len(answered_own)) if answered_own else None
    out["answered_gold_answer_cases"] = len(answered_own)
    out["latency_p50_ms"] = p50(lat)
    out["verifier_calls_pct"] = calls
    return out


def main():
    inputs = {g["id"]: g for g in (json.loads(l) for l in (R / "gen_inputs.jsonl").read_text().splitlines() if l.strip()) if g["split"] == "test"}
    bench = json.loads((R / "results.json").read_text())
    L = bench["latencyMs"]
    gate_ms = L["featureExtraction"]["p50"] + L["classifierPredict"]["p50"]
    gate_pass = {i: g["p_gate_oof"] >= 0.5 for i, g in inputs.items()}
    metrics = {"split": "test", "instances": len(inputs), "rows": {}}

    metrics["rows"]["Feature gate only"] = system_scores(
        {i: int(gate_pass[i]) for i in inputs},
        {i: g["p_gate_oof"] for i, g in inputs.items()},
        {i: g["top1_page"] for i, g in inputs.items()},
        inputs, [gate_ms] * len(inputs), 0.0,
    )
    metrics["rows"]["Feature gate only"]["page_source"] = "top-ranked retrieved page"
    metrics["rows"]["Feature gate only"]["latency_where"] = bench["machine"]

    res_path = R / "cascade_results.jsonl"
    rows = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()] if res_path.exists() else []
    if any(r.get("model") == "stub" for r in rows):
        raise SystemExit("cascade_results.jsonl comes from a DRY_RUN (stub model); refusing to report it")
    builds = sorted({r["build"] for r in rows})
    metrics["status"] = "real" if builds else "pending"
    cascade_calls = 100.0 * sum(gate_pass.values()) / len(inputs)
    metrics["cascade_verifier_calls_pct"] = cascade_calls
    for b in builds:
        V = {r["id"]: r for r in rows if r["build"] == b and r["id"] in inputs}
        if len(V) != len(inputs):
            print(f"warning: build {b} covers {len(V)}/{len(inputs)} test instances")
        ids = [i for i in inputs if i in V]
        sub = {i: inputs[i] for i in ids}
        conf_v = {i: (V[i]["p_answer"] if V[i]["p_answer"] is not None else float(V[i]["verdict"] == "ANSWER")) for i in ids}
        dec_v = {i: int(V[i]["verdict"] == "ANSWER") for i in ids}
        page_v = {i: int(re.sub(r"\D", "", V[i]["evidence_page"]) or -1) for i in ids}
        metrics["rows"][f"Verifier only ({b})"] = system_scores(dec_v, conf_v, page_v, sub, [V[i]["latency_check_s"] * 1000 for i in ids], 100.0)
        dec_c = {i: int(gate_pass[i] and dec_v[i]) for i in ids}
        conf_c = {i: (conf_v[i] if gate_pass[i] else sub[i]["p_gate_oof"]) for i in ids}
        lat_c = [gate_ms + (V[i]["latency_check_s"] * 1000 if gate_pass[i] else 0.0) for i in ids]
        metrics["rows"][f"Cascade: gate → verifier ({b})"] = system_scores(dec_c, conf_c, page_v, sub, lat_c, 100.0 * sum(gate_pass[i] for i in ids) / len(ids))
        own = [i for i in ids if sub[i]["kind"] == "own_doc"]
        answered = [i for i in own if dec_c[i]]
        metrics.setdefault("generation", {})[b] = {
            "answer_accuracy_auto_graded_gold_answer_cases": sum(int(dec_c[i] and grade(sub[i]["gold_answer"], V[i]["answer"])) for i in own) / len(own),
            "answer_cites_only_snippet_pages": (sum(int(set(V[i]["cited_pages"]) <= set(sub[i]["snippet_pages"]) and bool(V[i]["cited_pages"])) for i in answered) / len(answered)) if answered else None,
            "tokens_per_s_median": p50([V[i]["tokens_per_s"] for i in ids]),
            "prompt_tokens_check_max": max([V[i]["prompt_tokens_check"] or 0 for i in ids] or [0]),
            "optimizer": rows[0].get("optimizer"),
            "dev_metric": rows[0].get("dev_metric"),
        }
    (R / "cascade_metrics.json").write_text(json.dumps(metrics, indent=2))

    f = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    md = "| System | Easy acc. | Hard acc. | ECE ↓ (Easy / Hard) | Citation page match | p50 latency | Verifier calls |\n|---|---|---|---|---|---|---|\n"
    for name, m in metrics["rows"].items():
        cite = f(m["citation_page_match"]) + (" (top-ranked page)" if m.get("page_source") else "")
        lat = f"{m['latency_p50_ms']:.2f} ms" if name.startswith("Feature") else f"{m['latency_p50_ms']:.0f} ms"
        md += f"| {name} | {f(m['easy_acc'])} | {f(m['hard_acc'])} | {f(m['easy_ece'])} / {f(m['hard_ece'])} | {cite} | {lat} | {m['verifier_calls_pct']:.0f}% |\n"
    if not builds:
        md += "| Verifier only (Qwen2.5-3B, DSPy-compiled) | pending (Kaggle) | pending (Kaggle) | pending | pending | pending | 100% |\n"
        md += f"| Cascade: gate → verifier | pending (Kaggle) | pending (Kaggle) | pending | pending | pending | {cascade_calls:.0f}% |\n"
    md += (f"\nTest split only: {len(inputs)} instances (company-grouped; {sum(g['kind'] == 'own_doc' for g in inputs.values())} per set). "
           f"Gate latency: features + LR on {bench['machine']}; verifier latency: one verifier call on the Kaggle T4 build named in the row. "
           f"Retrieval (shared by all rows) is not included.\n")
    (R / "cascade_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
