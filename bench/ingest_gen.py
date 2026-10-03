"""Merge Kaggle generation results into the bench.

Inputs:  bench/results/gen_inputs.jsonl   (cloud bench: questions, gold, retrieved pages, LR gate p)
         bench/results/gen_results.jsonl  (Kaggle: answers, cited pages, latency, logprob conf, router P(YES))
Outputs: bench/results/gen_metrics.json, bench/results/gen_table.md

Metrics
- citation accuracy: answered gold-answer cases whose cited pages are all in the context AND include a gold page
- answer accuracy (auto-graded): numeric gold answers must match a number in the model answer within 1% relative;
  non-numeric gold answers count as correct at token-F1 >= 0.4. This is an approximation of the human grading
  FinanceBench used; the per-row grades are written out for spot checks.
- end-to-end decision accuracy per gate: correct = (gold-answer AND answered AND answer correct) OR (gold-abstain AND abstained)
- tokens/sec, latency (median, p95)

If gen_results.jsonl is missing, every metric is written as "pending".
"""
import json
import re
import statistics
import sys
from pathlib import Path

R = Path("bench/results")
GATES = {"No gate": None, "Calibrated LR gate": "p_gate_oof", "LLM-router (3B YES/NO)": "p_yes", "Logprob confidence": "logprob_conf"}


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
    g = nums(gold)
    if g and re.search(r"[\d]", gold) and len(gold) < 40:  # short numeric gold answer
        a = nums(ans)
        return any(abs(x - g[0]) <= 0.01 * max(1.0, abs(g[0])) for x in a)
    return f1(gold, ans) >= 0.4


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def main():
    inputs = {json.loads(l)["id"]: json.loads(l) for l in (R / "gen_inputs.jsonl").read_text().splitlines() if l.strip()}
    res_path = R / "gen_results.jsonl"
    if not res_path.exists():
        out = {"status": "pending", "reason": "bench/results/gen_results.jsonl not found; run bench/kaggle_gen.ipynb on Kaggle"}
        (R / "gen_metrics.json").write_text(json.dumps(out, indent=2))
        (R / "gen_table.md").write_text("| Metric | Value |\n|---|---|\n| Citation accuracy | pending |\n| Answer accuracy | pending |\n| Tokens/sec | pending |\n")
        print("pending")
        return
    rows = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()]
    if any(r.get("model") == "stub" for r in rows):
        sys.exit("gen_results.jsonl comes from a DRY_RUN (stub model); refusing to report it as real numbers")
    ans = {r["id"]: r for r in rows if r["arm"] == "answer"}
    router = {r["id"]: r for r in rows if r["arm"] == "llm_router"}
    graded = []
    for i, a in ans.items():
        inp = inputs[i]
        ctx = {h["page"] for h in inp["hits"]}
        cited = set(a["cited_pages"])
        own = inp["kind"] == "own_doc"
        answered = (not a["refused"]) and bool(cited) and cited <= ctx
        cite_ok = own and answered and bool(cited & set(inp["gold_pages"]))
        correct = own and answered and grade(inp["gold_answer"], a["answer"])
        graded.append({"id": i, "kind": inp["kind"], "answered": answered, "cite_ok": cite_ok, "correct": correct,
                       "p_gate_oof": inp.get("p_gate_oof"), "p_yes": router.get(i, {}).get("p_yes"), "logprob_conf": a.get("logprob_conf"),
                       "tokens_per_s": a.get("tokens_per_s"), "latency_s": a["latency_s"], "question_type": inp["question_type"]})
    own = [g for g in graded if g["kind"] == "own_doc"]
    own_answered = [g for g in own if g["answered"]]
    metrics = {
        "status": "real",
        "model": rows[0]["model"],
        "instances": len(graded),
        "citation_accuracy_on_answered": sum(g["cite_ok"] for g in own_answered) / max(1, len(own_answered)),
        "answer_accuracy_gold_answer_cases": sum(g["correct"] for g in own) / max(1, len(own)),
        "refusal_rate_gold_abstain_cases": sum(not g["answered"] for g in graded if g["kind"] == "off_doc") / max(1, sum(g["kind"] == "off_doc" for g in graded)),
        "tokens_per_s_median": statistics.median([g["tokens_per_s"] for g in graded if g["tokens_per_s"]] or [0]),
        "latency_s_p50": pct([g["latency_s"] for g in graded], 0.5),
        "latency_s_p95": pct([g["latency_s"] for g in graded], 0.95),
        "end_to_end": {},
    }
    for name, key in GATES.items():
        ok = 0
        n = 0
        for g in graded:
            p = 1.0 if key is None else g.get(key)
            if p is None:
                continue
            n += 1
            gate_pass = p >= 0.5
            final_answer = gate_pass and g["answered"]
            ok += (g["kind"] == "own_doc" and final_answer and g["correct"]) or (g["kind"] == "off_doc" and not final_answer)
        metrics["end_to_end"][name] = {"accuracy": ok / n if n else None, "n": n}
    (R / "gen_metrics.json").write_text(json.dumps(metrics, indent=2))
    (R / "gen_graded.jsonl").write_text("\n".join(json.dumps(g) for g in graded) + "\n")
    md = "| Metric | Value |\n|---|---|\n"
    md += f"| Citation accuracy (answered, gold-answer cases) | {metrics['citation_accuracy_on_answered']:.3f} |\n"
    md += f"| Answer accuracy (auto-graded, gold-answer cases) | {metrics['answer_accuracy_gold_answer_cases']:.3f} |\n"
    md += f"| Refusal rate on gold-abstain cases (no gate) | {metrics['refusal_rate_gold_abstain_cases']:.3f} |\n"
    md += f"| Tokens/sec (median, T4) | {metrics['tokens_per_s_median']:.1f} |\n"
    for name, v in metrics["end_to_end"].items():
        md += f"| End-to-end accuracy: {name} | {v['accuracy']:.3f} (n={v['n']}) |\n" if v["accuracy"] is not None else f"| End-to-end accuracy: {name} | n/a |\n"
    md += "| End-to-end accuracy: Jev | [ASK: what is Jev + can it run locally] |\n"
    (R / "gen_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
