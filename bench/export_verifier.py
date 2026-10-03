"""Export a (compiled or uncompiled) Lens program to app/prompts/verifier.json.

The JSON holds, per predictor, the instructions, fields and demos the browser needs to
rebuild the prompt (lib/dspy-chat.ts), plus reference prompts rendered by DSPy's own
ChatAdapter. tests/unit/dspy-chat.test.ts asserts the browser builds identical messages.

  python bench/export_verifier.py                                   # uncompiled baseline
  python bench/export_verifier.py --program verifier.json --model "Qwen2.5-3B fp16 (vLLM)"
  python bench/export_verifier.py --edge-cases tests/fixtures/dspy-edge-cases.json
"""
import argparse
import json
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent))
from lens_program import Lens  # noqa: E402


def field_desc(name, info):
    d = (info.json_schema_extra or {}).get("desc", "")
    return "" if d == f"${{{name}}}" else d


def spec_of(pred):
    sig = pred.signature
    keys = list(sig.fields)
    return {
        "instructions": sig.instructions,
        "inputs": [{"name": k, "desc": field_desc(k, v)} for k, v in sig.input_fields.items()],
        "outputs": [{"name": k, "desc": field_desc(k, v)} for k, v in sig.output_fields.items()],
        "demos": [{k: d.get(k) for k in keys if k in d} for d in pred.demos],
    }


def references(pred, ref_inputs):
    adapter = dspy.ChatAdapter()
    return [{"inputs": inp, "messages": adapter.format(pred.signature, pred.demos, inp)} for inp in ref_inputs]


def export(lens, out_path, ref_inputs, meta):
    data = {
        "dspy_version": dspy.__version__,
        **meta,
        "predictors": {
            name: {**spec_of(pred), "references": references(pred, ref_inputs)}
            for name, pred in (("check", lens.check), ("answer", lens.answer))
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return data


def edge_cases(out_path):
    """Synthetic cases DSPy formats specially: multi-line indented instructions, an
    incomplete demo, a field with no description."""

    class Edge(dspy.Signature):
        """First line of the task.
            Indented second line.

        Third line after a blank."""
        question: str = dspy.InputField()
        snippets: str = dspy.InputField(desc="tagged snippets")
        verdict: str = dspy.OutputField(desc="ANSWER or ABSTAIN")
        evidence_page: str = dspy.OutputField()

    p = dspy.Predict(Edge)
    p.demos = [
        dspy.Example(question="Q incomplete?", snippets="[p.1] a", verdict="ABSTAIN"),
        dspy.Example(question="Q full?", snippets="[p.2] b\n[p.3] c", verdict="ANSWER", evidence_page="p.3"),
    ]
    inp = {"question": "What was it?", "snippets": "[p.9] value 1,234 [x]"}
    cases = [{**spec_of(p), "references": references(p, [inp])}]
    Path(out_path).write_text(json.dumps({"dspy_version": dspy.__version__, "cases": cases}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", help="path saved by lens.save(...); omit for the uncompiled baseline")
    ap.add_argument("--out", default="app/prompts/verifier.json")
    ap.add_argument("--inputs", default="bench/results/gen_inputs.jsonl")
    ap.add_argument("--model", default="none (uncompiled)")
    ap.add_argument("--optimizer", default="none")
    ap.add_argument("--edge-cases")
    a = ap.parse_args()
    if a.edge_cases:
        edge_cases(a.edge_cases)
    lens = Lens()
    if a.program:
        lens.load(a.program)
    rows = [json.loads(l) for l in Path(a.inputs).read_text().splitlines() if l.strip()]
    test = [r for r in rows if r.get("split") == "test"] or rows  # first run: inputs not split yet
    for r in test:
        r.setdefault("snippets", "\n".join(f"[p.{h['page']}] {h['text']}" for h in r["hits"][:2]))
    refs = [{"question": r["question"], "snippets": r["snippets"]} for r in (test[:1] + [r for r in test if r["kind"] == "own_doc"][:1])]
    export(lens, a.out, refs, {"compiled": bool(a.program), "model": a.model, "optimizer": a.optimizer})
    print("wrote", a.out, "| compiled:", bool(a.program), "| demos:", {n: len(p.demos) for n, p in (("check", lens.check), ("answer", lens.answer))})
