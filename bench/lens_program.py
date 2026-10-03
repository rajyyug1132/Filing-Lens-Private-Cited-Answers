"""The DSPy program behind Filing Lens' verifier + cited answerer.

Shared by bench/kaggle_gen.ipynb (compile/eval on Kaggle) and bench/export_verifier.py
(writes app/prompts/verifier.json, which the browser formats with lib/dspy-chat.ts).
"""
import re

import dspy


class Answerable(dspy.Signature):
    """Decide ONLY from the snippets whether they contain the answer. Never use outside knowledge."""
    question: str = dspy.InputField()
    snippets: str = dspy.InputField(desc="top-k sentence snippets, each tagged [p.N]")
    verdict: str = dspy.OutputField(desc="ANSWER or ABSTAIN")
    evidence_page: str = dspy.OutputField(desc="p.N that holds the answer, or NONE")


class CitedAnswer(dspy.Signature):
    """Answer in 1-2 sentences using only the snippets. Cite every number as [p.N]."""
    question: str = dspy.InputField()
    snippets: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Lens(dspy.Module):
    def __init__(self):
        self.check = dspy.Predict(Answerable)
        self.answer = dspy.Predict(CitedAnswer)

    def forward(self, question, snippets):
        v = self.check(question=question, snippets=snippets)
        if (v.verdict or "").strip().upper() != "ANSWER":
            return dspy.Prediction(verdict="ABSTAIN", answer="", evidence_page="NONE")
        a = self.answer(question=question, snippets=snippets)
        return dspy.Prediction(verdict="ANSWER", answer=a.answer, evidence_page=v.evidence_page)


def norm_page(p):
    """'p.42', '42', '[p.42]', 'page 42' -> 'p.42'; anything else unchanged (e.g. 'NONE')."""
    m = re.search(r"\d+", p or "")
    return f"p.{int(m.group())}" if m else (p or "").strip()


def metric(gold, pred, trace=None):
    if gold.verdict == "ABSTAIN":
        return float(pred.verdict == "ABSTAIN")
    return float(pred.verdict == "ANSWER" and norm_page(pred.evidence_page) in gold.gold_pages)


def to_example(inst):
    """gen_inputs.jsonl row -> dspy.Example. Answer cases carry their gold page(s) and a
    gold answer tagged with the first gold page (FinanceBench answers have no [p.N] tags)."""
    pages = [f"p.{p}" for p in inst["gold_pages"]]
    ans = inst["kind"] == "own_doc"
    return dspy.Example(
        id=inst["id"],
        kind=inst["kind"],
        question=inst["question"],
        snippets=inst["snippets"],
        verdict="ANSWER" if ans else "ABSTAIN",
        evidence_page=pages[0] if ans and pages else "NONE",
        gold_pages=pages,
        answer=f"{inst['gold_answer']} [{pages[0]}]" if ans and pages else "",
    ).with_inputs("question", "snippets")
