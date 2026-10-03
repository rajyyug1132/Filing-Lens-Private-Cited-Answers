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
    """gen_inputs.jsonl row -> dspy.Example.

    Gold verdict is ANSWER only when the question is on its own filing AND a gold page is among
    the snippets the model sees (the instruction says decide ONLY from the snippets); otherwise
    ABSTAIN. `reachable` keeps that distinction for reporting. Only ANSWER rows carry an `answer`
    (the gold answer tagged with the gold snippet page), so abstain rows never become empty-answer
    demos for the answerer."""
    gold_in_snips = [p for p in inst["gold_pages"] if p in inst.get("snippet_pages", [])]
    answer = inst["kind"] == "own_doc" and bool(gold_in_snips)
    fields = dict(
        id=inst["id"],
        kind=inst["kind"],
        reachable=answer,
        question=inst["question"],
        snippets=inst["snippets"],
        verdict="ANSWER" if answer else "ABSTAIN",
        evidence_page=f"p.{gold_in_snips[0]}" if answer else "NONE",
        gold_pages=[f"p.{p}" for p in inst["gold_pages"]],
    )
    if answer:
        fields["answer"] = f"{inst['gold_answer']} [p.{gold_in_snips[0]}]"
    return dspy.Example(**fields).with_inputs("question", "snippets")


def stratified(examples):
    """Interleave ANSWER and ABSTAIN examples (ANSWER first) so BootstrapFewShot sees both kinds
    early instead of stopping after two abstain traces."""
    pos = [e for e in examples if e.verdict == "ANSWER"]
    neg = [e for e in examples if e.verdict != "ANSWER"]
    out = []
    for i in range(max(len(pos), len(neg))):
        out += ([pos[i]] if i < len(pos) else []) + ([neg[i]] if i < len(neg) else [])
    return out


def balanced_score(program, devset):
    """Mean of per-class metric (ANSWER cases, ABSTAIN cases): an always-ABSTAIN program scores 0.5."""
    by = {"ANSWER": [], "ABSTAIN": []}
    for ex in devset:
        try:
            pred = program(question=ex.question, snippets=ex.snippets)
            by[ex.verdict].append(metric(ex, pred))
        except Exception:
            by[ex.verdict].append(0.0)
    return sum(sum(v) / len(v) for v in by.values() if v) / sum(1 for v in by.values() if v)
