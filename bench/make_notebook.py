"""Generates bench/kaggle_gen.ipynb. Edit the cells here, not the .ipynb."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(keepends=True)}

cells = [
    md("""
# Filing Lens: DSPy verifier + cascade bench (Kaggle, GPU T4, Internet on)

Inputs come from the cloud bench (`npm run bench`) on `main`: `bench/results/gen_inputs.jsonl`, which holds 150 FinanceBench
questions × 3 sets (`own_doc` = gold answer, `off_doc` = Easy abstain from the same CV fold, `gold_removed` = Hard abstain with answer-leaking questions excluded), with the exact
sentence snippets the browser builds, a company-grouped `split` (train / dev / test) and the feature gate's out-of-fold `p_gate_oof`.

1. Serve **Qwen2.5-3B-Instruct fp16 with vLLM** (`dtype=\"half\"`; if vLLM won't start, llama.cpp on the fp16 GGUF) and compile the DSPy `Lens` program (verifier `Answerable` → `CitedAnswer`)
   with `BootstrapFewShot` (2 demos) on the train split. Optionally try **MIPROv2 auto="light"** and keep it only if dev improves.
2. Evaluate the compiled program on the test split with the **fp16 build** and with a **GGUF Q4_K_M build (llama.cpp)**, the
   quantisation closest to the phone. Each call uses exactly the messages DSPy's ChatAdapter renders, which are also what the browser sends.
3. Write `cascade_results.jsonl`, `token_counts.json`, `verifier.json` (DSPy save) and `app_verifier.json` (→ commit as `app/prompts/verifier.json`).

`DRY_RUN=1` runs every cell on CPU with a stub LM and no downloads.
"""),
    code("""
import json, math, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
RUN_MIPRO = os.environ.get("RUN_MIPRO", "1") == "1" and not DRY_RUN
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 4 * 3600))  # skip optional work (MIPROv2) past half of this
LIMIT = int(os.environ.get("LIMIT", "0")) or None                  # cap test instances for a quick run
REPO_RAW = os.environ.get("REPO_RAW", "https://raw.githubusercontent.com/rajyyug1132/Filing-Lens-Private-Cited-Answers/main")
HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GGUF_REPO, GGUF_FILE = "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "bench/results/kaggle"))
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
print({"DRY_RUN": DRY_RUN, "RUN_MIPRO": RUN_MIPRO, "OUT": str(OUT)})
"""),
    code("""
if not DRY_RUN:
    pip = lambda *a, **kw: subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], **kw)
    pip("dspy", "vllm", "huggingface_hub", "openai", check=True)
    ok = any(pip("--only-binary=:all:", "llama-cpp-python[server]", "--extra-index-url",
                 f"https://abetlen.github.io/llama-cpp-python/whl/{cu}").returncode == 0 for cu in ("cu124", "cu121"))
    if not ok:  # no prebuilt CUDA wheel: build it (~15-20 min)
        pip("--no-cache-dir", "llama-cpp-python[server]", check=True, env=dict(os.environ, CMAKE_ARGS="-DGGML_CUDA=on", FORCE_CMAKE="1"))
"""),
    code("""
def fetch(rel):
    \"\"\"Repo file: local checkout first (dry run / CI), else the main branch on GitHub.\"\"\"
    p = Path(rel)
    return p.read_text() if p.exists() else urllib.request.urlopen(f"{REPO_RAW}/{rel}", timeout=60).read().decode()

if not Path("bench/lens_program.py").exists():
    for mod in ("bench/lens_program.py", "bench/export_verifier.py"):
        Path(Path(mod).name).write_text(fetch(mod))
sys.path.insert(0, "bench" if Path("bench/lens_program.py").exists() else ".")
import dspy
from lens_program import Lens, balanced_score, metric, norm_page, stratified, to_example
import export_verifier

rows = [json.loads(l) for l in fetch("bench/results/gen_inputs.jsonl").splitlines() if l.strip()]
by_split = {s: [to_example(r) for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
by_split["train"] = stratified(by_split["train"])  # ANSWER/ABSTAIN interleaved so bootstrapping sees both
if DRY_RUN:
    by_split = {s: stratified(v)[:6] for s, v in by_split.items()}
elif LIMIT:
    by_split["test"] = by_split["test"][:LIMIT]
print({s: len(v) for s, v in by_split.items()}, "| dspy", dspy.__version__)
"""),
    code("""
def start_server(cmd, port, name):
    log = open(OUT / f"{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited; see {OUT}/{name}.log")
            time.sleep(10)
    raise RuntimeError(f"{name} did not come up")

def start_fp16():
    # fp16 arm: vLLM with dtype="half" (the T4 has no bf16). If vLLM fails to start, fall back to the
    # llama.cpp server on an fp16 GGUF. DSPy and the eval only need an OpenAI-compatible endpoint on :8000.
    try:
        return start_server([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", HF_MODEL, "--served-model-name", "qwen2.5-3b",
                             "--dtype", "half", "--max-model-len", "4096", "--gpu-memory-utilization", "0.85", "--port", "8000"], 8000, "vllm"), "vllm"
    except Exception as e:
        print("vLLM failed to start, falling back to llama.cpp fp16:", e)
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
        from huggingface_hub import snapshot_download
        d = snapshot_download(GGUF_REPO, allow_patterns=["*fp16*.gguf"])
        first = sorted(Path(d).glob("*fp16*.gguf"))[0]  # llama.cpp loads the remaining split shards itself
        return start_server([sys.executable, "-m", "llama_cpp.server", "--model", str(first), "--model_alias", "qwen2.5-3b",
                             "--n_gpu_layers", "-1", "--n_ctx", "4096", "--port", "8000"], 8000, "llamacpp_fp16"), "llamacpp"

if DRY_RUN:
    from dspy.utils.dummies import DummyLM
    lm = DummyLM({"## evidence_page ##": {"verdict": "ANSWER", "evidence_page": "p.1"}, "## answer ##": {"answer": "Stub [p.1]."}})
    fp16_server, FP16_BACKEND = None, "stub"
else:
    fp16_server, FP16_BACKEND = start_fp16()
    lm = dspy.LM("openai/qwen2.5-3b", api_base="http://localhost:8000/v1", api_key="x", temperature=0.0, max_tokens=256)
print("fp16 backend:", FP16_BACKEND)
dspy.configure(lm=lm)
"""),
    code("""
# Compile: BootstrapFewShot, 2 demos, train split (company-grouped, no leakage into dev/test)
opt = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2, max_labeled_demos=2)
lens = opt.compile(Lens(), trainset=by_split["train"])

def clean_answer_demos(prog):
    # the answerer must only ever see demos that actually contain an answer
    prog.answer.demos = [d for d in prog.answer.demos if (d.get("answer") or "").strip()]
    return prog

lens = clean_answer_demos(lens)
# Balanced dev score = mean of per-class metric, so an always-ABSTAIN program scores 0.5 (raw dev is 2/3 abstain)
score = lambda prog: balanced_score(prog, by_split["dev"])
dev_bfs = score(lens)
optimizer, dev_score = "BootstrapFewShot(2 demos)", dev_bfs
n_ans = sum(e.verdict == "ANSWER" for e in by_split["dev"])
print(f"dev balanced metric, BootstrapFewShot: {dev_bfs:.3f} (ANSWER {n_ans}, ABSTAIN {len(by_split['dev']) - n_ans}); demos check/answer:", len(lens.check.demos), len(lens.answer.demos))
"""),
    code("""
# Optional: MIPROv2 light, kept only if dev improves
if RUN_MIPRO and time.time() - T0 < TIME_BUDGET_S * 0.5:
    mipro = dspy.MIPROv2(metric=metric, auto="light", max_bootstrapped_demos=2, max_labeled_demos=2, num_threads=8)
    cand = clean_answer_demos(mipro.compile(lens.deepcopy(), trainset=by_split["train"], valset=by_split["dev"]))
    dev_mipro = score(cand)
    print(f"dev balanced metric, MIPROv2 light: {dev_mipro:.3f}")
    if dev_mipro > dev_bfs:
        lens, optimizer, dev_score = cand, "BootstrapFewShot(2) -> MIPROv2 light", dev_mipro
else:
    print("MIPROv2 skipped (dry run, disabled, or out of time)")
lens.save(str(OUT / "verifier.json"))
print("kept:", optimizer, "| dev", dev_score, "| demos", len(lens.check.demos), len(lens.answer.demos))
"""),
    code("""
# Evaluation calls mirror the browser: ChatAdapter-rendered messages, one raw chat call per predictor.
from dspy.adapters import ChatAdapter
adapter = ChatAdapter()

def p_answer_from_logprobs(content):
    \"\"\"P(ANSWER) at the first non-blank token after the verdict header: mass on tokens starting AN.. vs AB..\"\"\"
    seen, text = False, ""
    for tok in content or []:
        text += tok["token"]
        if not seen:
            seen = "## verdict ## ]]" in text
            continue
        if not tok["token"].strip():
            continue
        pa = sum(math.exp(t["logprob"]) for t in tok["top_logprobs"] if t["token"].strip().upper().startswith("AN"))
        pb = sum(math.exp(t["logprob"]) for t in tok["top_logprobs"] if t["token"].strip().upper().startswith("AB"))
        return pa / (pa + pb) if pa + pb > 0 else None
    return None

def call(client, model, messages, max_tokens):
    if client is None:  # dry-run stub with the same response shape
        is_check = "evidence_page" in messages[0]["content"]
        txt = ("[[ ## verdict ## ]]\\nANSWER\\n\\n[[ ## evidence_page ## ]]\\np.1\\n\\n[[ ## completed ## ]]" if is_check
               else "[[ ## answer ## ]]\\nStub [p.1].\\n\\n[[ ## completed ## ]]")
        lp = [{"token": "[[ ## verdict ## ]]\\n", "logprob": 0.0, "top_logprobs": []},
              {"token": "ANSWER", "logprob": -0.1, "top_logprobs": [{"token": "ANSWER", "logprob": -0.1}, {"token": "AB", "logprob": -2.4}]}]
        return txt, lp, {"prompt_tokens": None, "completion_tokens": 12}
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens, logprobs=True, top_logprobs=5)
    c = r.choices[0]
    lp = [{"token": t.token, "logprob": t.logprob, "top_logprobs": [{"token": x.token, "logprob": x.logprob} for x in t.top_logprobs]}
          for t in (c.logprobs.content if c.logprobs and c.logprobs.content else [])]
    return c.message.content or "", lp, {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}

def parse(sig, text):
    try:
        return adapter.parse(sig, text)
    except Exception:
        return {}

CITE_RE = re.compile(r"\\[\\s*(?:p(?:age|g)?\\.?\\s*)(\\d+)")

def run_build(build, client, model):
    out = []
    for ex in by_split["test"]:
        inp = {"question": ex.question, "snippets": ex.snippets}
        t = time.perf_counter()
        txt, lp, use = call(client, model, adapter.format(lens.check.signature, lens.check.demos, inp), 40)
        t_check = time.perf_counter() - t
        v = parse(lens.check.signature, txt)
        verdict = "ANSWER" if (v.get("verdict") or "").strip().upper() == "ANSWER" else "ABSTAIN"
        row = {"id": ex.id, "kind": ex.kind, "build": build, "verdict": verdict, "evidence_page": norm_page(v.get("evidence_page")),
               "p_answer": p_answer_from_logprobs(lp), "latency_check_s": t_check, "prompt_tokens_check": use["prompt_tokens"],
               "answer": "", "cited_pages": [], "latency_answer_s": None, "tokens_per_s": None, "parse_ok": bool(v)}
        if verdict == "ANSWER":
            t = time.perf_counter()
            txt, _, use = call(client, model, adapter.format(lens.answer.signature, lens.answer.demos, inp), 200)
            dt = time.perf_counter() - t
            a = parse(lens.answer.signature, txt).get("answer") or ""
            row.update(answer=a, cited_pages=[int(n) for n in CITE_RE.findall(a)], latency_answer_s=dt,
                       tokens_per_s=(use["completion_tokens"] or 0) / dt if dt > 0 else None)
        out.append(row)
    return out

results = []
if DRY_RUN:
    results += run_build("dry_run_stub", None, None)
else:
    from openai import OpenAI
    results += run_build(f"fp16_{FP16_BACKEND}", OpenAI(base_url="http://localhost:8000/v1", api_key="x"), "qwen2.5-3b")
print(len(results), "rows so far")
"""),
    code("""
# Exact Qwen2.5 token counts (the model's own tokenizer + chat template) for the worst-case prompts:
# the compiled verifier and answerer with their demos, over every test instance.
token_counts = {"tokenizer": "skipped in dry run (no download)"}
if not DRY_RUN:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    n = lambda msgs: len(tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"])
    token_counts = {"tokenizer": HF_MODEL + " (apply_chat_template)"}
    for name, pred in (("verifier", lens.check), ("answer", lens.answer)):
        counts = sorted((n(adapter.format(pred.signature, pred.demos, {"question": e.question, "snippets": e.snippets})), e.id) for e in by_split["test"])
        token_counts[name] = {"max": counts[-1][0], "worst_id": counts[-1][1], "p95": counts[int(0.95 * (len(counts) - 1))][0],
                              "median": counts[len(counts) // 2][0], "demos": len(pred.demos), "budget": 3000, "fits": counts[-1][0] <= 3000}
(OUT / "token_counts.json").write_text(json.dumps(token_counts, indent=2))
print(json.dumps(token_counts, indent=2))
"""),
    code("""
# Same compiled program on the quantised GGUF Q4_K_M build (llama.cpp), closest to what the phone runs
if not DRY_RUN:
    fp16_server.terminate(); fp16_server.wait(timeout=120)
    from huggingface_hub import hf_hub_download
    from openai import OpenAI
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE)
    llama = start_server([sys.executable, "-m", "llama_cpp.server", "--model", gguf, "--model_alias", "qwen2.5-3b-q4",
                          "--n_gpu_layers", "-1", "--n_ctx", "4096", "--port", "8001"], 8001, "llamacpp")
    results += run_build("gguf_q4", OpenAI(base_url="http://localhost:8001/v1", api_key="x"), "qwen2.5-3b-q4")
    llama.terminate()

with (OUT / "cascade_results.jsonl").open("w") as f:
    for r in results:
        f.write(json.dumps({**r, "model": "stub" if DRY_RUN else HF_MODEL, "optimizer": optimizer, "dev_metric": dev_score}) + "\\n")

refs = [{"question": e.question, "snippets": e.snippets} for e in by_split["test"][:2]]
export_verifier.export(lens, OUT / "app_verifier.json", refs,
                       {"compiled": not DRY_RUN, "model": "stub (dry run)" if DRY_RUN else f"{HF_MODEL} fp16 via {FP16_BACKEND}",
                        "optimizer": optimizer, "dev_metric": dev_score})
print("wrote", OUT / "cascade_results.jsonl", OUT / "verifier.json", OUT / "app_verifier.json", f"| {time.time() - T0:.0f}s")
"""),
    md("""
**Next:** from the notebook's Output tab download `cascade_results.jsonl`, `token_counts.json`, `app_verifier.json` and `verifier.json`, then
`cp app_verifier.json app/prompts/verifier.json`, `cp cascade_results.jsonl token_counts.json verifier.json bench/results/`, run
`python bench/ingest_gen.py` and `npm test` (the prompt-equality test checks the browser rebuilds the compiled prompts exactly).
Or push them and tell Claude "ingest".
"""),
]
for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
Path("bench/kaggle_gen.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote bench/kaggle_gen.ipynb")
