"""Generates bench/kaggle_gen.ipynb. Edit the cells here, not the .ipynb."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(keepends=True)}

cells = [
    md("""
# Filing Lens: generation bench (Kaggle T4)

Runs **Qwen2.5-3B-Instruct Q4_K_M (GGUF, llama.cpp)** on the exact retrieved pages and prompt the browser app uses
(`bench/results/gen_inputs.jsonl`, `bench/results/prompt.json`, both written by the cloud bench `npm run bench`).

Arms, one row per (instance, arm) in `gen_results.jsonl`:
- `answer`: generate an answer with `[p.N]` citations for every instance (no gate). Gated arms (calibrated LR, logprob threshold) are applied afterwards by `bench/ingest_gen.py`.
- `llm_router`: the same 3B model asked "can these excerpts answer the question? YES/NO". Its P(YES) from token logprobs is the router confidence.
- `jev`: **[ASK: what is Jev + can it run locally]**, not implemented.

Set `DRY_RUN = True` to run every cell top-to-bottom on CPU with a stub model, no downloads (used to check the notebook in CI).
"""),
    code("""
import json, os, re, sys, time, statistics, urllib.request
from pathlib import Path

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"   # set True to smoke-test without a GPU or model download
LIMIT = int(os.environ.get("LIMIT", "0")) or None  # e.g. 20 for a quick partial run
REPO_RAW = os.environ.get("REPO_RAW", "https://raw.githubusercontent.com/rajyyug1132/Filing-Lens-Private-Cited-Answers/claude/charming-keller-fo4et8")
GGUF_REPO, GGUF_FILE = "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "bench/results")) / "gen_results.jsonl"
print({"DRY_RUN": DRY_RUN, "LIMIT": LIMIT, "OUT": str(OUT)})
"""),
    code("""
def load_jsonl_or_json(name):
    # local repo checkout first (dry run / CI), then the pushed branch on GitHub
    for p in [Path("bench/results") / name, Path("/kaggle/input/filing-lens") / name]:
        if p.exists():
            txt = p.read_text()
            break
    else:
        txt = urllib.request.urlopen(f"{REPO_RAW}/bench/results/{name}", timeout=60).read().decode()
    return json.loads(txt) if name.endswith(".json") else [json.loads(l) for l in txt.splitlines() if l.strip()]

prompt = load_jsonl_or_json("prompt.json")
inputs = load_jsonl_or_json("gen_inputs.jsonl")
if DRY_RUN:
    inputs = inputs[:6]
elif LIMIT:
    inputs = inputs[:LIMIT]
print(len(inputs), "instances;", sum(i["kind"] == "own_doc" for i in inputs), "gold-answer")
"""),
    code("""
if not DRY_RUN:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    # Try prebuilt CUDA wheels first (fast), then fall back to building with CUDA (~15-20 min).
    ok = False
    for cu in ("cu124", "cu121"):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", "llama-cpp-python",
                            "--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{cu}"])
        if r.returncode == 0:
            ok = True
            break
    if not ok:
        env = dict(os.environ, CMAKE_ARGS="-DGGML_CUDA=on", FORCE_CMAKE="1")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "llama-cpp-python"], check=True, env=env)
"""),
    code("""
class StubLLM:
    \"\"\"Dry-run stand-in with llama_cpp's create_chat_completion shape.\"\"\"
    def create_chat_completion(self, messages, max_tokens=200, logprobs=False, top_logprobs=None, **kw):
        user = messages[-1]["content"]
        pages = re.findall(r"\\[p\\.(\\d+)\\]", user)
        if max_tokens == 1:
            text, lp = "YES", {"content": [{"token": "YES", "logprob": -0.4, "top_logprobs": [{"token": "YES", "logprob": -0.4}, {"token": "NO", "logprob": -1.1}]}]}
        else:
            text, lp = f"Stub answer [p.{pages[0] if pages else 1}].", {"content": [{"token": "x", "logprob": -0.2}] * 5}
        return {"choices": [{"message": {"content": text}, "logprobs": lp}], "usage": {"completion_tokens": 5 if max_tokens > 1 else 1}}

if DRY_RUN:
    llm = StubLLM()
else:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    path = hf_hub_download(GGUF_REPO, GGUF_FILE)
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, logits_all=True, verbose=False, seed=0)
print(type(llm).__name__)
"""),
    code("""
CITE_RE = re.compile(r"\\[\\s*(?:p(?:age|g)?\\.?\\s*)(\\d+(?:\\s*[,;\\u2013-]\\s*(?:p(?:age|g)?\\.?\\s*)?\\d+)*)\\s*\\]", re.I)

def cited_pages(text):
    out = []
    for m in CITE_RE.finditer(text):
        out += [int(re.sub(r"\\D", "", n)) for n in re.split(r"[,;\\u2013-]", m.group(1)) if re.sub(r"\\D", "", n)]
    return out

def user_prompt(inst):
    ctx = prompt["excerpt_joiner"].join(prompt["excerpt_format"].format(page=h["page"], text=h["text"]) for h in inst["hits"])
    return prompt["user_template"].replace("[p.0] {text}", ctx).replace("{question}", inst["question"])

ROUTER_SYSTEM = "You check whether excerpts from a company filing contain the answer to a question. Reply with exactly one word: YES or NO."

def run_answer(inst):
    msgs = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user_prompt(inst)}]
    t = time.perf_counter()
    r = llm.create_chat_completion(messages=msgs, max_tokens=200, temperature=0.0, logprobs=True, top_logprobs=1)
    dt = time.perf_counter() - t
    text = r["choices"][0]["message"]["content"].strip()
    lps = [c["logprob"] for c in ((r["choices"][0].get("logprobs") or {}).get("content") or [])]
    ntok = r.get("usage", {}).get("completion_tokens") or len(lps)
    return {"arm": "answer", "answer": text, "cited_pages": cited_pages(text), "refused": prompt["refusal"] in text.upper(),
            "latency_s": dt, "completion_tokens": ntok, "tokens_per_s": ntok / dt if dt > 0 else None,
            "logprob_conf": (2.718281828 ** (sum(lps) / len(lps))) if lps else None}

def run_router(inst):
    ctx = user_prompt(inst).rsplit("\\nAnswer with", 1)[0]
    msgs = [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": ctx + "\\nCan the excerpts answer the question? YES or NO:"}]
    t = time.perf_counter()
    r = llm.create_chat_completion(messages=msgs, max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=10)
    dt = time.perf_counter() - t
    tops = (((r["choices"][0].get("logprobs") or {}).get("content") or [{}])[0]).get("top_logprobs") or []
    import math
    p = {"YES": 0.0, "NO": 0.0}
    for tl in tops:
        k = tl["token"].strip().upper()
        if k in p:
            p[k] += math.exp(tl["logprob"])
    p_yes = p["YES"] / (p["YES"] + p["NO"]) if (p["YES"] + p["NO"]) > 0 else None
    return {"arm": "llm_router", "answer": r["choices"][0]["message"]["content"].strip(), "p_yes": p_yes, "latency_s": dt}
"""),
    code("""
OUT.parent.mkdir(parents=True, exist_ok=True)
rows = []
t0 = time.time()
with OUT.open("w") as f:
    for i, inst in enumerate(inputs):
        for fn in (run_answer, run_router):
            row = {"id": inst["id"], "qid": inst["qid"], "kind": inst["kind"], "model": "stub" if DRY_RUN else f"{GGUF_REPO}/{GGUF_FILE}", **fn(inst)}
            rows.append(row)
            f.write(json.dumps(row) + "\\n")
        if i % 25 == 0:
            print(i, f"{time.time() - t0:.0f}s")
ans = [r for r in rows if r["arm"] == "answer"]
tps = [r["tokens_per_s"] for r in ans if r["tokens_per_s"]]
print(f"wrote {len(rows)} rows to {OUT}; median tokens/s {statistics.median(tps):.1f}" if tps else f"wrote {len(rows)} rows")
"""),
    md("""
**Next:** download `gen_results.jsonl` from the notebook's Output tab, commit it to `bench/results/gen_results.jsonl`, then run
`python bench/ingest_gen.py` (or tell Claude "ingest").
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"
Path("bench/kaggle_gen.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote bench/kaggle_gen.ipynb")
