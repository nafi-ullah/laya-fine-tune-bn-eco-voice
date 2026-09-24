"""Stage and push the fine-tuned model to the Hugging Face Hub (private by default, config/hf.yaml → model).

Repo layout
  /             PyTorch checkpoint  → laya.load("<repo>")
  mlx/          MLX FP16 checkpoint → laya_mlx.load("<repo>", subfolder="mlx")
  questions.yaml the exact question schemas the model was trained on
  results/      FINETUNE_RESULTS.md, ANALYSIS.md, data_card.md
  README.md     model card (results, usage, training details, credits, licence)

    .venv/bin/python scripts/11_push_model.py            # stage + push
    .venv/bin/python scripts/11_push_model.py --stage    # stage only (data/hf_model/)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import yaml
from huggingface_hub import HfApi

from common import BUILD, CONFIG, DATA, REPORTS, ROOT, load_cfg, load_env, read_jsonl

STAGE = DATA / "hf_model"


def cascade_stats() -> dict:
    out = {}
    for s in ("confirm_frozen", "confirm_test"):
        rows = list(read_jsonl(BUILD / f"cascade_out_{s}.jsonl"))
        if rows:
            out[s] = {"acc": sum(r["pred"] == r["label"] for r in rows) / len(rows),
                      "leaks": sum({r["pred"], r["label"]} == {"yes", "no"} for r in rows), "n": len(rows)}
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def card(cfg: dict, user: str, res: dict, casc: dict, run: dict, dataset_repo: str) -> str:
    m = cfg["model"]
    M, lat, mem = res["metrics"], res["latency"], res["memory_mib"]
    leaks = {k: {s: len(v) for s, v in d.items()} for k, d in res["leaks"].items()}
    repo = f"{user}/{m['repo_name']}"
    decisions = load_cfg("decisions")
    oc, it, es = decisions["order_confirm"], decisions["intent"], decisions["escalate"]

    def row(name, key, fn, c=None):
        return f"| {name} | " + " | ".join(fn(M[k][key]) for k in ("base", "R1")) + f" | {c if c is not None else '—'} |"

    ev = [("order_confirm (generated test)", "confirm_test"), ("order_confirm (hand-written frozen)", "confirm_frozen"),
          ("intent (BanglaEComIntent test)", "intent_test"), ("escalate", "escalate_test")]
    model_index = "\n".join(
        f"""  - task: {{type: text-classification, name: {name}}}
    dataset: {{type: {dataset_repo}, name: {cfg['pretty_name']}}}
    metrics:
    - {{type: accuracy, value: {M['R1'][key]['acc']:.4f}}}"""
        for name, key in ev)

    head = f"""---
license: {m['license']}
base_model: {m['base_model']}
library_name: laya
language:
- bn
- en
pipeline_tag: text-classification
datasets:
- {dataset_repo}
- Badhon/BanglaEComIntent
tags:
- laya
- typed-decisions
- system-one
- bangla
- bengali
- banglish
- e-commerce
- voice-agent
- intent-detection
- escalation
- order-confirmation
- mlx
model-index:
- name: {m['repo_name']}
  results:
{model_index}
---
"""
    body = f"""
# Laya-multilingual · Bangla e-commerce voice agent (R1)

A fine-tune of **[{m['base_model']}](https://huggingface.co/{m['base_model']})** (322M, mmBERT
encoder) that makes three **typed decisions** for a Bangla-speaking e-commerce voice agent, in
one forward pass and with no generated tokens:

| decision | type | options |
|---|---|---|
| `order_confirm` | choice | `yes` · `no` · `repeat` · `out_of_scope` — how the customer answered "shall I confirm your order?" |
| `intent` | choice | 15 support intents (order status, cancel, refund, delivery, payment, complaint, human agent, …) |
| `escalate` | noul | P(a human agent should take over) |

Both **PyTorch** (repo root) and **MLX FP16 for Apple Silicon** (`mlx/`) checkpoints are included.
PyTorch and MLX picked the same option on 50/50 test cases.

> **Status: research preview (round 1).** It is strong on intent, escalation and full-sentence
> replies, but it misreads **one- and two-word replies** (হ্যাঁ, জি, না) as `out_of_scope`. See
> [Limitations](#limitations). Don't use it as the only yes/no decision-maker.

## Results

Every model gets the **same questions** ([`questions.yaml`](questions.yaml), `instructions[0]`).
Latency is measured on an M5 MacBook Pro 16 GB with the MLX FP16 checkpoint, one question per call.

| test set | base (zero-shot) | **this model** | rule/embedding cascade |
|---|---:|---:|---:|
""" + "\n".join([
        row("order_confirm, generated test (186)", "confirm_test", lambda x: pct(x["acc"]),
            pct(casc["confirm_test"]["acc"]) if "confirm_test" in casc else None),
        f"| ↳ yes↔no mix-ups | {leaks['base']['confirm_test']} | **{leaks['R1']['confirm_test']}** | "
        f"{casc.get('confirm_test', {}).get('leaks', '—')} |",
        row("order_confirm, hand-written frozen set (23)", "confirm_frozen", lambda x: pct(x["acc"]),
            pct(casc["confirm_frozen"]["acc"]) if "confirm_frozen" in casc else None),
        f"| ↳ yes↔no mix-ups | {leaks['base']['confirm_frozen']} | **{leaks['R1']['confirm_frozen']}** | "
        f"{casc.get('confirm_frozen', {}).get('leaks', '—')} |",
        row("intent, BanglaEComIntent test, all scripts (1,091)", "intent_test", lambda x: pct(x["acc"])),
        row("escalate (1,091)", "escalate_test", lambda x: pct(x["acc"])),
        row("router suite, Bangla (16)", "router_bn", lambda x: pct(x["acc"])),
        row("router suite, English (16)", "router_en", lambda x: pct(x["acc"])),
        f"| latency P50 / P95 per question | {lat['base']['p50']:.1f} / {lat['base']['p95']:.1f} ms | "
        f"**{lat['R1']['p50']:.1f} / {lat['R1']['p95']:.1f} ms** | < 0.1 ms |",
        f"| peak memory (MLX) | {mem['base']:.0f} MiB | {mem['R1']:.0f} MiB | — |",
    ]) + f"""

- **Intent by script:** Bangla script 87.1%, Bangla transliterated 93.7%, Banglish 73.5%, English 85.4%,
  mixed 94.3% (base: 17–48%).
- **Confidence is meaningful in-distribution.** Intent predictions with top probability ≥ 0.8 are
  {pct(M['R1']['intent_test']['acc_hi'])} accurate and cover {pct(M['R1']['intent_test']['cov_hi'])} of
  messages; for escalate, {pct(M['R1']['escalate_test']['acc_hi'])} at {pct(M['R1']['escalate_test']['cov_hi'])}.
  (Base: 39% and 13%.) Brier score on intent is {M['R1']['intent_test']['brier']:.3f} (base {M['base']['intent_test']['brier']:.3f}).
- **Every frozen-set error is `out_of_scope`**, not a yes↔no swap. In a voice agent that means "escalate / ask again",
  which is the safe outcome.
- The **cascade** column is the agent's existing regex + embedding intent cascade, which only makes the order-confirm
  decision. It's near-perfect on the phrases it was built from but drops to {pct(casc.get('confirm_test', {}).get('acc', 0))} with
  {casc.get('confirm_test', {}).get('leaks', '?')} yes↔no mix-ups on unseen phrasings. This model scores {pct(M['R1']['confirm_test']['acc'])}
  with {leaks['R1']['confirm_test']} mix-up(s).

Full per-set tables, confusion matrices and error lists: [`results/FINETUNE_RESULTS.md`](results/FINETUNE_RESULTS.md).
Analysis: [`results/ANALYSIS.md`](results/ANALYSIS.md).

## Usage

The model was trained on specific question wordings and option sets, so use the schemas in
[`questions.yaml`](questions.yaml) (`instructions[0]`, options as listed). For `order_confirm`, pass the
conversation: the agent's question, then the customer's reply.

**Apple Silicon (MLX)**, `pip install laya-mlx`:

```python
import laya_mlx as laya

agent = laya.load("{repo}", subfolder="mlx")
state = [
    {{"role": "assistant", "content": "{oc['agent_turns'][0]}"}},
    {{"role": "user", "content": "না না ঠিক আছে, দিয়ে দেন"}},
]
questions = {{
    "order_confirm": {{
        "type": "choice",
        "instructions": "{oc['instructions'][0]}",
        "criteria": {json.dumps(oc['criteria'], ensure_ascii=False)},
    }}
}}
print(agent.predict(state, questions)["answers"]["order_confirm"])
```

**PyTorch (CPU / CUDA / MPS)**, `pip install laya`:

```python
import laya

agent = laya.load("{repo}")
answers = agent.predict(
    "আমার অর্ডারটা এখনো আসেনি, কবে পাব?",
    {{
        "intent": {{"type": "choice", "instructions": "{it['instructions'][0]}",
                    "criteria": {{...}}}},          # the 15 intents from questions.yaml
        "escalate": {{"type": "noul", "instructions": "{es['instructions'][0]}",
                      "criteria": {json.dumps(es['criteria'], ensure_ascii=False)}}},
    }},
)["answers"]
```

Both calls return `choice` + `probabilities` (choice) or `noul` = P(true), plus `confidence`.

## Training

| | |
|---|---|
| base | [{m['base_model']}](https://huggingface.co/{m['base_model']}) (mmBERT-base encoder + 2-layer decision head) |
| data | [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}), `laya` config: {run['train_items']:,} training items (order_confirm ×2 weight), 10% calibration slice |
| objective | Laya's RLCD recipe: soft cross-entropy on gold probabilities + noisy-logit policy gradient with a proper-scoring-rule reward |
| optimiser | AdamW; lr 2.5e-5 (encoder) / 1e-4 (head); cosine schedule; weight decay 0.01; grad-clip 1.0 |
| batch | micro-batch 8 × grad-accum 4; 4 epochs; exploration σ 0.4 → 0.1; group size 4 |
| precision | bf16 autocast, gradient checkpointing |
| hardware | Apple M5 MacBook Pro 16 GB (MPS), {run['minutes']:.0f} min, {run['sec_per_step']:.2f} s/step |
| calibration | per-type temperature fitted on the held-out slice: {[round(t, 3) for t in run['temperatures']]} (choice, score, noul) |
| calibration-slice accuracy | {", ".join(f"{k} {100*v:.1f}%" for k, v in run['calib_accuracy'].items())} |
| MLX export | `laya-mlx convert --dtype float16` |

Training inputs are built with Laya's own `build_sequence` (conversation states truncated from the left, as at
inference). Train cases use varied instruction wordings and shuffled option order; evaluation uses a single
fixed wording.

## Limitations

- **Short replies.** Only 5 of 917 generated order-confirm training replies were ≤ 3 words, so the model maps
  one- and two-word replies (হ্যাঁ, জি, না, চাই না) to `out_of_scope` with high confidence. Pair it with a
  keyword rule for single-word answers, or wait for round 2, which adds short replies to the training data.
- **Unreviewed labels.** The yes/no order-confirm training replies have not yet been individually reviewed by a person.
- **Mostly synthetic data.** Real phone transcripts (disfluencies, dialects, STT errors) are under-represented.
  Validate on your own call data before relying on it.
- **Out-of-distribution confidence.** On the hand-written frozen set, the model was confidently wrong on short
  replies. Don't use a confidence threshold as a safety guarantee outside the training distribution.
- `escalate` reflects a policy (explicit requests, strong anger, fraud or legal issues), not ground truth.

## Credits

This model builds directly on the following work. Thank you to their authors.

| work | licence | used for |
|---|---|---|
""" + "\n".join(f"| [{c['name']}]({c['url']}) | {c['license']} | {c['role']} |" for c in m["credits"]) + f"""

**Changes from the base model:** all weights (encoder and decision head) were fine-tuned on the data above,
and the calibration temperatures were re-fitted. The architecture, tokenizer and input format are unchanged.

## Licence

The fine-tuned weights are released under **CC BY-NC-SA 4.0**, because they were trained on data under that
licence ([BanglaEComIntent](https://huggingface.co/datasets/Badhon/BanglaEComIntent)): **non-commercial use
only**, with attribution, and derivatives shared under the same licence. The base checkpoint and code remain under
their own licences (Apache-2.0 for Laya and laya-mlx, MIT for mmBERT); their notices are kept in [`NOTICE`](NOTICE).
"""
    return head + body


def notice(cfg: dict) -> str:
    lines = ["This repository contains a fine-tuned derivative of the works listed below.", ""]
    for c in cfg["model"]["credits"]:
        lines.append(f"- {c['name']} — {c['url']} — licensed under {c['license']} ({c['role']}).")
    lines += ["", "Modifications: all model weights were fine-tuned and the calibration temperatures re-fitted",
              "(see README.md → Training). Apache-2.0 and MIT licence texts: https://www.apache.org/licenses/LICENSE-2.0",
              "and https://opensource.org/license/mit."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true", help="stage only, do not push")
    args = ap.parse_args()

    load_env()
    cfg = load_cfg("hf")
    m = cfg["model"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{m['repo_name']}"
    dataset_repo = f"{user}/{cfg['repo_name']}"

    res = json.loads((REPORTS / "FINETUNE_RESULTS.json").read_text())
    run = next(r for r in reversed(list(read_jsonl(REPORTS / "runs.jsonl"))) if r["out"] == m["checkpoint"])

    shutil.rmtree(STAGE, ignore_errors=True)
    shutil.copytree(ROOT / m["checkpoint"], STAGE, copy_function=shutil.copyfile)
    shutil.copytree(ROOT / m["mlx_checkpoint"], STAGE / "mlx", copy_function=shutil.copyfile)
    (STAGE / "results").mkdir()
    for f in ("FINETUNE_RESULTS.md", "ANALYSIS.md", "data_card.md"):
        shutil.copyfile(REPORTS / f, STAGE / "results" / f)
    shutil.copyfile(CONFIG / "decisions.yaml", STAGE / "questions.yaml")
    (STAGE / "NOTICE").write_text(notice(cfg), encoding="utf-8")
    (STAGE / "README.md").write_text(card(cfg, user, res, cascade_stats(), run, dataset_repo), encoding="utf-8")
    print(f"staged → {STAGE.relative_to(ROOT)}")
    if args.stage:
        return

    api.create_repo(repo_id, repo_type="model", private=m["private"], exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(STAGE),
                      commit_message="Add Laya-multilingual Bangla e-commerce voice-agent fine-tune (R1)")
    print(f"pushed ({'private' if m['private'] else 'public'}) → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
