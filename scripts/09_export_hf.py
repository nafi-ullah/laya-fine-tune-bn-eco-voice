"""Export the dataset for the Hugging Face Hub → data/hf_export/ (Parquet per config/split + README.md).

Configs
  intent         row-level: original splits, text, 15-way intent, script, Bangla-script
                 transliteration, second-choice intent, escalate flag
  order_confirm  replies to "shall I confirm your order?" → yes / no / repeat / out_of_scope
  laya           ready-to-train typed-decision cases (state / questions / gold as JSON strings)

    .venv/bin/python scripts/09_export_hf.py
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from common import BUILD, CACHE, DATA, INTENTS, RAW, load_cfg, read_jsonl

OUT = DATA / "hf_export"
CONFIRM = ["yes", "no", "repeat", "out_of_scope"]
SCRIPT_NAMES = {"bn": "Bangla script", "bl": "Banglish (romanized Bangla)", "en": "English",
                "mx": "Mixed Bangla/Latin script", "bn_translit": "Bangla script (transliterated)"}


def export_intent() -> dict[str, pd.DataFrame]:
    labels = {r["row_id"]: r for r in read_jsonl(CACHE / "label.jsonl")}
    translit = {r["row_id"]: r["bn_text"] for r in read_jsonl(CACHE / "translit.jsonl")}
    out = {}
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(RAW / f"{split}.parquet")
        lab = df["row_id"].map(labels)
        df = pd.DataFrame({
            "id": df["row_id"],
            "text": df["text"],
            "intent": df["intent"],
            "script": df["script"],
            "bangla_text": df["row_id"].map(translit),
            "alt_intent": lab.map(lambda x: x["alt_intent"] if isinstance(x, dict) else None),
            "escalate": [int(i == "agent_request" or (isinstance(x, dict) and x["escalate"] == 1))
                         for i, x in zip(df["intent"], lab)],
        })
        out[split] = df
    return out


def export_order_confirm() -> dict[str, pd.DataFrame]:
    gen = {r["text"]: r for r in read_jsonl(CACHE / "confirm_gen.jsonl")}
    out = {}
    for split in ("train", "test"):
        rows = []
        for c in read_jsonl(BUILD / f"cases_{split}.jsonl"):
            if c["workflow"] != "order_confirm":
                continue
            text = c["meta"]["text"]
            g = gen.get(text)
            rows.append({"text": text, "label": c["gold"]["order_confirm"]["label"],
                         "origin": "synthetic" if c["meta"]["source"] == "llm_generated" else "intent_corpus",
                         "style": g["style"] if g else None,
                         "group": c["meta"]["group"]})
        out[split] = pd.DataFrame(rows)
    return out


def export_laya() -> dict[str, pd.DataFrame]:
    out = {}
    for name, fname in (("train", "train.jsonl"), ("calibration", "calib.jsonl"), ("test", "cases_test.jsonl")):
        seen, rows = set(), []
        for c in read_jsonl(BUILD / fname):
            if c["id"] in seen:                     # train.jsonl repeats order_confirm (class weight ×2)
                continue
            seen.add(c["id"])
            rows.append({"id": c["id"], "decision": c["workflow"],
                         "state": json.dumps(c["state"], ensure_ascii=False),
                         "questions": json.dumps(c["questions"], ensure_ascii=False),
                         "gold": json.dumps(c["gold"], ensure_ascii=False),
                         "text": c["meta"]["text"], "script": c["meta"]["script"]})
        out[name] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------ card

def md_table(header: list[str], rows: list[list]) -> str:
    numeric = [all(isinstance(r[i], (int, float)) for r in rows if i < len(r)) for i in range(len(header))]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dist_table(frames: dict[str, pd.DataFrame], col: str, order: list[str]) -> str:
    splits = list(frames)
    rows = []
    for v in order:
        rows.append([f"`{v}`"] + [int((frames[s][col] == v).sum()) for s in splits])
    rows.append(["**total**"] + [len(frames[s]) for s in splits])
    return md_table([col] + splits, rows)


def card(cfg: dict, intent: dict, oc: dict, laya: dict) -> str:
    src = cfg["source"]
    n_intent = sum(len(d) for d in intent.values())
    n_translit = sum(int(d["bangla_text"].notna().sum()) for d in intent.values())
    n_romanized = sum(int(d["script"].isin(["bl", "mx"]).sum()) for d in intent.values())
    n_alt = sum(int(d["alt_intent"].notna().sum()) for d in intent.values())
    n_esc = sum(int(d["escalate"].sum()) for d in intent.values())
    n_syn = sum(int((d["origin"] == "synthetic").sum()) for d in oc.values())
    decisions = load_cfg("decisions")

    script_rows = []
    for s in ("bn", "bl", "en", "mx"):
        script_rows.append([f"`{s}` — {SCRIPT_NAMES[s]}"] + [int((intent[k]["script"] == s).sum()) for k in intent])
    laya_rows = []
    for d in ("order_confirm", "intent", "escalate"):
        laya_rows.append([f"`{d}`"] + [int((laya[k]["decision"] == d).sum()) for k in laya])
    esc_rows = [[f"`{v}`"] + [int((intent[k]["escalate"] == v).sum()) for k in intent] for v in (0, 1)]
    style_counts = Counter(s for d in oc.values() for s in d["style"].dropna())

    example_intent = intent["train"][intent["train"]["bangla_text"].notna()].iloc[0]
    example_oc = oc["train"][oc["train"]["label"] == "no"].iloc[0]

    yaml_head = f"""---
license: {src['license']}
language:
- bn
- en
pretty_name: {cfg['pretty_name']}
task_categories:
- text-classification
task_ids:
- intent-classification
- multi-class-classification
multilinguality:
- multilingual
size_categories:
- 10K<n<100K
source_datasets:
- {src['repo']}
annotations_creators:
- machine-generated
language_creators:
- machine-generated
- expert-generated
tags:
- bangla
- bengali
- banglish
- code-mixing
- transliteration
- e-commerce
- customer-support
- voice-agent
- intent-detection
- escalation
- order-confirmation
- typed-decisions
configs:
- config_name: intent
  default: true
  data_files:
  - split: train
    path: intent/train.parquet
  - split: validation
    path: intent/validation.parquet
  - split: test
    path: intent/test.parquet
- config_name: order_confirm
  data_files:
  - split: train
    path: order_confirm/train.parquet
  - split: test
    path: order_confirm/test.parquet
- config_name: laya
  data_files:
  - split: train
    path: laya/train.parquet
  - split: calibration
    path: laya/calibration.parquet
  - split: test
    path: laya/test.parquet
---
"""
    body = f"""
# {cfg['pretty_name']}

Decision data for a **Bangla-speaking e-commerce voice agent**. Given a customer utterance (as a
speech-to-text system would transcribe it), it answers three questions:

1. **What does the customer want?** — 15-way support intent.
2. **Should a human take over?** — escalation flag.
3. **How did the customer answer "shall I confirm your order?"** — `yes` / `no` / `repeat` / `out_of_scope`.

Text is in Bangla script, Banglish (romanized Bangla), English, and mixed script, reflecting how
Bangladeshi customers actually write and speak. {n_translit:,} of the {n_romanized:,} Banglish or
mixed-script messages also have a **Bangla-script version**, matching the output of Bangla speech
recognition.

## Source and credit

This dataset is a derivative of **[{src['repo']}]({src['url']})**, which is licensed
**CC BY-NC-SA 4.0**. The `intent` config keeps its texts, intent labels, script tags and
train/validation/test splits unchanged, and adds new columns. The `order_confirm` config adds new
order-confirmation replies. All credit for the original intent corpus goes to its authors;
please cite and credit that dataset whenever you use this one.

Following the ShareAlike term, this dataset is released under the same **CC BY-NC-SA 4.0**
licence: **non-commercial use only**.

## Configurations

| config | rows | splits | use it for |
|---|---:|---|---|
| `intent` | {n_intent:,} | train / validation / test | intent classification, escalation, script-robust NLU |
| `order_confirm` | {sum(len(d) for d in oc.values()):,} | train / test | classifying the reply to an order-confirmation question |
| `laya` | {sum(len(d) for d in laya.values()):,} | train / calibration / test | training typed-decision models directly (soft targets) |

```python
from datasets import load_dataset

intent = load_dataset("{{user}}/{cfg['repo_name']}", "intent")
confirm = load_dataset("{{user}}/{cfg['repo_name']}", "order_confirm")
cases = load_dataset("{{user}}/{cfg['repo_name']}", "laya")
```

## `intent`

### Fields

| field | type | description |
|---|---|---|
| `id` | string | row id, `<split>-<index>` in the original split |
| `text` | string | customer message as written |
| `intent` | string | gold intent — one of the 15 below (unchanged from the source) |
| `script` | string | `bn` Bangla script · `bl` Banglish · `en` English · `mx` mixed |
| `bangla_text` | string \\| null | Bangla-script rendering of `bl` / `mx` messages (null otherwise) |
| `alt_intent` | string \\| null | a second plausible intent for genuinely ambiguous messages (never equal to `intent`) |
| `escalate` | int (0/1) | 1 if a human agent should take over (always 1 for `agent_request`) |

Example:

```json
{json.dumps({k: (None if pd.isna(v) else (int(v) if k == "escalate" else v)) for k, v in example_intent.items()}, ensure_ascii=False, indent=2)}
```

### Intent labels

{md_table(["intent", "meaning"], [[f"`{k}`", v] for k, v in decisions["intent"]["criteria"].items()])}

### Label distribution

{dist_table(intent, "intent", INTENTS)}

### Script distribution

{md_table(["script"] + list(intent), script_rows)}

{n_translit:,} messages have a `bangla_text` rendering; {n_alt:,} have an `alt_intent`.

### Escalation distribution

{md_table(["escalate"] + list(intent), esc_rows)}

## `order_confirm`

Customer replies to the agent's question **"আপনার অর্ডারটি কি কনফার্ম করে দেব?"** ("Shall I confirm
your order?"), in Bangla script.

### Fields

| field | type | description |
|---|---|---|
| `text` | string | the customer's reply |
| `label` | string | `yes` confirm · `no` refuse/cancel · `repeat` asks to hear it again · `out_of_scope` says something else |
| `origin` | string | `synthetic` — written for this dataset; `intent_corpus` — an `intent` message used as an off-topic reply |
| `style` | string \\| null | phrasing style of a synthetic reply (see below) |
| `group` | string | generation group; the train/test split is by group, so near-paraphrases never cross splits |

Example:

```json
{json.dumps({k: (None if pd.isna(v) else v) for k, v in example_oc.items()}, ensure_ascii=False, indent=2)}
```

### Label distribution

{dist_table(oc, "label", CONFIRM)}

{n_syn:,} replies are synthetic. They deliberately include **hard negatives**, the cases where
keyword rules fail:

{md_table(["style", "replies"], [[s, n] for s, n in style_counts.most_common()])}

For example, "না না ঠিক আছে দিয়ে দেন" (*no no, it's fine, send it*) is `yes`, and
"হ্যাঁ, এখন আর লাগবে না" (*yes, I don't need it any more*) is `no`.

## `laya`

The same data as ready-to-train **typed-decision cases**, in the format of
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): one
question per case, `gold` holding **soft target probabilities**.

| field | type | description |
|---|---|---|
| `id` | string | case id |
| `decision` | string | `order_confirm` · `intent` · `escalate` |
| `state` | JSON string | the input: a message, or for `order_confirm` a conversation `[agent question, customer reply]` |
| `questions` | JSON string | `{{name: {{type, instructions, criteria}}}}` — `choice` or `noul` (yes/no proposition) |
| `gold` | JSON string | `{{name: {{type, label, probabilities}}}}` |
| `text`, `script` | string | the customer text and its script, for filtering |

{md_table(["decision"] + list(laya), laya_rows)}

**Soft targets.**
- `intent`: 0.90 on the gold intent, the rest spread evenly. Ambiguous messages get 0.70 gold / 0.20 `alt_intent` / 0.10 spread.
- `order_confirm`: 0.92 on the gold label.
- `escalate`: P(true) = 0.95 for `agent_request`, 0.80 for other escalations, 0.10 otherwise.

**Phrasing variety.** Training cases draw one of several paraphrased instructions and shuffle the
option order. Test and calibration cases use one fixed wording.

**No leakage.**
- `test` is built only from the source's test split and the order-confirm test groups.
- Training cases that are exact or near duplicates of any test text were removed (multilingual
  embeddings, cosine ≥ 0.98), along with a small held-out evaluation set (cosine ≥ 0.95).
- `calibration` is a 10% slice of the training pool, held out for temperature scaling.

## How the extra fields were made

- `bangla_text` and `alt_intent` / `escalate` are **machine-assisted annotations**. The gold `intent` is
  never changed. `alt_intent` is only set where an independent annotation pass disagreed with the gold
  label or saw a genuine second reading.
- The synthetic `order_confirm` replies are machine-generated from seed phrases in several styles.
  An independent blind classification pass agreed with every intended label.

## Limitations

- **The yes/no order-confirm replies have not yet been individually reviewed by a person.** Use the
  `test` split for evaluation with that in mind.
- The source corpus is itself largely machine-generated, so real customer phrasing (typos,
  disfluencies, dialects) is under-represented. Validate on real transcripts before relying on a
  model trained on this.
- `escalate` reflects a policy (explicit requests, strong anger, fraud or legal issues), not ground truth.
- Order-confirm replies cover Bangla script only.

## Licence

**CC BY-NC-SA 4.0**, inherited from [{src['repo']}]({src['url']}). You may share and adapt it for
**non-commercial** purposes. You must **give appropriate credit** to the original dataset and to this
one, and you must distribute derivatives under the same licence.
"""
    return yaml_head + body


def main() -> None:
    cfg = load_cfg("hf")
    intent, oc, laya = export_intent(), export_order_confirm(), export_laya()
    for name, frames in (("intent", intent), ("order_confirm", oc), ("laya", laya)):
        (OUT / name).mkdir(parents=True, exist_ok=True)
        for split, df in frames.items():
            df.to_parquet(OUT / name / f"{split}.parquet", index=False)
        print(f"{name}: " + ", ".join(f"{s}={len(d)}" for s, d in frames.items()))
    (OUT / "README.md").write_text(card(cfg, intent, oc, laya), encoding="utf-8")
    print(f"→ {OUT.relative_to(DATA.parent)}/ (README.md + parquet)")


if __name__ == "__main__":
    main()
