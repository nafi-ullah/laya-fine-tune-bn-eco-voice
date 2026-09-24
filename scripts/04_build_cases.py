"""Step 4 — build Laya typed-decisions cases (one question per case) with soft targets.

Inputs : data/raw/*.parquet, data/cache/{translit,label,confirm_gen}.jsonl, reports/confirm_review.csv,
         data/extra/*.jsonl (your own labelled rows — see config/split.yaml → extra_dir)
Outputs: data/build/cases_{train,test}.jsonl   (train = dataset train+validation pool)

Train cases get a random instruction paraphrase + shuffled option order (and a random agent
turn for order_confirm); test cases use the canonical wording and order (`instructions[0]`,
`agent_turns[0]`) — the format used at inference.

    .venv/bin/python scripts/04_build_cases.py
    .venv/bin/python scripts/04_build_cases.py --allow-unreviewed   # smoke test ONLY: takes
        non-suspect yes/no rows that nobody reviewed yet; the run is tagged unreviewed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter

import jsonschema
import pandas as pd

from common import BUILD, CACHE, INTENTS, RAW, REPORTS, ROOT, SCHEMAS, load_cfg, read_jsonl, script_of, write_jsonl

POOL = {"train": "train", "validation": "train", "test": "test"}


class Builder:
    def __init__(self, decisions: dict, rng: random.Random, validator):
        self.d = decisions
        self.rng = rng
        self.validator = validator
        self.cases: dict[str, list[dict]] = {"train": [], "test": []}

    def question(self, name: str, split: str) -> dict:
        spec = self.d[name]
        crit = dict(spec["criteria"])
        if split == "train":
            ins = self.rng.choice(spec["instructions"])
            keys = list(crit)
            self.rng.shuffle(keys)
            crit = {k: crit[k] for k in keys}
        else:
            ins = spec["instructions"][0]
        return {"type": spec["type"], "instructions": ins, "criteria": crit}

    def add(self, workflow: str, qname: str, split: str, state, probs: dict, meta: dict, cid: str):
        q = self.question(qname, split)
        if q["type"] == "choice":
            probs = {k: round(probs.get(k, 0.0), 6) for k in q["criteria"]}   # option order of this case
        label = max(probs, key=probs.get)
        case = {"id": cid, "workflow": workflow, "state": state, "questions": {qname: q},
                "gold": {qname: {"type": q["type"], "label": label, "probabilities": probs}},
                "meta": meta}
        self.validator.validate(case)
        self.cases[split].append(case)


def spread(gold: str, p_gold: float, options: list[str], alt: str | None = None, p_alt: float = 0.0) -> dict:
    rest = [o for o in options if o not in (gold, alt)]
    left = 1.0 - p_gold - (p_alt if alt else 0.0)
    probs = {o: left / len(rest) for o in rest}
    probs[gold] = p_gold
    if alt:
        probs[alt] = p_alt
    return probs


def load_review(allow_unreviewed: bool) -> dict[tuple[str, str], bool]:
    path = REPORTS / "confirm_review.csv"
    if not path.exists():
        return {}
    ok = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mark = r["accept"].strip().lower()
            if r["label"] == "repeat":
                ok[(r["label"], r["text"])] = mark not in ("n", "no", "0")
            elif mark in ("y", "yes", "1"):
                ok[(r["label"], r["text"])] = True
            elif allow_unreviewed and mark == "" and not r["suspect"]:
                ok[(r["label"], r["text"])] = True
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unreviewed", action="store_true")
    args = ap.parse_args()

    split_cfg = load_cfg("split")
    decisions = load_cfg("decisions")
    st = split_cfg["soft_targets"]
    rng = random.Random(split_cfg["seed"])
    schema = json.loads((SCHEMAS / "laya_case.schema.json").read_text())
    b = Builder(decisions, rng, jsonschema.Draft202012Validator(schema))

    df = pd.concat([pd.read_parquet(RAW / f"{s}.parquet") for s in POOL], ignore_index=True)
    translit = {r["row_id"]: r["bn_text"] for r in read_jsonl(CACHE / "translit.jsonl")}
    labels = {r["row_id"]: r for r in read_jsonl(CACHE / "label.jsonl")}
    enriched = set(labels) | set(translit)
    if enriched:   # smoke runs enrich a sample; only build cases for rows the LLM saw
        df = df[df["row_id"].isin(enriched)]

    # ---- intent + escalate (dataset rows, plus a Bangla-script transliteration where available)
    esc = st["escalate"]
    for r in df.to_dict("records"):
        split = POOL[r["row_id"].split("-")[0]]
        lab = labels.get(r["row_id"])
        variants = [(r["text"], r["script"])]
        if r["row_id"] in translit:
            variants.append((translit[r["row_id"]], "bn_translit"))
        if lab and lab["ambiguity"] == "high" and lab["alt_intent"]:
            ip = spread(r["intent"], st["intent_ambiguous_gold"], INTENTS, lab["alt_intent"], st["intent_ambiguous_alt"])
        else:
            ip = spread(r["intent"], st["intent_gold"], INTENTS)
        if r["intent"] == "agent_request":
            p_esc = esc["agent_request"]
        elif lab is not None:
            p_esc = esc["llm_escalate_true"] if lab["escalate"] == 1 else esc["llm_escalate_false"]
        else:
            p_esc = None                                  # no label → no escalate case
        for vi, (text, script) in enumerate(variants):
            meta = {"source": "banglaecomintent", "text": text, "script": script, "nc": True,
                    "row_id": r["row_id"], "group": r["row_id"]}
            b.add("intent", "intent", split, text, ip, meta, f"intent-{r['row_id']}-{vi}")
            if p_esc is not None:
                b.add("escalate", "escalate", split, text, {"false": 1 - p_esc, "true": p_esc},
                      meta, f"escalate-{r['row_id']}-{vi}")

    # ---- order_confirm: generated yes/no/repeat (reviewed) + real out_of_scope rows
    oc = decisions["order_confirm"]
    options = list(oc["criteria"])
    accepted = load_review(args.allow_unreviewed)
    gen = [r for r in read_jsonl(CACHE / "confirm_gen.jsonl") if accepted.get((r["label"], r["text"]))]
    groups = sorted({r["group"] for r in gen})
    rng.shuffle(groups)
    n_test = round(len(groups) * split_cfg["confirm_generation"]["test_fraction"])
    test_groups = set(groups[:n_test])

    def confirm_state(split: str, text: str) -> list[dict]:
        agent = oc["agent_turns"][0] if split == "test" else rng.choice(oc["agent_turns"])
        return [{"role": "assistant", "content": agent}, {"role": "user", "content": text}]

    for i, r in enumerate(gen):
        split = "test" if r["group"] in test_groups else "train"
        meta = {"source": "llm_generated", "text": r["text"], "script": "bn", "nc": False,
                "group": f"gen-{r['group']}", "style": r["style"]}
        b.add("order_confirm", "order_confirm", split, confirm_state(split, r["text"]),
              spread(r["label"], st["confirm_gold"], options), meta, f"confirm-gen-{i}")

    oos_cfg = split_cfg["confirm_out_of_scope"]
    per_class = oos_cfg["per_class_target"]
    oos_pool = df[df["intent"].isin(oos_cfg["from_intents"])]
    for split, n in (("train", per_class),
                     ("test", round(per_class * split_cfg["confirm_generation"]["test_fraction"]))):
        rows = oos_pool[oos_pool["row_id"].map(lambda x: POOL[x.split("-")[0]]) == split]
        # Phone calls are transcribed in Bangla script: prefer bn rows and transliterations.
        ids = set(rows["row_id"])
        texts = [(t, rid) for rid, t in translit.items() if rid in ids]
        texts += [(t, rid) for t, rid, s in zip(rows["text"], rows["row_id"], rows["script"]) if s == "bn"]
        rng.shuffle(texts)
        for j, (text, rid) in enumerate(texts[:n]):
            meta = {"source": "banglaecomintent", "text": text, "script": "bn", "nc": True,
                    "row_id": rid, "group": rid}
            b.add("order_confirm", "order_confirm", split, confirm_state(split, text),
                  spread("out_of_scope", st["confirm_gold"], options), meta, f"confirm-oos-{split}-{j}")

    # ---- your own labelled rows (data/extra/*.jsonl) — not derived from the NC dataset
    n_extra = Counter()
    for path in sorted((ROOT / split_cfg["extra_dir"]).glob("*.jsonl")):
        for i, r in enumerate(read_jsonl(path)):
            dec, text, label = r["decision"], r["text"].strip(), str(r["label"])
            split = r.get("split", "train")
            meta = {"source": r.get("source", path.stem), "text": text, "script": script_of(text), "nc": False,
                    "group": f"extra-{path.stem}-{i}"}
            cid = f"extra-{path.stem}-{i}"
            if dec == "order_confirm":
                assert label in options, f"{path.name}:{i} bad order_confirm label {label!r}"
                b.add(dec, dec, split, confirm_state(split, text), spread(label, st["confirm_gold"], options), meta, cid)
            elif dec == "intent":
                assert label in INTENTS, f"{path.name}:{i} bad intent {label!r}"
                b.add(dec, dec, split, text, spread(label, st["intent_gold"], INTENTS), meta, cid)
            elif dec == "escalate":
                p = esc["llm_escalate_true"] if label.lower() == "true" else esc["llm_escalate_false"]
                b.add(dec, dec, split, text, {"false": 1 - p, "true": p}, meta, cid)
            else:
                raise ValueError(f"{path.name}:{i} unknown decision {dec!r}")
            n_extra[(dec, split)] += 1
    if n_extra:
        print("extra rows: " + ", ".join(f"{d}/{s}={n}" for (d, s), n in sorted(n_extra.items())))

    for split, cases in b.cases.items():
        n = write_jsonl(BUILD / f"cases_{split}.jsonl", cases)
        c = Counter((x["workflow"], x["gold"][next(iter(x["gold"]))]["label"]) for x in cases)
        print(f"{split}: {n} cases · " + ", ".join(f"{w}:{l}={k}" for (w, l), k in sorted(c.items())
                                                   if w != "intent"))
    tag = "UNREVIEWED yes/no (smoke only)" if args.allow_unreviewed else "reviewed"
    print(f"order_confirm generated rows used: {len(gen)} ({tag})")
    (BUILD / "build_meta.json").write_text(json.dumps({"confirm_review": tag}, indent=2))


if __name__ == "__main__":
    main()
