"""Step 3 — generate ORDER_CONFIRM replies (yes / no / repeat) with the LLM (Gemini).

Seeds = seeds/anchors/{yes,no,repeat}.yaml MINUS every utterance in the
frozen eval suites (several anchors are eval items). Each LLM call gets a meaning, a style
(round-robin over config/llm.yaml → confirm_gen.styles, to force hard negatives) and a few
seeds, and returns ~12 replies. A call is a "group": the train/test split is done per group,
so paraphrases of the same seeds never land on both sides.

Output: data/cache/confirm_gen.jsonl (resumable) and reports/confirm_review.csv.

HUMAN REVIEW: open reports/confirm_review.csv and put y / n in the `accept` column for every
yes/no row (repeat rows are accepted unless marked n). 04_build_cases.py only uses accepted
yes/no rows — never auto-promote a phrase into yes/no without review.

    .venv/bin/python scripts/03_generate_confirm.py
    .venv/bin/python scripts/03_generate_confirm.py --per-class 24     # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re

import yaml

from common import (CACHE, REPORTS, ROOT, append_jsonl, frozen_eval_texts, load_cfg, load_env, norm_text,
                    read_jsonl)
from llm import GeminiLLM, extract_json

CLASSES = ("yes", "no", "repeat")

# Cheap lexical sanity flags to focus the human review (not used to auto-label).
_NEG = re.compile(r"না|নেই|নাই|নিব না|লাগবে না|থাক|ক্যান্সেল|ক্যানসেল|বাতিল|বাদ|cancel", re.I)
_TRAILING_NEG = re.compile(r"(লাগবে না|নিব না|নেব না|চাই না|দরকার নেই|থাক|ক্যান্সেল|ক্যানসেল|বাতিল|বাদ দিন|বাদ দেন)\s*[।.!?]*$")


def suspect(label: str, text: str) -> str:
    if label == "no" and not _NEG.search(text):
        return "no negation word"
    if label == "yes" and _TRAILING_NEG.search(text):
        return "ends with a negation"
    if not any("ঀ" <= c <= "৿" for c in text):
        return "no Bangla script"
    return ""


def blind_check(llm, gcfg_v: dict, rows: list[dict]) -> None:
    """Classify generated replies without showing the intended label (cached per text)."""
    path = CACHE / "confirm_verify.jsonl"
    have = {r["text"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["text"] not in have]
    bs = gcfg_v["batch_size"]
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        payload = [{"id": j + 1, "text": r["text"]} for j, r in enumerate(batch)]
        user = (f"Example input:\n{gcfg_v['example_in']}\nExample output:\n{gcfg_v['example_out']}\n\n"
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}")
        try:
            out = {row[0]: row[1] for row in extract_json(llm.chat(gcfg_v["system"], user, gcfg_v["temperature"],
                                                                     15 * len(batch) + 40))
                   if isinstance(row, list) and len(row) == 2}
        except (ValueError, TypeError):
            continue
        for j, r in enumerate(batch):
            if out.get(j + 1) in ("yes", "no", "repeat", "out_of_scope"):
                append_jsonl(path, {"text": r["text"], "llm_check": out[j + 1]})


def load_seeds(split_cfg: dict, frozen: set[str]) -> dict[str, list[str]]:
    d = ROOT / split_cfg["confirm_seeds"]["anchors_dir"]
    seeds = {}
    for c in CLASSES:
        with open(d / f"{c}.yaml", encoding="utf-8") as f:
            phrases = yaml.safe_load(f)["phrases"]
        seeds[c] = [p for p in phrases if norm_text(p) not in frozen]
        print(f"seeds[{c}]: {len(seeds[c])} (dropped {len(phrases) - len(seeds[c])} that are eval items)")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, help="override confirm_generation.per_class_target")
    args = ap.parse_args()

    load_env()
    llm_cfg = load_cfg("llm")
    split_cfg = load_cfg("split")
    gcfg = llm_cfg["passes"]["confirm_gen"]
    target = args.per_class or split_cfg["confirm_generation"]["per_class_target"]
    frozen = frozen_eval_texts(split_cfg)
    seeds = load_seeds(split_cfg, frozen)
    rng = random.Random(split_cfg["seed"])

    out_path = CACHE / "confirm_gen.jsonl"
    existing = list(read_jsonl(out_path))
    seen = {norm_text(r["text"]) for r in existing} | frozen | {norm_text(s) for v in seeds.values() for s in v}
    have = {c: sum(1 for r in existing if r["label"] == c) for c in CLASSES}
    next_group = max((r["group"] for r in existing), default=-1) + 1

    llm = None
    for c in CLASSES:
        styles = gcfg["styles"][c]
        calls = 0
        prior_calls = len({r["group"] for r in existing if r["label"] == c})   # resume the rotation
        while have[c] < target and calls < 3 * (target // gcfg["per_call"] + 1):
            if llm is None:
                llm = GeminiLLM(llm_cfg)
            style = styles[(prior_calls + calls) % len(styles)]   # every style, incl. the hard negatives
            examples = rng.sample(seeds[c], min(5, len(seeds[c])))
            user = (f"Meaning: {gcfg['meanings'][c]}\nStyle: {style}\n"
                    f"Seed examples (do not copy): {examples}\n"
                    f"Write {gcfg['per_call']} different replies.")
            calls += 1
            try:
                reply = llm.chat(gcfg["system"], user, gcfg["temperature"], 40 * gcfg["per_call"] + 40)
                utts = extract_json(reply)["utterances"]
            except (ValueError, KeyError, TypeError):
                continue
            group, added = next_group, 0
            next_group += 1
            for u in utts:
                if not isinstance(u, str):
                    continue
                u = u.strip().strip('"')
                k = norm_text(u)
                if not k or k in seen or len(u) > 120:
                    continue
                seen.add(k)
                append_jsonl(out_path, {"label": c, "text": u, "style": style, "group": group,
                                        "prompt_version": gcfg["prompt_version"]})
                have[c] += 1
                added += 1
            print(f"  [{c}] +{added} ({have[c]}/{target}) style='{style[:40]}'", flush=True)

    rows = list(read_jsonl(out_path))
    if llm is None:
        llm = GeminiLLM(llm_cfg)
    blind_check(llm, llm_cfg["passes"]["confirm_verify"], rows)
    check = {r["text"]: r["llm_check"] for r in read_jsonl(CACHE / "confirm_verify.jsonl")}

    # Review sheet — keeps any accept marks already entered by a reviewer.
    review_path = REPORTS / "confirm_review.csv"
    prior = {}
    if review_path.exists():
        with open(review_path, encoding="utf-8") as f:
            prior = {(r["label"], r["text"]): r["accept"] for r in csv.DictReader(f)}
    REPORTS.mkdir(exist_ok=True)
    def risk(r):   # disagreements and lexical suspects first, yes/no before repeat
        agree = check.get(r["text"]) == r["label"]
        return (agree, not suspect(r["label"], r["text"]), r["label"] == "repeat", r["group"])

    rows.sort(key=risk)
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "text", "llm_check", "suspect", "style", "group", "accept"])
        for r in rows:
            w.writerow([r["label"], r["text"], check.get(r["text"], ""), suspect(r["label"], r["text"]),
                        r["style"], r["group"], prior.get((r["label"], r["text"]), "")])
    n_sus = sum(1 for r in rows if suspect(r["label"], r["text"]))
    n_dis = sum(1 for r in rows if check.get(r["text"]) != r["label"])
    print(f"blind cross-check disagrees with the intended label on {n_dis}/{len(rows)} rows (sorted to the top)")
    if llm is not None:
        print(f"usage: {llm.usage()} ({llm_cfg['model']})")
    print(f"generated {len(rows)} → reports/confirm_review.csv ({n_sus} flagged suspect). "
          f"Fill the `accept` column (y/n) for yes/no rows before step 04.")


if __name__ == "__main__":
    main()
