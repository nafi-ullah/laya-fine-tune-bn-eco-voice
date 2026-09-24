"""Step 6 — tokenize cases into Laya training items with upstream's own sequence builder.

    .venv/bin/python scripts/06_preprocess.py [--base convaiinnovations/laya-multilingual]
          [--only-nc-free]   # R2: drop everything derived from the CC-BY-NC dataset

Uses laya.common.build_sequence / render_options / QTYPES exactly as Agent.predict does
(including left-truncation for conversation-list states), so training inputs match
inference inputs. Writes data/build/{train,calib}_items[_ncfree].pt.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config, _load_tokenizer
from laya.common import QTYPES, build_sequence, render_options

from common import BUILD, read_jsonl

BASE_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


def base_dir(base: str) -> str:
    d = base if os.path.isdir(base) else snapshot_download(base, allow_patterns=BASE_FILES)
    _fix_tokenizer_config(d)
    return d


def to_item(tok, cfg: dict, case: dict):
    (qname, q), = case["questions"].items()
    gold = case["gold"][qname]
    iq = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    if q["type"] == "choice":
        target = [gold["probabilities"][k] for k in q["criteria"]]
    elif q["type"] == "noul":
        target = [gold["probabilities"]["false"], gold["probabilities"]["true"]]
    else:
        target = [gold["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    s = sum(target)
    target = [t / s for t in target]
    seq, markers = build_sequence(tok, case["state"], iq, cfg["max_len"], cfg["head_max_len"],
                                  truncate_left=isinstance(case["state"], list))
    if len(markers) != len(render_options(iq)):
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": target,
            "label": max(range(len(target)), key=target.__getitem__), "workflow": case["workflow"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--only-nc-free", action="store_true")
    args = ap.parse_args()

    d = base_dir(args.base)
    with open(os.path.join(d, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = _load_tokenizer(os.path.join(d, "tokenizer"), cfg)
    suffix = "_ncfree" if args.only_nc_free else ""

    for split in ("train", "calib"):
        items, dropped, lens = [], Counter(), []
        for case in read_jsonl(BUILD / f"{split}.jsonl"):
            if args.only_nc_free and case["meta"]["nc"]:
                continue
            it = to_item(tok, cfg, case)
            if it is None:
                dropped[case["workflow"]] += 1
                continue
            items.append(it)
            lens.append(len(it["ids"]))
        torch.save(items, BUILD / f"{split}_items{suffix}.pt")
        wf = Counter(it["workflow"] for it in items)
        lens.sort()
        print(f"{split}{suffix}: {len(items)} items {dict(wf)} · dropped (options overflow) {dict(dropped)} · "
              f"tokens p50={lens[len(lens)//2] if lens else 0} max={lens[-1] if lens else 0}")


if __name__ == "__main__":
    main()
