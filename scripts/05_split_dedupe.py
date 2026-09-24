"""Step 5 — leak-free splits: drop train cases that duplicate any test / frozen-eval text,
carve a calibration slice, apply class weights, write the data card.

    .venv/bin/python scripts/05_split_dedupe.py

Outputs: data/build/{train,calib}.jsonl, data/build/test_{order_confirm,intent,escalate}.jsonl,
         reports/data_card.md. Exits non-zero if any exact leak survives.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from common import BUILD, REPORTS, frozen_eval_raw, frozen_eval_texts, load_cfg, norm_text, read_jsonl, write_jsonl


def embed(model, texts: list[str]) -> np.ndarray:
    # e5 models expect a "query: " prefix for symmetric similarity.
    return model.encode([f"query: {t}" for t in texts], batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


_MODELS: dict = {}


def near_dup_mask(train_texts: list[str], ref_texts: list[str], thr: float, model_id: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if model_id not in _MODELS:
        _MODELS[model_id] = SentenceTransformer(model_id)
    model = _MODELS[model_id]
    uniq = sorted(set(train_texts))
    A, B = embed(model, uniq), embed(model, ref_texts)
    best = np.zeros(len(uniq), dtype=np.float32)
    for i in range(0, len(uniq), 2048):
        best[i:i + 2048] = (A[i:i + 2048] @ B.T).max(axis=1)
    hit = {t for t, s in zip(uniq, best) if s >= thr}
    return np.array([t in hit for t in train_texts])


def main() -> None:
    cfg = load_cfg("split")
    rng = random.Random(cfg["seed"])
    train = list(read_jsonl(BUILD / "cases_train.jsonl"))
    test = list(read_jsonl(BUILD / "cases_test.jsonl"))
    frozen = frozen_eval_texts(cfg)
    ref_norm = {norm_text(c["meta"]["text"]) for c in test} | frozen

    # 1) exact duplicates of test / frozen eval text
    n0 = len(train)
    exact_drop = Counter(c["workflow"] for c in train if norm_text(c["meta"]["text"]) in ref_norm)
    train = [c for c in train if norm_text(c["meta"]["text"]) not in ref_norm]

    # 2) near duplicates (multilingual embeddings); stricter against the frozen benchmark
    texts = [c["meta"]["text"] for c in train]
    frozen_raw = sorted(frozen_eval_raw(cfg))
    mask = (near_dup_mask(texts, frozen_raw, cfg["split"]["near_dup_cosine_frozen"], cfg["split"]["embedder"])
            | near_dup_mask(texts, sorted({c["meta"]["text"] for c in test}),
                            cfg["split"]["near_dup_cosine_test"], cfg["split"]["embedder"]))
    near_drop = Counter(c["workflow"] for c, m in zip(train, mask) if m)
    train = [c for c, m in zip(train, mask) if not m]

    # 3) calibration slice, by group so variants of one row stay on one side
    groups = sorted({c["meta"]["group"] for c in train})
    rng.shuffle(groups)
    calib_groups = set(groups[:round(len(groups) * cfg["split"]["calib_fraction"])])
    calib = [c for c in train if c["meta"]["group"] in calib_groups]
    train = [c for c in train if c["meta"]["group"] not in calib_groups]

    # 4) class weights (repeat whole cases; train only)
    weights = cfg["class_weights"]
    train = [c for c in train for _ in range(weights.get(c["workflow"], 1))]
    rng.shuffle(train)

    # leak assertion
    leaks = sum(1 for c in train + calib if norm_text(c["meta"]["text"]) in ref_norm)

    write_jsonl(BUILD / "train.jsonl", train)
    write_jsonl(BUILD / "calib.jsonl", calib)
    by_wf = defaultdict(list)
    for c in test:
        by_wf[c["workflow"]].append(c)
    for wf, cases in by_wf.items():
        write_jsonl(BUILD / f"test_{wf}.jsonl", cases)

    # data card
    def table(cases: list[dict], title: str) -> list[str]:
        cnt = Counter((c["workflow"], c["gold"][next(iter(c["gold"]))]["label"]) for c in cases)
        scr = Counter((c["workflow"], c["meta"]["script"]) for c in cases)
        src = Counter((c["workflow"], c["meta"]["source"]) for c in cases)
        out = [f"### {title} — {len(cases)} cases", "", "| workflow | label | n |", "|---|---|---:|"]
        out += [f"| {w} | {l} | {n} |" for (w, l), n in sorted(cnt.items())]
        out += ["", "| workflow | script | n |", "|---|---|---:|"]
        out += [f"| {w} | {s} | {n} |" for (w, s), n in sorted(scr.items())]
        out += ["", "| workflow | source | n |", "|---|---|---:|"]
        out += [f"| {w} | {s} | {n} |" for (w, s), n in sorted(src.items())]
        return out + [""]

    meta = (BUILD / "build_meta.json").read_text() if (BUILD / "build_meta.json").exists() else "{}"
    lines = ["# Data card — Laya fine-tune", "",
             f"- build: `{meta.strip()}`",
             f"- train pool before dedupe: {n0}",
             f"- dropped exact duplicates of test/frozen eval: {dict(exact_drop)}",
             f"- dropped near-duplicates (`{cfg['split']['embedder']}`; cos ≥ "
             f"{cfg['split']['near_dup_cosine_frozen']} vs frozen eval, ≥ {cfg['split']['near_dup_cosine_test']} "
             f"vs test splits): {dict(near_drop)}",
             f"- class weights applied to train: {weights}",
             f"- **exact leaks remaining: {leaks}** (must be 0)", ""]
    lines += table(train, "train (after weights)") + table(calib, "calib") + table(test, "test")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_card.md").write_text("\n".join(lines), encoding="utf-8")

    empty = [l for l in load_cfg("decisions")["order_confirm"]["criteria"]
             if not any(c["workflow"] == "order_confirm" and c["gold"]["order_confirm"]["label"] == l for c in train)]
    print(f"train={len(train)} calib={len(calib)} test={len(test)} · exact drop {dict(exact_drop)} · "
          f"near drop {dict(near_drop)} · leaks={leaks}" + (f" · EMPTY confirm classes: {empty}" if empty else ""))
    if leaks:
        raise SystemExit("leak check failed")


if __name__ == "__main__":
    main()
