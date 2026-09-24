"""Step 1 — pull Badhon/BanglaEComIntent (gated) into data/raw/*.parquet.

    .venv/bin/python scripts/01_pull_dataset.py

Needs HF_TOKEN (read from ./.env). The dataset's own test split is kept
untouched and becomes the intent benchmark.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from common import INTENTS, RAW, load_cfg, load_env


def main() -> None:
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (expected in ./.env)")

    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    # The repo's CSVs store intents as strings while its card declares a ClassLabel, so
    # `datasets.load_dataset` fails to cast them; read the CSVs directly instead.
    repo = load_cfg("split")["dataset"]["repo"]
    info = HfApi().dataset_info(repo, token=token)
    RAW.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, fname in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        path = hf_hub_download(repo, fname, repo_type="dataset", revision=info.sha, token=token)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        unknown = set(df["intent"]) - set(INTENTS)
        assert not unknown, f"unknown intents in {fname}: {unknown}"
        df.insert(0, "row_id", [f"{split}-{i}" for i in range(len(df))])
        df.to_parquet(RAW / f"{split}.parquet", index=False)
        counts[split] = len(df)

    (RAW / "SOURCE.json").write_text(json.dumps({
        "repo": repo,
        "revision": info.sha,
        "license": (info.card_data or {}).get("license"),
        "splits": counts,
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"pulled {repo}@{info.sha[:8]} → {counts}")


if __name__ == "__main__":
    main()
