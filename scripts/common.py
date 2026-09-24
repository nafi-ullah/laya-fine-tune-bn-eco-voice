"""Shared paths, config loading and text helpers for the Laya fine-tune pipeline."""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]            # repo root
CONFIG = ROOT / "config"
SCHEMAS = ROOT / "schemas"
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
BUILD = DATA / "build"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"

INTENTS = [
    "greeting", "goodbye", "thanks", "product_search", "product_availability", "price_inquiry",
    "order_status", "order_cancel", "return_refund", "shipping_delivery", "payment_issue",
    "discount_offer", "complaint", "agent_request", "out_of_scope",
]  # ClassLabel order of Badhon/BanglaEComIntent (verified against the dataset info)


def load_cfg(name: str) -> dict:
    with open(CONFIG / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load ./.env (HF_TOKEN, GEMINI_API_KEY, VOICE_AGENT_BACKEND) without overriding the real environment."""
    env = ROOT / ".env"
    if env.exists():
        from dotenv import load_dotenv
        load_dotenv(env, override=False)


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


_PUNCT = re.compile(r"[\s।॥.,!?;:'\"()\[\]{}\-–—…।]+")


def norm_text(s: str) -> str:
    """Normalization for exact-duplicate / leak detection (not for training text)."""
    s = unicodedata.normalize("NFC", s).lower()
    return _PUNCT.sub(" ", s).strip()


def frozen_eval_raw(split_cfg: dict) -> set[str]:
    """Every utterance in the frozen hand-written eval suites (as written)."""
    out: set[str] = set()
    for rel in split_cfg["split"]["frozen_eval"]:
        with open(ROOT / rel, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for item in (data.get("labels") or []) + (data.get("cases") or []):
            out.add(item["utterance"])
            if item.get("bn"):
                out.add(item["bn"])
    return out


def frozen_eval_texts(split_cfg: dict) -> set[str]:
    """Every utterance in the frozen hand-written eval suites (normalized)."""
    return {norm_text(t) for t in frozen_eval_raw(split_cfg)}


def script_of(text: str) -> str:
    """Rough script tag matching the dataset's: bn (Bangla), en (Latin), mx (both)."""
    bn = any("ঀ" <= c <= "৿" for c in text)
    lat = any("a" <= c.lower() <= "z" for c in text)
    return "mx" if bn and lat else ("bn" if bn else "en")
