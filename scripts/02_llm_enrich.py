"""Step 2 — enrich dataset rows with the LLM (Gemini; batched, parallel, JSON-validated, resumable).

Passes (prompts live in config/llm.yaml):
  translit  Banglish / mixed rows → Bangla script (extra training rows, same label)
  label     every row → llm_intent, alt_intent, ambiguity, escalate. The LLM does NOT see
            the dataset label (v1 did and just copied it); the gold intent always stays the
            dataset label, and an LLM top guess that differs becomes the alt + a review flag.

    .venv/bin/python scripts/02_llm_enrich.py --pass all
    .venv/bin/python scripts/02_llm_enrich.py --pass label --limit 200    # smoke test

Results are appended to data/cache/<pass>.jsonl keyed by sha1(pass, prompt_version, text,
intent), so a crash or restart only processes what is missing. Items that stay invalid after
the configured retries go to data/cache/rejects.jsonl and never reach training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import jsonschema
import pandas as pd

from common import CACHE, INTENTS, RAW, REPORTS, SCHEMAS, append_jsonl, load_cfg, load_env, read_jsonl
from llm import GeminiLLM, extract_json

SPLITS = ("train", "validation", "test")


def item_key(pass_name: str, version: int, text: str, intent: str) -> str:
    return hashlib.sha1(f"{pass_name}|{version}|{text}|{intent}".encode()).hexdigest()


def load_rows(limit: int | None) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(RAW / f"{s}.parquet") for s in SPLITS], ignore_index=True)
    if limit:  # stratified sample so a smoke run sees every intent
        df = df.groupby("intent").sample(n=max(1, limit // len(INTENTS)), random_state=0)
    return df.reset_index(drop=True)


def validate(pass_name: str, obj, row: dict, validator):
    """Return (normalized dict, None) or (None, error string)."""
    try:
        validator.validate(obj)
    except jsonschema.ValidationError as e:
        return None, f"schema: {e.message}"
    if pass_name == "translit":
        text = obj[1].strip()
        if not any("ঀ" <= c <= "৿" for c in text):
            return None, "no Bangla characters in output"
        return {"bn_text": text}, None
    _, top, alt, esc = obj
    if top not in INTENTS:
        return None, f"intent not in list: {top}"
    alt = alt if alt in INTENTS else None
    gold = row["intent"]
    # Soft-target alternative relative to the dataset's gold label.
    alt_for_gold = top if top != gold else (alt if alt != gold else None)
    return {"llm_intent": top, "llm_alt": alt, "alt_intent": alt_for_gold,
            "ambiguity": "high" if alt_for_gold else "low", "escalate": esc}, None


def reply_id(obj):
    if isinstance(obj, list) and obj:
        return obj[0]
    if isinstance(obj, dict):
        return obj.get("id")
    return None


def build_user(pcfg: dict, batch: list[dict]) -> str:
    payload = [{"id": i + 1, "text": r["text"]} for i, r in enumerate(batch)]
    return (f"Example input:\n{pcfg['example_in']}\nExample output:\n{pcfg['example_out']}\n\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=False)}")


def call_batch(llm: GeminiLLM, system: str, pcfg: dict, batch: list[dict], max_tokens: int) -> dict:
    try:
        parsed = extract_json(llm.chat(system, build_user(pcfg, batch), pcfg["temperature"], max_tokens))
        return {reply_id(o): o for o in parsed} if isinstance(parsed, list) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def run_pass(llm: GeminiLLM, pass_name: str, df: pd.DataFrame, cfg: dict) -> None:
    pcfg = cfg["passes"][pass_name]
    version = pcfg["prompt_version"]
    system = pcfg["system"].replace("{intents}", ", ".join(INTENTS))
    per_item = cfg["max_tokens_per_item"] if pass_name == "translit" else 25
    schema = json.loads((SCHEMAS / "llm_row.schema.json").read_text())
    validator = jsonschema.Draft202012Validator({**schema["$defs"][pass_name], "$defs": schema["$defs"]})
    out_path = CACHE / f"{pass_name}.jsonl"

    rows = df[df["script"].isin(["bl", "mx"])] if pass_name == "translit" else df
    done = {r["key"] for r in read_jsonl(out_path)}
    todo = []
    for r in rows.to_dict("records"):
        r["key"] = item_key(pass_name, version, r["text"], r["intent"])
        if r["key"] not in done:
            todo.append(r)
    print(f"[{pass_name}] {len(rows)} rows, {len(rows) - len(todo)} cached, {len(todo)} to do", flush=True)

    bs = pcfg["batch_size"]
    attempts = {r["key"]: 0 for r in todo}
    queue = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    n_ok = n_rej = 0
    t0 = time.perf_counter()
    last_print = 0
    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        running = {}
        while queue or running:
            while queue and len(running) < cfg["concurrency"]:
                batch = queue.pop(0)
                fut = pool.submit(call_batch, llm, system, pcfg, batch, per_item * len(batch) + 40)
                running[fut] = batch
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            retry = []
            for fut in finished:
                batch = running.pop(fut)
                by_id = fut.result()
                for i, r in enumerate(batch):
                    obj = by_id.get(i + 1)
                    res, err = (None, "missing from reply") if obj is None else validate(pass_name, obj, r, validator)
                    if err is None:
                        append_jsonl(out_path, {"key": r["key"], "row_id": r["row_id"], "text": r["text"],
                                                "intent": r["intent"], "script": r["script"], **res})
                        n_ok += 1
                        continue
                    attempts[r["key"]] += 1
                    if attempts[r["key"]] <= cfg["retries"]:
                        retry.append(r)
                    else:
                        append_jsonl(CACHE / "rejects.jsonl", {"pass": pass_name, "row_id": r["row_id"],
                                                               "text": r["text"], "error": err})
                        n_rej += 1
            # retried items go in small batches so one bad sentence can't sink a big batch again
            queue += [retry[i:i + 5] for i in range(0, len(retry), 5)]
            if n_ok + n_rej - last_print >= 500 or (not queue and not running):
                last_print = n_ok + n_rej
                el = time.perf_counter() - t0
                eta = el / max(1, last_print) * (len(todo) - last_print)
                print(f"  {last_print}/{len(todo)} ok={n_ok} rejected={n_rej} · {llm.usage()} "
                      f"· eta {eta/60:.1f} min", flush=True)
    print(f"[{pass_name}] done: ok={n_ok} rejected={n_rej}", flush=True)


def disagreement_report() -> None:
    rows = list(read_jsonl(CACHE / "label.jsonl"))
    dis = [r for r in rows if r["llm_intent"] != r["intent"]]
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "label_disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "text", "dataset_intent", "llm_intent", "llm_alt", "script"])
        for r in dis:
            w.writerow([r["row_id"], r["text"], r["intent"], r["llm_intent"], r["llm_alt"], r["script"]])
    print(f"label disagreements: {len(dis)}/{len(rows)} → reports/label_disagreements.csv "
          f"(review, not auto-relabelled)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_name", choices=["translit", "label", "all"], default="all")
    ap.add_argument("--limit", type=int, help="stratified sample size (smoke test)")
    args = ap.parse_args()

    load_env()
    cfg = load_cfg("llm")
    df = load_rows(args.limit)
    llm = GeminiLLM(cfg)
    for p in (["translit", "label"] if args.pass_name == "all" else [args.pass_name]):
        run_pass(llm, p, df, cfg)
    if args.pass_name in ("label", "all"):
        disagreement_report()
    print(f"usage: {llm.usage()} ({cfg['model']})")


if __name__ == "__main__":
    main()
