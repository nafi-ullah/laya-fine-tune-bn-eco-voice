"""Laya zero-shot vs Laya fine-tuned (R1) vs Jev, in Bangla and English → reports/JEV_COMPARISON.md

    .venv/bin/python scripts/12_jev_compare.py            (needs OPENROUTER_API_KEY, GEMINI_API_KEY once)

Jev (typesafe/jev-1.13, OpenRouter Decisions API) takes the same typed questions as Laya, so every
model gets byte-identical state + questions (the canonical ones from config/decisions.yaml).

Test sets, paired by language:
  confirm_frozen   bn: hybrid labeled + traps (23)      en: their one-to-one translation (order_confirm_en.yaml)
  confirm_test     bn: generated test split (186)       en: Gemini translation of the same rows (cached)
  intent / escalate  BanglaEComIntent test split: bn = Bangla script (+ transliterated),
                   en = English rows, banglish = romanized + mixed rows
  router           hand-written router suites (bn / en)
Jev answers are cached in data/cache/jev.jsonl, so re-runs cost nothing.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import httpx
import yaml

from common import BUILD, CACHE, REPORTS, ROOT, append_jsonl, load_cfg, load_env, read_jsonl, write_jsonl

bench = importlib.import_module("08_benchmark")
CONFIRM = bench.CONFIRM


# ------------------------------------------------------------------ Jev client

class JevAgent:
    """laya_mlx-compatible .predict(state, questions) backed by the OpenRouter Decisions API."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise SystemExit("OPENROUTER_API_KEY not set (expected in ./.env)")
        self.client = httpx.Client(timeout=60)
        self.cache_path = CACHE / "jev.jsonl"
        self.cache = {r["key"]: r["resp"] for r in read_jsonl(self.cache_path)} if self.cache_path.exists() else {}
        self.lock = threading.Lock()
        self.cost = 0.0
        self.calls = 0

    def _key(self, state, questions) -> str:
        blob = json.dumps([self.cfg["model"], state, questions], ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def _call(self, state, questions) -> dict:
        body = {"model": self.cfg["model"], "state": state, "questions": questions}
        delay = 2.0
        for _ in range(self.cfg["http_attempts"]):
            try:
                r = self.client.post(self.cfg["url"], json=body, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.TransportError:
                r = None
            if r is not None and r.status_code == 200:
                d = r.json()
                with self.lock:
                    self.calls += 1
                    self.cost += d.get("usage", {}).get("cost", 0.0)
                return d
            if r is not None and r.status_code not in (429, 500, 502, 503, 504, 524, 529):
                raise RuntimeError(f"Jev {r.status_code}: {r.text[:300]}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError("Jev: retries exhausted")

    def predict(self, state, questions) -> dict:
        k = self._key(state, questions)
        if k not in self.cache:
            d = self._call(state, questions)
            with self.lock:
                self.cache[k] = d
                append_jsonl(self.cache_path, {"key": k, "resp": d})
        return self.cache[k]

    def prefetch(self, items: list[dict]) -> None:
        todo = {self._key(it["state"], it["questions"]): it for it in items}
        todo = [it for k, it in todo.items() if k not in self.cache]
        if not todo:
            return
        print(f"jev: {len(todo)} uncached requests …", flush=True)
        with ThreadPoolExecutor(self.cfg["concurrency"]) as ex:
            list(ex.map(lambda it: self.predict(it["state"], it["questions"]), todo))

    def latency(self, items: list[dict], n: int) -> dict:
        ts = []
        for it in (items * (n // len(items) + 1))[:n]:            # uncached: real network round trips
            t0 = time.perf_counter()
            self._call(it["state"], it["questions"])
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return {"p50": ts[len(ts) // 2], "p95": ts[int(len(ts) * 0.95) - 1]}


# ------------------------------------------------------------------ English confirm_test (translated once)

def translated_confirm_test(tcfg: dict) -> dict[str, str]:
    out_path = BUILD / "test_order_confirm_en.jsonl"
    src = [c for c in read_jsonl(BUILD / "test_order_confirm.jsonl")]
    done = {r["id"]: r["en"] for r in read_jsonl(out_path)} if out_path.exists() else {}
    todo = [c for c in src if c["id"] not in done]
    if todo:
        from llm import GeminiLLM, extract_json
        llm = GeminiLLM(load_cfg("llm"))
        user_head = f"Example input: {tcfg['example_in']}\nExample output: {tcfg['example_out']}\n\nInput: "
        for attempt in range(3):
            batches = [todo[i:i + tcfg["batch_size"]] for i in range(0, len(todo), tcfg["batch_size"])]

            def run(batch):
                payload = json.dumps([{"id": i, "text": c["meta"]["text"]} for i, c in enumerate(batch)],
                                     ensure_ascii=False)
                try:
                    rows = extract_json(llm.chat(tcfg["system"], user_head + payload, tcfg["temperature"],
                                                 60 * len(batch)))
                except Exception as e:                      # bad JSON → items retried next attempt
                    print(f"translate batch failed: {e}")
                    return {}
                return {batch[i]["id"]: t for i, t in rows if isinstance(i, int) and 0 <= i < len(batch) and t}

            with ThreadPoolExecutor(8) as ex:
                for res in ex.map(run, batches):
                    done.update(res)
            todo = [c for c in src if c["id"] not in done]
            if not todo:
                break
        print(f"translate: {llm.usage()}")
        write_jsonl(out_path, [{"id": c["id"], "bn": c["meta"]["text"], "en": done[c["id"]]}
                               for c in src if c["id"] in done])
    return done


# ------------------------------------------------------------------ paired test sets

def build_sets(decisions: dict, bcfg: dict, jcfg: dict) -> dict[str, list[dict]]:
    base = bench.load_sets(decisions, bcfg)
    oc_q = bench.canonical_question(decisions, "order_confirm")
    agent_en = jcfg["agent_turn_en"]

    def conv_en(text):
        return [{"role": "assistant", "content": agent_en}, {"role": "user", "content": text}]

    def oc(text, label, lang):
        return {"text": text, "state": conv_en(text), "questions": oc_q, "qname": "order_confirm",
                "gold": {label}, "label": label, "script": lang}

    y = yaml.safe_load((ROOT / "zero_shot/cases/order_confirm_en.yaml").read_text(encoding="utf-8"))
    en_frozen = [oc(x["utterance"], x["expect"]["intent"], "en") for x in y["cases"]]

    en_map = translated_confirm_test(jcfg["translate"])
    by_text = {c["meta"]["text"]: c["id"] for c in read_jsonl(BUILD / "test_order_confirm.jsonl")}
    en_test = [oc(en_map[by_text[r["text"]]], r["label"], "en") for r in base["confirm_test"]
               if by_text.get(r["text"]) in en_map]

    def by_script(rows, scripts):
        return [r for r in rows if r["script"] in scripts]

    BN, EN, BL = ("bn", "bn_translit"), ("en",), ("bl", "mx")
    return {
        "confirm_frozen/bn": base["confirm_frozen"], "confirm_frozen/en": en_frozen,
        "confirm_test/bn": base["confirm_test"], "confirm_test/en": en_test,
        "intent/bn": by_script(base["intent_test"], BN), "intent/en": by_script(base["intent_test"], EN),
        "intent/banglish": by_script(base["intent_test"], BL),
        "escalate/bn": by_script(base["escalate_test"], BN), "escalate/en": by_script(base["escalate_test"], EN),
        "escalate/banglish": by_script(base["escalate_test"], BL),
        "router/bn": base["router_bn"], "router/en": base["router_en"],
    }


# ------------------------------------------------------------------ main

def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> int:
    load_env()
    jcfg, bcfg, decisions = load_cfg("jev"), load_cfg("benchmark"), load_cfg("decisions")
    sets = build_sets(decisions, bcfg, jcfg)
    all_items = [it for items in sets.values() for it in items]
    thr = bcfg["confidence_threshold"]
    lat_mix = sets["confirm_test/bn"][:20] + sets["confirm_test/en"][:20] + sets["intent/bn"][:20] + sets["intent/en"][:20]

    results, lat, info = {}, {}, {}

    import laya_mlx
    import mlx.core as mx
    for label, mid in jcfg["laya"].items():
        path = ROOT / mid
        mx.reset_peak_memory()
        agent = laya_mlx.load(str(path) if path.exists() else mid)
        results[label] = {s: bench.run_model(agent, items) for s, items in sets.items()}
        lat[label] = bench.latency(agent, lat_mix, bcfg["latency"]["warmup"], bcfg["latency"]["n_calls"])
        info[label] = {"where": "local, M5 MacBook (MLX FP16)", "memory": f"{mx.get_peak_memory() / 2**20:.0f} MiB",
                       "cost": "$0 (runs locally)"}
        print(f"{label}: done · P50 {lat[label]['p50']:.1f} ms", flush=True)
        del agent

    jev = JevAgent(jcfg["jev"])
    jev.prefetch(all_items)
    results["jev"] = {s: bench.run_model(jev, items) for s, items in sets.items()}
    snapshot = next(iter(jev.cache.values()))["model"]
    run_cost = sum(d.get("usage", {}).get("cost", 0.0) for d in jev.cache.values())
    lat["jev"] = jev.latency(lat_mix, jcfg["jev"]["latency_calls"])
    per_1k = 1000 * run_cost / len(jev.cache)
    info["jev"] = {"where": f"cloud API (OpenRouter → TypeSafe), `{snapshot}`", "memory": "—",
                   "cost": f"${run_cost:.3f} for {len(jev.cache):,} decisions (≈ ${per_1k:.4f} / 1k)"}
    print(f"jev: done · P50 {lat['jev']['p50']:.0f} ms · ${run_cost:.4f} total", flush=True)

    models = list(results)
    M = {m: {s: bench.metrics(results[m][s], thr) for s in sets} for m in models}
    lk = {m: {s: len(bench.leaks(results[m][s])) for s in sets if s.startswith("confirm")} for m in models}
    names = {"laya_zero_shot": "Laya zero-shot", "laya_finetuned": "Laya fine-tuned (R1)", "jev": "Jev 1.13"}

    def best(s, key="acc"):
        return max(models, key=lambda m: M[m][s][key])

    def cell(m, s):
        v = pct(M[m][s]["acc"])
        return f"**{v}**" if m == best(s) else v

    L = ["# Laya (zero-shot, fine-tuned) vs Jev — Bangla and English", "",
         f"- date: {datetime.now():%Y-%m-%d %H:%M}",
         "- **Laya zero-shot** = `aac6fef/laya-multilingual-mlx` (322M, open, MLX on the M5).",
         "- **Laya fine-tuned (R1)** = `checkpoints/R1-mlx`, this repo's fine-tune "
         "([HF](https://huggingface.co/nafiullah/laya-multilingual-bn-ecom-voice)).",
         f"- **Jev 1.13** = TypeSafe's closed decision model through OpenRouter's Decisions API (`{snapshot}`). "
         "Same choice / noul question format as Laya.",
         "- All three models get **byte-identical** state and questions: the canonical wording from "
         "`config/decisions.yaml`. Order confirm state = [agent question, customer reply]; the English sets use "
         f"the agent turn \"{jcfg['agent_turn_en']}\".",
         "- Bold = best in row. Fine-tuned Laya has seen the *training* splits of these task families; "
         "zero-shot Laya and Jev have not. The test rows are never trained on.", ""]
    notes = REPORTS / "JEV_TAKEAWAYS.md"                        # hand-written reading of the numbers
    if notes.exists():
        L += [notes.read_text(encoding="utf-8").strip(), ""]

    L += ["## Headline — accuracy", "", "| test set | lang | n | " + " | ".join(names[m] for m in models) + " |",
          "|---|---|---:|" + "---:|" * len(models)]
    for s in sets:
        fam, lang = s.split("/")
        L.append(f"| {fam} | {lang} | {len(sets[s])} | " + " | ".join(cell(m, s) for m in models) + " |")
    L.append("")

    L += ["## yes↔no mix-ups on order confirm (the dangerous error — gate is 0)", "",
          "| test set | lang | n | " + " | ".join(names[m] for m in models) + " |",
          "|---|---|---:|" + "---:|" * len(models)]
    for s in [s for s in sets if s.startswith("confirm")]:
        fam, lang = s.split("/")
        L.append(f"| {fam} | {lang} | {len(sets[s])} | " + " | ".join(str(lk[m][s]) for m in models) + " |")
    L.append("")

    # language summary: micro-average over the paired families
    L += ["## By language (all decisions pooled)", "", "| language | n | " + " | ".join(names[m] for m in models) + " |",
          "|---|---:|" + "---:|" * len(models)]
    for lang in ("bn", "en", "banglish"):
        ss = [s for s in sets if s.endswith("/" + lang)]
        n = sum(len(sets[s]) for s in ss)
        cells = [pct(sum(r["ok"] for s in ss for r in results[m][s]) / n) for m in models]
        L.append(f"| {lang} | {n} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Speed, cost, deployment", "", "| | " + " | ".join(names[m] for m in models) + " |",
          "|---|" + "---:|" * len(models),
          "| latency P50 / P95 per decision | " + " | ".join(f"{lat[m]['p50']:.1f} / {lat[m]['p95']:.1f} ms" for m in models) + " |",
          "| runs on | " + " | ".join(info[m]["where"] for m in models) + " |",
          "| peak memory | " + " | ".join(info[m]["memory"] for m in models) + " |",
          "| cost of this benchmark | " + " | ".join(info[m]["cost"] for m in models) + " |",
          "| weights | open (Apache-2.0 base) | open (CC BY-NC-SA 4.0) | closed |", "",
          "Jev latency is a full HTTPS round trip from this Mac to OpenRouter, measured on "
          f"{jcfg['jev']['latency_calls']} sequential uncached calls; Laya latency is in-process on the Mac.", ""]

    L += [f"## Calibration (confidence ≥ {thr})", "",
          "`cov` = share of items whose top probability ≥ threshold, `acc` = accuracy on those items.", "",
          "| test set | " + " | ".join(f"{names[m]} cov / acc" for m in models) + " |", "|---|" + "---:|" * len(models)]
    for s in sets:
        L.append(f"| {s} | " + " | ".join(f"{pct(M[m][s]['cov_hi'])} / {pct(M[m][s]['acc_hi'])}" for m in models) + " |")
    L.append("")

    for s in ("confirm_frozen/bn", "confirm_frozen/en", "confirm_test/bn", "confirm_test/en"):
        L += [f"## Confusion — {s} (rows = expected, cols = predicted)", ""]
        for m in models:
            cm = Counter((r["label"], r["pred"]) for r in results[m][s])
            L += [f"**{names[m]}**", "", "| exp \\ pred | " + " | ".join(CONFIRM) + " |", "|---|" + "---:|" * 4]
            L += [f"| {e} | " + " | ".join(str(cm[(e, p)]) for p in CONFIRM) + " |" for e in CONFIRM]
            L.append("")

    L += ["## Frozen order-confirm set, reply by reply", "",
          "| Bangla reply | English reply | expected | " + " | ".join(f"{names[m]} bn / en" for m in models) + " |",
          "|---|---|---|" + "---|" * len(models)]
    for rb, re_ in zip(sets["confirm_frozen/bn"], sets["confirm_frozen/en"]):
        cells = []
        for m in models:
            pb = next(r for r in results[m]["confirm_frozen/bn"] if r["text"] == rb["text"])
            pe = next(r for r in results[m]["confirm_frozen/en"] if r["text"] == re_["text"])
            cells.append(f"{'✅' if pb['ok'] else '❌ ' + pb['pred']} / {'✅' if pe['ok'] else '❌ ' + pe['pred']}")
        L.append(f"| {rb['text']} | {re_['text']} | {rb['label']} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Escalate errors by direction", "",
          "missed = should go to a human but was kept by the bot; extra = sent to a human without need.", "",
          "| test set | " + " | ".join(f"{names[m]} missed / extra" for m in models) + " |", "|---|" + "---:|" * len(models)]
    for s in [s for s in sets if s.startswith("escalate") or s.startswith("router")]:
        cells = []
        for m in models:
            rs = [r for r in results[m][s] if r["qname"] == "escalate"]
            cells.append(f"{sum(r['label'] and not r['pred'] for r in rs)} / {sum(r['pred'] and not r['label'] for r in rs)}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")

    for m in models:
        errs = Counter((r["label"], r["pred"]) for s in ("intent/bn", "intent/en") for r in results[m][s] if not r["ok"])
        L += [f"## {names[m]} — most common intent confusions (bn + en)", "", "| expected | predicted | n |", "|---|---|---:|"]
        L += [f"| {a} | {b} | {n} |" for (a, b), n in errs.most_common(8)]
        L.append("")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "JEV_COMPARISON.md").write_text("\n".join(L), encoding="utf-8")
    (REPORTS / "JEV_COMPARISON.json").write_text(json.dumps(
        {"metrics": M, "latency": lat, "info": info, "yes_no_leaks": lk,
         "leaks": {m: {s: [r["text"] for r in bench.leaks(results[m][s])] for s in lk[m]} for m in models}},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for s in sets:
        print(f"{s:20s} " + "  ".join(f"{m} {pct(M[m][s]['acc'])}" for m in models))
    print("→ reports/JEV_COMPARISON.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
