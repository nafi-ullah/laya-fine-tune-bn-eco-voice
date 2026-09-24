"""Step 9 — benchmark base vs fine-tuned Laya (MLX) vs the hybrid cascade → reports/FINETUNE_RESULTS.md

    .venv/bin/python scripts/08_benchmark.py

Test sets (all scored with the canonical production question format from config/decisions.yaml):
  confirm_frozen   hybrid labeled.yaml (18) + our 5 order_confirm traps — frozen, never trained on
  confirm_test     generated order-confirm test split (split by generation group) + real out_of_scope rows
  intent_test      BanglaEComIntent test split (+ its Bangla-script transliterations), by script
  escalate_test    same rows, escalate decision (labels: agent_request prior + Gemini)
  router_bn/en     our hand-written router suites (coarse labels mapped via config/benchmark.yaml)

Exit code 1 if the R1 model has any yes↔no mix-up (same convention as run_laya.py).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from common import BUILD, REPORTS, ROOT, load_cfg, load_env, read_jsonl, write_jsonl

CONFIRM = ["yes", "no", "repeat", "out_of_scope"]


# ------------------------------------------------------------------ test sets

def canonical_question(decisions: dict, name: str) -> dict:
    spec = decisions[name]
    return {name: {"type": spec["type"], "instructions": spec["instructions"][0], "criteria": dict(spec["criteria"])}}


def load_sets(decisions: dict, bcfg: dict) -> dict[str, list[dict]]:
    """Each item: {text, state, questions, qname, gold (label or set), script}."""
    agent = decisions["order_confirm"]["agent_turns"][0]
    oc_q = canonical_question(decisions, "order_confirm")
    sets: dict[str, list[dict]] = {}

    def conv(text):
        return [{"role": "assistant", "content": agent}, {"role": "user", "content": text}]

    frozen = []
    y = yaml.safe_load((ROOT / "eval_sets/hybrid_order_confirm_labeled.yaml").read_text(encoding="utf-8"))
    frozen += [(x["utterance"], x["intent"]) for x in y["labels"]]
    y = yaml.safe_load((ROOT / "zero_shot/cases/order_confirm.yaml").read_text(encoding="utf-8"))
    frozen += [(x["utterance"], x["expect"]["intent"]) for x in y.get("cases") or []]
    sets["confirm_frozen"] = [{"text": t, "state": conv(t), "questions": oc_q, "qname": "order_confirm",
                               "gold": {g}, "label": g, "script": "bn"} for t, g in frozen]

    def from_build(wf):
        out = []
        for c in read_jsonl(BUILD / f"test_{wf}.jsonl"):
            q = next(iter(c["questions"]))
            g = c["gold"][q]["label"]
            g = (g == "true") if c["questions"][q]["type"] == "noul" else g
            out.append({"text": c["meta"]["text"], "state": c["state"], "questions": c["questions"], "qname": q,
                        "gold": {g}, "label": g, "script": c["meta"]["script"]})
        return out

    sets["confirm_test"] = from_build("order_confirm")
    sets["intent_test"] = from_build("intent")
    sets["escalate_test"] = from_build("escalate")

    imap = bcfg["router_intent_map"]
    int_q, esc_q = canonical_question(decisions, "intent"), canonical_question(decisions, "escalate")
    for lang, fname in (("bn", "ecommerce_router.yaml"), ("en", "ecommerce_router_en.yaml")):
        y = yaml.safe_load((ROOT / "zero_shot/cases" / fname).read_text(encoding="utf-8"))
        items = []
        for x in y["cases"]:
            e = x["expect"]
            if "intent" in e:
                items.append({"text": x["utterance"], "state": x["utterance"], "questions": int_q, "qname": "intent",
                              "gold": set(imap[e["intent"]]), "label": e["intent"], "script": lang})
            if "escalate" in e:
                items.append({"text": x["utterance"], "state": x["utterance"], "questions": esc_q, "qname": "escalate",
                              "gold": {bool(e["escalate"])}, "label": bool(e["escalate"]), "script": lang})
        sets[f"router_{lang}"] = items
    return sets


# ------------------------------------------------------------------ scoring

def run_model(agent, items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        a = agent.predict(it["state"], it["questions"])["answers"][it["qname"]]
        if a["type"] == "choice":
            pred, probs = a["choice"], a["probabilities"]
            pmax = max(probs.values())
        else:
            pred, probs = a["noul"] >= 0.5, {"false": 1 - a["noul"], "true": a["noul"]}
            pmax = max(a["noul"], 1 - a["noul"])
        out.append({**it, "pred": pred, "probs": probs, "pmax": pmax, "ok": pred in it["gold"]})
    return out


def metrics(rows: list[dict], thr: float) -> dict:
    n = len(rows)
    acc = sum(r["ok"] for r in rows) / n
    briers, confs, oks = [], [], []
    for r in rows:
        if len(r["gold"]) == 1:
            g = next(iter(r["gold"]))
            key = ("true" if g else "false") if isinstance(g, bool) else g
            briers.append(sum((p - (1.0 if k == key else 0.0)) ** 2 for k, p in r["probs"].items()))
        confs.append(r["pmax"])
        oks.append(r["ok"])
    confs, oks = np.array(confs), np.array(oks, dtype=float)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.any():
            ece += m.mean() * abs(oks[m].mean() - confs[m].mean())
    hi = confs >= thr
    return {"n": n, "acc": acc, "brier": float(np.mean(briers)) if briers else None, "ece": float(ece),
            "cov_hi": float(hi.mean()), "acc_hi": float(oks[hi].mean()) if hi.any() else None}


def leaks(rows: list[dict]) -> list[dict]:
    return [r for r in rows if {r["label"], r["pred"]} == {"yes", "no"}]


def latency(agent, items: list[dict], warmup: int, n: int) -> dict:
    sample = (items * (n // max(1, len(items)) + 1))[:n]
    for it in sample[:warmup]:
        agent.predict(it["state"], it["questions"])
    ts = []
    for it in sample:
        t0 = time.perf_counter()
        agent.predict(it["state"], it["questions"])
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50": ts[len(ts) // 2], "p95": ts[int(len(ts) * 0.95) - 1]}


def run_cascade(bcfg: dict, sets: dict) -> dict[str, list[dict]]:
    backend = os.environ.get("VOICE_AGENT_BACKEND")
    if not backend:
        print("cascade column skipped (set VOICE_AGENT_BACKEND to the voice-agent backend/ dir)")
        return {}
    py = Path(backend) / bcfg["backend_python"]
    if not py.exists():
        print(f"cascade column skipped ({py} not found)")
        return {}
    res = {}
    for name in ("confirm_frozen", "confirm_test"):
        src, dst = BUILD / f"cascade_in_{name}.jsonl", BUILD / f"cascade_out_{name}.jsonl"
        write_jsonl(src, [{"text": r["text"], "label": r["label"]} for r in sets[name]])
        p = subprocess.run([str(py), str(ROOT / "scripts" / "cascade_eval.py"), str(src), str(dst)],
                           cwd=backend, capture_output=True, text=True,
                           env={**os.environ, "VOICE_AGENT_BACKEND": backend})
        if p.returncode != 0:
            print(f"cascade failed on {name}: {p.stderr[-400:]}")
            continue
        res[name] = [{**r, "ok": r["pred"] == r["label"]} for r in read_jsonl(dst)]
    return res


# ------------------------------------------------------------------ report

def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> int:
    load_env()
    bcfg = load_cfg("benchmark")
    decisions = load_cfg("decisions")
    sets = load_sets(decisions, bcfg)
    thr = bcfg["confidence_threshold"]

    import laya_mlx
    import mlx.core as mx

    results, lat, mem = {}, {}, {}
    for label, mid in bcfg["models"].items():
        path = ROOT / mid
        if not path.exists() and "/" in mid and mid.startswith("checkpoints"):
            print(f"skip {label}: {mid} not found")
            continue
        mx.reset_peak_memory()
        agent = laya_mlx.load(str(path) if path.exists() else mid)
        t0 = time.perf_counter()
        results[label] = {name: run_model(agent, items) for name, items in sets.items()}
        mix = sets["confirm_test"][:50] + sets["intent_test"][:50]
        lat[label] = latency(agent, mix, bcfg["latency"]["warmup"], bcfg["latency"]["n_calls"])
        mem[label] = mx.get_peak_memory() / 2**20
        print(f"{label}: scored {sum(len(v) for v in sets.values())} items in {time.perf_counter()-t0:.0f}s · "
              f"P50 {lat[label]['p50']:.1f} ms", flush=True)
        del agent

    cascade = run_cascade(bcfg, sets)
    models = list(results)
    M = {m: {s: metrics(results[m][s], thr) for s in sets} for m in models}

    def bn_intent(m):
        rows = [r for r in results[m]["intent_test"] if r["script"] in ("bn", "bn_translit")]
        return sum(r["ok"] for r in rows) / len(rows)

    # ---------------- markdown
    meta = json.loads((BUILD / "build_meta.json").read_text()) if (BUILD / "build_meta.json").exists() else {}
    runs = list(read_jsonl(REPORTS / "runs.jsonl"))
    L = ["# Laya fine-tune — benchmark results", "",
         f"- date: {datetime.now():%Y-%m-%d %H:%M} · machine: M5 16 GB · runtime: laya-mlx FP16",
         f"- data build: order-confirm yes/no rows **{meta.get('confirm_review', '?')}**",
         f"- models: " + ", ".join(f"`{m}` = `{bcfg['models'][m]}`" for m in models),
         "- every model is asked the **same canonical questions** (config/decisions.yaml → instructions[0]; "
         "order-confirm state = [agent question, customer reply]). The base numbers therefore differ from "
         "COMPARISON.md, which used different question wording.", ""]

    head = "| test set (n) | " + " | ".join(models) + (" | cascade |" if cascade else " |")
    sep = "|---|" + "---:|" * (len(models) + (1 if cascade else 0))
    L += ["## Headline", "", head, sep]

    def row(name, fn, casc=None):
        cells = [fn(m) for m in models]
        return f"| {name} | " + " | ".join(cells) + (f" | {casc if casc is not None else '—'} |" if cascade else " |")

    for s in ("confirm_frozen", "confirm_test"):
        c = cascade.get(s)
        L.append(row(f"**{s}** accuracy ({len(sets[s])})", lambda m, s=s: pct(M[m][s]["acc"]),
                     pct(sum(r["ok"] for r in c) / len(c)) if c else None))
        L.append(row(f"**{s} yes↔no mix-ups** (gate 0)", lambda m, s=s: str(len(leaks(results[m][s]))),
                     str(len(leaks(c))) if c else None))
    L.append(row(f"intent_test accuracy, all scripts ({len(sets['intent_test'])})", lambda m: pct(M[m]["intent_test"]["acc"])))
    L.append(row("intent_test accuracy, Bangla script (bn + transliterated)", lambda m: pct(bn_intent(m))))
    L.append(row(f"escalate_test accuracy ({len(sets['escalate_test'])})", lambda m: pct(M[m]["escalate_test"]["acc"])))
    for s in ("router_bn", "router_en"):
        L.append(row(f"{s} (intent+escalate, {len(sets[s])})", lambda m, s=s: pct(M[m][s]["acc"])))
    L.append(row("latency P50 / P95 per question", lambda m: f"{lat[m]['p50']:.1f} / {lat[m]['p95']:.1f} ms",
                 (f"{statistics.median(r['ms'] for r in cascade['confirm_test']):.1f} ms" if cascade.get("confirm_test") else None)))
    L.append(row("peak MLX memory", lambda m: f"{mem[m]:.0f} MiB"))
    L.append("")
    if cascade:
        kind = next(iter(cascade.values()))[0]["embedder"]
        L += [f"Cascade = hybrid ORDER_CONFIRM cascade (`{kind}` embedder); escalation counted as out_of_scope. "
              "It only makes the order-confirm decision, so the other rows are —.", ""]

    # gate
    g = bcfg["gate"]
    L += ["## Success gate", "", "| model | 0 yes↔no | confirm ≥ 90% | Bangla intent ≥ 85% | P50 ≤ 15 ms | verdict |",
          "|---|---|---|---|---|---|"]
    verdict = {}
    for m in models:
        lk = len(leaks(results[m]["confirm_frozen"])) + len(leaks(results[m]["confirm_test"]))
        ca = min(M[m]["confirm_frozen"]["acc"], M[m]["confirm_test"]["acc"])
        checks = [lk <= g["yes_no_leaks"], ca >= g["order_confirm_accuracy"],
                  bn_intent(m) >= g["intent_accuracy_bn"], lat[m]["p50"] <= g["p50_ms"]]
        verdict[m] = all(checks)
        L.append(f"| {m} | {'✅' if checks[0] else '❌'} {lk} | {'✅' if checks[1] else '❌'} {pct(ca)} | "
                 f"{'✅' if checks[2] else '❌'} {pct(bn_intent(m))} | {'✅' if checks[3] else '❌'} {lat[m]['p50']:.1f} | "
                 f"**{'PASS' if verdict[m] else 'FAIL'}** |")
    L.append("")

    # per-script intent
    scripts = sorted({r["script"] for r in sets["intent_test"]})
    L += ["## Intent accuracy by script (dataset test split)", "",
          "| script (n) | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
    for sc in scripts:
        n = sum(1 for r in sets["intent_test"] if r["script"] == sc)
        cells = []
        for m in models:
            rs = [r for r in results[m]["intent_test"] if r["script"] == sc]
            cells.append(pct(sum(r["ok"] for r in rs) / len(rs)))
        L.append(f"| {sc} ({n}) | " + " | ".join(cells) + " |")
    L.append("")

    # calibration
    L += [f"## Calibration (can a confidence threshold be trusted?)", "",
          f"`acc@≥{thr}` = accuracy on items whose top probability ≥ {thr}; `cov` = share of items above it.", "",
          "| model | set | Brier ↓ | ECE ↓ | cov@≥" + str(thr) + " | acc@≥" + str(thr) + " |", "|---|---|---:|---:|---:|---:|"]
    for m in models:
        for s in ("confirm_frozen", "confirm_test", "intent_test", "escalate_test"):
            x = M[m][s]
            L.append(f"| {m} | {s} | {x['brier']:.3f} | {x['ece']:.3f} | {pct(x['cov_hi'])} | {pct(x['acc_hi'])} |")
    L.append("")

    # confusion (order confirm)
    for s in ("confirm_frozen", "confirm_test"):
        L += [f"## Confusion — {s} (rows = expected, cols = predicted)", ""]
        cols = [("cascade", cascade.get(s))] if cascade.get(s) else []
        for m, rows in [(m, results[m][s]) for m in models] + cols:
            cm = Counter((r["label"], r["pred"]) for r in rows)
            L += [f"**{m}**", "", "| exp \\ pred | " + " | ".join(CONFIRM) + " |", "|---|" + "---:|" * 4]
            L += [f"| {e} | " + " | ".join(str(cm[(e, p)]) for p in CONFIRM) + " |" for e in CONFIRM]
            L.append("")

    # errors of the best fine-tuned model
    ft = [m for m in models if m != "base"]
    if ft:
        m = ft[0]
        L += [f"## {m} errors — order confirm", "", "| set | reply | expected | predicted | p |", "|---|---|---|---|---:|"]
        for s in ("confirm_frozen", "confirm_test"):
            for r in results[m][s]:
                if not r["ok"]:
                    L.append(f"| {s} | {r['text']} | {r['label']} | {r['pred']} | {r['pmax']:.2f} |")
        L.append("")
        errs = Counter((r["label"], r["pred"]) for r in results[m]["intent_test"] if not r["ok"])
        L += [f"## {m} most common intent confusions", "", "| expected | predicted | n |", "|---|---|---:|"]
        L += [f"| {a} | {b} | {n} |" for (a, b), n in errs.most_common(12)]
        L.append("")

    if runs:
        L += ["## Training runs (reports/runs.jsonl)", "", "| out | items | epochs | device | min | s/step | calib acc |",
              "|---|---:|---:|---|---:|---:|---|"]
        L += [f"| {r['out']} | {r['train_items']} | {r['epochs']} | {r['device']}/{r['amp']} | {r['minutes']} | "
              f"{r['sec_per_step']} | {r['calib_accuracy']} |" for r in runs]
        L.append("")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "FINETUNE_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    (REPORTS / "FINETUNE_RESULTS.json").write_text(json.dumps(
        {"metrics": M, "latency": lat, "memory_mib": mem, "verdict": verdict,
         "leaks": {m: {s: [r["text"] for r in leaks(results[m][s])] for s in ("confirm_frozen", "confirm_test")}
                   for m in models}}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for m in models:
        print(f"{m}: {'PASS' if verdict[m] else 'FAIL'} · confirm_frozen {pct(M[m]['confirm_frozen']['acc'])} "
              f"· confirm_test {pct(M[m]['confirm_test']['acc'])} · bn intent {pct(bn_intent(m))} "
              f"· leaks {len(leaks(results[m]['confirm_frozen'])) + len(leaks(results[m]['confirm_test']))}")
    print("→ reports/FINETUNE_RESULTS.md")
    fail = "R1" in results and (leaks(results["R1"]["confirm_frozen"]) or leaks(results["R1"]["confirm_test"]))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
