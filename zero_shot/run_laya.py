"""Laya (MLX) decision-model eval + latency benchmark for the Bangla voice agent.

Runs every case in cases/*.yaml through Laya's typed decisions (decisions.yaml),
then reports accuracy per question, yes↔no leakage (ORDER_CONFIRM hard gate),
per-call latency (P50/P95/max), model-load time and memory.

Writes reports/laya_<suite>.md and reports/laya_<suite>.json.

    .venv/bin/python run_laya.py                        # all suites, multilingual
    .venv/bin/python run_laya.py --suite order_confirm --repeats 5
    .venv/bin/python run_laya.py --model aac6fef/laya-mlx   # English checkpoint
    .venv/bin/python run_laya.py --offline              # never touch the network
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CASES_DIR = HERE / "cases"
REPORTS_DIR = HERE / "reports"
DEFAULT_MODEL = "aac6fef/laya-multilingual-mlx"


# ---------------------------------------------------------------- loading

def load_decisions() -> dict:
    with open(HERE / "decisions.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_suite(name: str) -> tuple[str, list[dict]]:
    """Return (decision_name, cases). Cases are {utterance, expect{...}}."""
    path = CASES_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases: list[dict] = []
    if src := data.get("source"):
        # Hybrid labeled format: labels: [{utterance, intent}]
        with open((path.parent / src).resolve(), encoding="utf-8") as f:
            ext = yaml.safe_load(f)
        cases += [{"utterance": x["utterance"], "expect": {"intent": x["intent"]}, "origin": "hybrid"}
                  for x in ext["labels"]]
    cases += [{**c, "origin": c.get("origin", "laya")} for c in data.get("cases") or []]
    return data["decision"], cases


def all_suites() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("*.yaml"))


# ---------------------------------------------------------------- scoring

def extract(answer: dict):
    """Normalize one Laya answer to a comparable prediction."""
    t = answer["type"]
    if t == "choice":
        return answer["choice"]
    if t == "noul":
        return answer["noul"] >= 0.5
    if t == "score":
        return round(answer["score"])
    raise ValueError(f"unknown answer type {t}")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def peak_mem_mb() -> float:
    import mlx.core as mx
    getter = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return getter() / 2**20


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


# ---------------------------------------------------------------- run

def run_suite(agent, suite: str, decisions: dict, repeats: int, warmup: int) -> dict:
    decision_name, cases = load_suite(suite)
    questions = decisions[decision_name]

    for _ in range(warmup):  # JIT / kernel warmup — excluded from timings
        agent.predict(cases[0]["utterance"], questions)

    rows, latencies = [], []
    for case in cases:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = agent.predict(case["utterance"], questions)
            times.append((time.perf_counter() - t0) * 1000)
        latencies += times
        answers = result["answers"]
        preds = {q: extract(a) for q, a in answers.items()}
        checks = {q: preds[q] == exp for q, exp in case["expect"].items()}
        rows.append({
            "utterance": case["utterance"],
            "origin": case["origin"],
            "bn": case.get("bn"),  # Bangla source for translated suites
            "expect": case["expect"],
            "pred": preds,
            "ok": checks,
            "confidence": {q: a["confidence"] for q, a in answers.items()},
            "probabilities": {q: a.get("probabilities") for q, a in answers.items()},
            "input_tokens": result.get("usage", {}).get("input_tokens"),
            "latency_ms": statistics.median(times),
        })

    per_q: dict[str, dict] = {}
    for r in rows:
        for q, ok in r["ok"].items():
            s = per_q.setdefault(q, {"correct": 0, "total": 0})
            s["total"] += 1
            s["correct"] += int(ok)

    leaks = [r for r in rows
             if {r["expect"].get("intent"), r["pred"].get("intent")} == {"yes", "no"}]

    return {
        "suite": suite,
        "decision": decision_name,
        "questions": list(questions),
        "rows": rows,
        "per_question": per_q,
        "yes_no_leaks": [r["utterance"] for r in leaks],
        "latency_ms": {
            "n": len(latencies),
            "p50": round(pct(latencies, 50), 2),
            "p95": round(pct(latencies, 95), 2),
            "mean": round(statistics.fmean(latencies), 2),
            "max": round(max(latencies), 2),
        },
    }


def render_md(meta: dict, res: dict) -> str:
    L = [f"# Laya eval — `{res['suite']}`", ""]
    L += [f"- model: `{meta['model']}` ({meta['dtype']}, compile={meta['compile']})",
          f"- machine: {meta['machine']}",
          f"- load: **{meta['load_s']:.2f} s** · peak MLX mem **{meta['peak_mlx_mb']:.0f} MiB** · "
          f"RSS **{meta['rss_mb']:.0f} MiB**",
          f"- questions per call: {len(res['questions'])} ({', '.join(res['questions'])})",
          f"- latency per call (repeats={meta['repeats']}): P50 **{res['latency_ms']['p50']} ms** · "
          f"P95 **{res['latency_ms']['p95']} ms** · max {res['latency_ms']['max']} ms",
          ""]
    for q, s in res["per_question"].items():
        L.append(f"- accuracy `{q}`: **{s['correct']}/{s['total']}** ({100 * s['correct'] / s['total']:.0f}%)")
    if res["decision"] == "order_confirm":
        n = len(res["yes_no_leaks"])
        L.append(f"- yes↔no leakage (hard gate = 0): **{n}** {'✅' if n == 0 else '❌'}")
        for u in res["yes_no_leaks"]:
            L.append(f"  - {u}")
    L += ["", "| # | utterance | expected | predicted | conf | ms | src |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(res["rows"], 1):
        exp = ", ".join(f"{k}={v}" for k, v in r["expect"].items())
        pred = ", ".join(f"{k}={r['pred'][k]}" for k in r["expect"])
        conf = ", ".join(f"{r['confidence'][k]:.2f}" for k in r["expect"])
        mark = "✅" if all(r["ok"].values()) else "❌"
        L.append(f"| {i} | {r['utterance']} | {exp} | {mark} {pred} | {conf} | "
                 f"{r['latency_ms']:.1f} | {r['origin']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--suite", default="all", help=f"one of {all_suites()} or 'all'")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--repeats", type=int, default=3, help="timed calls per case (median kept)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compile", action="store_true", help="MLX compile + prompt cache")
    ap.add_argument("--offline", action="store_true", help="HF_HUB_OFFLINE=1 (cached weights only)")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import laya_mlx as laya  # after env is set

    load_kw = {"dtype": args.dtype}
    if args.compile:
        load_kw.update(compile=True, pad_to_multiple=16, cache_prompts=True)
    t0 = time.perf_counter()
    agent = laya.load(args.model, **load_kw)
    load_s = time.perf_counter() - t0

    suites = all_suites() if args.suite == "all" else [args.suite]
    decisions = load_decisions()
    REPORTS_DIR.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1]
    exit_code = 0

    for suite in suites:
        res = run_suite(agent, suite, decisions, args.repeats, args.warmup)
        meta = {
            "model": args.model, "dtype": args.dtype, "compile": args.compile,
            "repeats": args.repeats, "load_s": load_s,
            "peak_mlx_mb": peak_mem_mb(), "rss_mb": rss_mb(),
            "machine": f"{platform.machine()} macOS {platform.mac_ver()[0]}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        stem = f"laya_{suite}" if tag == DEFAULT_MODEL.split("/")[-1] else f"laya_{suite}__{tag}"
        if args.compile:
            stem += "__compiled"
        (REPORTS_DIR / f"{stem}.json").write_text(
            json.dumps({"meta": meta, **res}, ensure_ascii=False, indent=2), encoding="utf-8")
        (REPORTS_DIR / f"{stem}.md").write_text(render_md(meta, res), encoding="utf-8")

        acc = " · ".join(f"{q} {s['correct']}/{s['total']}" for q, s in res["per_question"].items())
        print(f"[{suite}] {acc} · P50 {res['latency_ms']['p50']} ms · P95 {res['latency_ms']['p95']} ms"
              f" · leaks {len(res['yes_no_leaks'])} → reports/{stem}.md")
        if res["yes_no_leaks"]:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
