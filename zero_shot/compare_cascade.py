"""Side-by-side: Laya vs the hybrid intent cascade on the ORDER_CONFIRM set.

Needs the voice-agent repo: set VOICE_AGENT_BACKEND=/path/to/voice-agent-v1/backend and run
with that backend's venv (the cascade imports app/), after run_laya.py has written
reports/laya_order_confirm.json:

    make zs-compare            # from the repo root

Writes reports/compare_order_confirm.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_BACKEND = Path(os.environ["VOICE_AGENT_BACKEND"]).resolve()
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "tests" / "hybrid-agent" / "testkit"))

import run_intent_eval  # noqa: E402  (hybrid testkit — single source of the cascade eval)


def main() -> int:
    laya_path = HERE / "reports" / "laya_order_confirm.json"
    if not laya_path.exists():
        print("run `make zs-eval` first")
        return 2
    laya = json.loads(laya_path.read_text(encoding="utf-8"))
    laya_by_utt = {r["utterance"]: r for r in laya["rows"]}

    t0 = time.perf_counter()
    cascade = run_intent_eval.run_eval("prod")
    cascade_ms = (time.perf_counter() - t0) * 1000  # includes embedder load

    L = ["# Laya vs intent cascade — ORDER_CONFIRM (hybrid labeled set)", "",
         f"- cascade embedder: **{cascade['embedder_kind']}** "
         f"(eval wall time incl. load {cascade_ms:.0f} ms)",
         f"- Laya model: `{laya['meta']['model']}` · P50 {laya['latency_ms']['p50']} ms/call", ""]
    rows, lc = [], 0
    for utt, expected, predicted, source, conf in cascade["rows"]:
        lr = laya_by_utt.get(utt)
        lp = lr["pred"]["intent"] if lr else "—"
        lc += int(lp == expected)
        rows.append(f"| {utt} | {expected} | {'✅' if predicted == expected else '❌'} {predicted} "
                    f"({source} {conf}) | {'✅' if lp == expected else '❌'} {lp} |")
    n = cascade["total"]
    laya_leaks = [u for u in laya["yes_no_leaks"] if any(u == r[0] for r in cascade["rows"])]
    L += [f"| | cascade | Laya |", "|---|---|---|",
          f"| accuracy | **{cascade['correct']}/{n}** | **{lc}/{n}** |",
          f"| yes↔no leaks | **{len(cascade['leaks'])}** | **{len(laya_leaks)}** |", "",
          "| utterance | expected | cascade | Laya |", "|---|---|---|---|", *rows]
    out = HERE / "reports" / "compare_order_confirm.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"cascade {cascade['correct']}/{n} (leaks {len(cascade['leaks'])}) · "
          f"Laya {lc}/{n} (leaks {len(laya_leaks)}) → {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
