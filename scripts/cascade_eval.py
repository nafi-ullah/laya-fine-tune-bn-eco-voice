"""Run the hybrid intent cascade (ORDER_CONFIRM state) over a JSONL of {"text", "label"} rows.

Must run with the voice-agent *backend* venv (it imports app/). Called by 08_benchmark.py when
VOICE_AGENT_BACKEND points at the voice-agent backend/ directory:

    $VOICE_AGENT_BACKEND/venv/bin/python scripts/cascade_eval.py IN.jsonl OUT.jsonl

Mirrors the voice agent's tests/hybrid-agent/testkit/run_intent_eval.py: prod profile (real embedder)
with a regex-only fallback; escalation is reported as `out_of_scope`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(os.environ["VOICE_AGENT_BACKEND"]).resolve()
sys.path.insert(0, str(BACKEND))

from app.voice.hybrid.config.loader import load_flow  # noqa: E402
from app.voice.hybrid.nlu.cascade import build_cascade  # noqa: E402


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    flow = load_flow("order_confirmation")
    state, globals_ = flow["states"]["ORDER_CONFIRM"], flow.get("globals", [])
    try:
        cascade, kind = build_cascade("prod"), "real"
        if cascade._embedder is None:
            kind = "regex-only"
    except Exception:  # noqa: BLE001 — same fallback as run_intent_eval
        cascade, kind = build_cascade("test"), "regex-only"
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        t0 = time.perf_counter()
        d = cascade.decide(row["text"], state, globals_)
        ms = (time.perf_counter() - t0) * 1000
        out.append({**row, "pred": "out_of_scope" if d.is_escalate else d.intent,
                    "tier": d.source.value, "ms": round(ms, 3), "embedder": kind})
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
