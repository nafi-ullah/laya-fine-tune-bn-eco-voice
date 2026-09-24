"""Step 8 — convert a fine-tuned PyTorch checkpoint to MLX FP16 and check parity.

    .venv/bin/python scripts/07_convert_mlx.py checkpoints/R1          # → checkpoints/R1-mlx

Parity: the same N test cases must pick the same option under PyTorch (the upstream `laya`
Agent) and MLX (`laya_mlx`). Gate: ≥ 98% identical choices (plan: ≥ 49/50).
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

from common import BUILD, ROOT, read_jsonl


def pick(answer: dict):
    return answer["choice"] if answer["type"] == "choice" else answer["noul"] >= 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    src = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    dst = src.with_name(src.name + "-mlx")
    exe = Path(sys.executable).with_name("laya-mlx")
    subprocess.run([str(exe), "convert", "--model", str(src), "--dtype", "float16", "--output", str(dst)], check=True)

    import laya
    import laya_mlx

    cases = [c for wf in ("order_confirm", "intent", "escalate") for c in read_jsonl(BUILD / f"test_{wf}.jsonl")]
    random.Random(0).shuffle(cases)
    cases = cases[:args.n]
    pt = laya.load(str(src), device="cpu")
    mx = laya_mlx.load(str(dst))
    same = 0
    for c in cases:
        a = pt.predict(c["state"], c["questions"])["answers"]
        b = mx.predict(c["state"], c["questions"])["answers"]
        q = next(iter(c["questions"]))
        same += int(pick(a[q]) == pick(b[q]))
    ok = same >= 0.98 * len(cases)
    print(f"converted → {dst.relative_to(ROOT)} · parity {same}/{len(cases)} identical choices "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
