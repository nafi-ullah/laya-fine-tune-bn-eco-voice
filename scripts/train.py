"""Step 7 — single-device RLCD fine-tuning of a Laya checkpoint (MPS / CUDA / CPU).

Port of upstream `train_ddp.py` (NandhaKishorM/laya notebooks/laya_finetune_typed_decisions_2xT4_kaggle
.ipynb) with DDP removed: same loss (noisy-logit policy gradient with a proper-scoring-rule
reward + soft cross-entropy), same optimiser split and learning rates, cosine schedule,
rolling per-epoch checkpoint, and post-hoc temperature calibration on the held-out calib set.

    # M5 dry run (measures s/step, extrapolates the full run)
    .venv/bin/python scripts/train.py --out checkpoints/dryrun --max-items 500 --epochs 1
    # full run
    .venv/bin/python scripts/train.py --out checkpoints/R1
    # Colab (CUDA): same command; AMP + no gradient checkpointing are picked automatically
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from laya.common import build_model, collate_items, proper_reward
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, REPORTS, append_jsonl  # noqa: E402

_prep = __import__("06_preprocess")   # reuse base_dir() (download + tokenizer-config fix)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_one_temp(sel) -> float:
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@torch.no_grad()
def eval_logits(model, items, pad_id, device, amp_ctx, bs=32):
    model.eval()
    out = []
    for chunk in batches(items, bs):
        b = collate_items([chunk], pad_id)
        with amp_ctx():
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        l = logits.float().cpu().numpy()
        for r, it in enumerate(chunk):
            out.append((it, l[r, :len(it["markers"])]))
    return out


def save_checkpoint(model, cfg, base, out_dir: Path, extra_cfg: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out_dir / "model.safetensors"))
    for sub in ("encoder", "tokenizer"):      # copy the base files verbatim (laya-mlx convert reads them)
        # copyfile, not copy2: HF cache blobs are read-only and a copied mode would make the
        # next per-epoch save fail with EACCES.
        shutil.copytree(os.path.join(base, sub), out_dir / sub, dirs_exist_ok=True,
                        copy_function=shutil.copyfile)
    c = dict(cfg)
    if extra_cfg:
        c.update(extra_cfg)
    (out_dir / "rl_agent_config.json").write_text(json.dumps(c, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items-suffix", default="", help="'_ncfree' for the R2 run")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, help="dry run: train on the first N items only")
    ap.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp = args.amp
    if amp == "auto":
        # MPS: bf16 autocast measured 1.46x faster than fp32 on the M5 (0.98 vs 1.44 s/step).
        if device.type == "cuda":
            amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            amp = "bf16" if device.type == "mps" else "none"
    grad_ckpt = args.grad_ckpt == "on" or (args.grad_ckpt == "auto" and device.type != "cuda")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)

    def amp_ctx():
        return torch.autocast(device.type, dtype=dtype) if dtype else torch.autocast(device.type, enabled=False)

    base = _prep.base_dir(args.base)
    with open(os.path.join(base, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    from laya.agent import _load_tokenizer
    tok = _load_tokenizer(os.path.join(base, "tokenizer"), cfg)

    model = build_model(cfg, encoder_dir=os.path.join(base, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(base, "model.safetensors")), strict=True)
    if grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)

    train_items = torch.load(BUILD / f"train_items{args.items_suffix}.pt", weights_only=False)
    calib_items = torch.load(BUILD / f"calib_items{args.items_suffix}.pt", weights_only=False)
    if args.max_items:
        train_items = train_items[:args.max_items]
        calib_items = calib_items[:max(50, args.max_items // 10)]

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_encoder},
                                   {"params": head_params, "lr": args.lr_head}], weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_items) / (args.micro_batch * args.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates_per_epoch * args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))

    out = Path(args.out)
    print(f"device={device} amp={amp} grad_ckpt={grad_ckpt} · train={len(train_items)} calib={len(calib_items)} "
          f"· {updates_per_epoch} updates/epoch × {args.epochs}", flush=True)
    t0 = time.time()
    step_times = []
    history = []
    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(args.seed + epoch)
        rng.shuffle(train_items)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * (epoch / max(1, args.epochs - 1))
        ep_loss, n_b = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for b_idx, chunk in enumerate(batches(train_items, args.micro_batch)):
            ts = time.perf_counter()
            b = collate_items([chunk], tok.pad_token_id)
            with amp_ctx():
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                                    b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            qtype = b["qtype"].to(device)

            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()

            last = (b_idx + 1) * args.micro_batch >= len(train_items)
            if (b_idx + 1) % args.grad_accum == 0 or last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if device.type == "mps":
                torch.mps.synchronize()
            step_times.append(time.perf_counter() - ts)
            ep_loss += loss.item() * args.grad_accum
            n_b += 1
            if n_b % 50 == 0:
                sps = sum(step_times[-50:]) / 50
                left = sps * (math.ceil(len(train_items) / args.micro_batch) * (args.epochs - epoch) - n_b)
                print(f"  epoch {epoch+1}/{args.epochs} step {n_b} loss {loss.item()*args.grad_accum:.4f} "
                      f"ce {loss_ce.item():.4f} · {sps:.2f}s/step · eta {left/60:.1f} min", flush=True)
        history.append({"epoch": epoch + 1, "avg_loss": ep_loss / max(1, n_b)})
        print(f"=== epoch {epoch+1} avg loss {ep_loss/max(1,n_b):.4f} · {(time.time()-t0)/60:.1f} min", flush=True)
        save_checkpoint(model, cfg, base, out / "checkpoint_latest", {"epoch": epoch + 1})

    # ---- temperature calibration on held-out calib items + calib accuracy per workflow
    preds = eval_logits(model, calib_items, tok.pad_token_id, device, amp_ctx)
    temps = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, it["target"]) for it, z in preds if it["qtype"] == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    acc = {}
    for it, z in preds:
        a = acc.setdefault(it["workflow"], [0, 0])
        a[0] += int(int(z.argmax()) == it["label"])
        a[1] += 1
    calib_acc = {w: round(c / n, 4) for w, (c, n) in acc.items()}

    cfg.pop("temperature_by_options", None)     # per-type fit; stale buckets would override it
    cfg.update({"fine_tuned": True, "model_name": "laya-kormi-bn", "temperature": temps,
                "training": {"base": args.base, "epochs": args.epochs, "train_items": len(train_items),
                             "items_suffix": args.items_suffix, "device": str(device), "amp": amp,
                             "minutes": round((time.time() - t0) / 60, 1), "history": history,
                             "calib_accuracy": calib_acc}})
    save_checkpoint(model, cfg, base, out)
    shutil.rmtree(out / "checkpoint_latest", ignore_errors=True)
    sps = sum(step_times) / max(1, len(step_times))
    summary = {"out": str(out), "device": str(device), "amp": amp, "grad_ckpt": grad_ckpt,
               "train_items": len(train_items), "epochs": args.epochs, "sec_per_step": round(sps, 3),
               "minutes": round((time.time() - t0) / 60, 1), "temperatures": temps, "calib_accuracy": calib_acc,
               "history": history}
    append_jsonl(REPORTS / "runs.jsonl", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
