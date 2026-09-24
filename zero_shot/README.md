# Laya decision-model test (local, Apple Silicon)

Evaluates **Laya** — the open-weight non-autoregressive "System 1" typed-decision model
([upstream](https://github.com/NandhaKishorM/laya)) — as a local decision engine for the
Bangla voice agent, via the **MLX port** [`laya-mlx`](https://github.com/mizorewww/laya-mlx)
and the multilingual checkpoint `aac6fef/laya-multilingual-mlx` (mmBERT-base, 322M params,
1,024-token context).

This is the **zero-shot baseline** that motivated the fine-tune in the repo root: it scores the
untouched checkpoints on hand-written Bangla and English suites. It uses the repo's shared `.venv`.
`compare_cascade.py` also needs the voice-agent repo (`VOICE_AGENT_BACKEND` in `.env`).

## Layout

| file | purpose |
|---|---|
| `decisions.yaml` | typed question schemas (`choice` / `noul` / `score`) — nothing hard-coded in `.py` |
| `cases/order_confirm.yaml` | reuses `../eval_sets/hybrid_order_confirm_labeled.yaml` verbatim + 5 extra traps |
| `cases/ecommerce_router.yaml` | inbound-support routing: intent + escalate + frustration |
| `cases/*_en.yaml` | one-to-one English translations (the `bn` field holds the Bangla source) |
| `run_laya.py` | accuracy + yes↔no leakage + latency P50/P95 + load time + memory → `reports/` |
| `compare_cascade.py` | Laya vs the voice agent's intent cascade on identical utterances |

## Run (from the repo root)

```bash
make zs-eval                         # multilingual checkpoint, all suites → zero_shot/reports/
make zs-eval-en                      # English 421M checkpoint, for contrast
make zs-compare                      # vs the cascade → zero_shot/reports/compare_order_confirm.md
make zs-eval ARGS="--suite order_confirm --dtype float32 --repeats 10"
```

`run_laya.py` exits 1 if any yes↔no mix-up occurs.

See **[COMPARISON.md](COMPARISON.md)** for the Bangla vs English comparison (`cases/*_en.yaml`).

## Results — 2026-09-24, M5 MacBook Pro 16 GB, macOS 26.4.1, FP16

| | multilingual 322M | English 421M |
|---|---:|---:|
| model load (cached) | 0.9 s | — |
| peak MLX memory | ~785 MiB | — |
| ORDER_CONFIRM, 1 question, P50 / P95 | **10.1 / 11.7 ms** | 31 / 55 ms |
| router, 3 questions, P50 / P95 | **15.6 / 19.8 ms** | 47 / 81 ms |
| ORDER_CONFIRM intent accuracy (23) | 11/23 (48%) | 8/23 |
| **yes↔no leaks** (hard gate = 0) | **4 ❌** | 7 ❌ |
| router intent accuracy (12) | 5/12 | 2/12 |
| router escalate (4) / frustration (3) | 3/4 · 0/3 | 1/4 · 0/3 |

`--compile` adds nothing measurable here (9.9 vs 10.1 ms P50).

**Compared with the current cascade on the 18 hybrid-labeled utterances**
(`reports/compare_order_confirm.md`): cascade **18/18, 0 leaks**; Laya **8/18, 4 leaks**:
`ঠিক আছে নিয়ে নিন`→no, `আচ্ছা রাখেন`→no, `হ্যাঁ লাগবে না`→yes, `ক্যান্সেল করে দেন`→yes.

A prompt-framing check (list labels, English descriptions, conversation-list state) scored
between 9/23 and 12/23. The low accuracy comes from the model, not the prompt.

## Verdict

- **Speed and footprint are as advertised.** About 10 ms per decision and under 1 GB of memory. It runs fully offline on this Mac.
- **Zero-shot accuracy on short Bangla replies is not usable for confirm/cancel decisions.**
  It confuses yes and no, which the DSM's hard gate forbids. Confidence is also very low
  (mostly 0.05–0.35), so a confidence threshold can't filter out the bad answers.
- Where it did well: clear order-status and refund requests (≥0.98 confidence), and asking
  for a human. Possible use: an extra **escalation / routing signal** alongside the cascade,
  never in place of the yes/no regex + embedding tiers.
- To be a real candidate it would need fine-tuning on our Bangla utterances. Training (RLCD)
  lives in the upstream repo; `laya-mlx` is inference only.
