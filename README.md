# laya-fine-tune-bn-eco-voice

Fine-tuning **[Laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)** (322M,
non-autoregressive "System 1" typed-decision model) into a fast, local **decision engine for a
Bangla-speaking e-commerce voice agent**. It runs on a MacBook (MLX) at ~10–16 ms per decision, with
no tokens generated.

It makes three decisions from a (speech-to-text) customer utterance:

| decision | type | options |
|---|---|---|
| `order_confirm` | choice | `yes` · `no` · `repeat` · `out_of_scope` — reply to "shall I confirm your order?" |
| `intent` | choice | 15 support intents (order status, cancel, refund, delivery, payment, complaint, human agent, …) |
| `escalate` | noul | P(a human agent should take over) |

**Published artefacts** (private until you flip them on the Hub):
- model: [`nafiullah/laya-multilingual-bn-ecom-voice`](https://huggingface.co/nafiullah/laya-multilingual-bn-ecom-voice) — PyTorch + MLX (`mlx/`)
- dataset: [`nafiullah/bangla-ecom-voice-decisions`](https://huggingface.co/datasets/nafiullah/bangla-ecom-voice-decisions) — `intent`, `order_confirm`, `laya` configs

## Results so far (round 1)

| | zero-shot base | **fine-tuned R1** | rule/embedding cascade |
|---|---:|---:|---:|
| order-confirm, unseen generated replies (186) | 43.0% | **99.5%** | 72.6% |
| ↳ yes↔no mix-ups | 52 | **1** | 13 |
| order-confirm, hand-written frozen set (23) | 65.2% | 69.6% | **95.7%** |
| intent, Bangla script | 21.7% | **90.5%** | — |
| escalate | 13.9% | **95.1%** | — |
| latency P50 (M5, MLX fp16) | 13.9 ms | 16.2 ms | < 0.1 ms |

Fine-tuning works (intent 22% → 91%, escalate 14% → 95%). The main open issue is that **one- and
two-word replies (হ্যাঁ, জি, না) are read as `out_of_scope`**, because almost no training replies were that
short. See [`reports/ANALYSIS.md`](reports/ANALYSIS.md) for the diagnosis and the round-2 fix.
Details: [`reports/FINETUNE_RESULTS.md`](reports/FINETUNE_RESULTS.md) · zero-shot study:
[`zero_shot/COMPARISON.md`](zero_shot/COMPARISON.md) · against Jev (TypeSafe's closed decision model, via OpenRouter),
in Bangla and English: [`reports/JEV_COMPARISON.md`](reports/JEV_COMPARISON.md).

## Repository layout

```
├── Makefile                 every step is a make target (run from the repo root)
├── PLAN.md                  the original plan + what changed during implementation
├── config/
│   ├── decisions.yaml       the 3 question schemas (+ training paraphrases) — the model's "API"
│   ├── llm.yaml             Gemini model, batch sizes, every prompt (no prompts in .py)
│   ├── split.yaml           soft targets, class weights, dedupe thresholds, frozen eval sets, extra data dir
│   ├── benchmark.yaml       which models to benchmark, success gate
│   └── hf.yaml              Hub repo names, visibility, licence, credits
├── schemas/                 JSON schemas: LLM output rows, Laya training case
├── scripts/                 01…11 pipeline steps + train.py, llm.py, common.py, cascade_eval.py
├── seeds/anchors/           hand-written yes/no/repeat seed phrases for order-confirm generation
├── eval_sets/               frozen hand-written order-confirm eval set (never trained on)
├── zero_shot/               zero-shot baseline study (Bangla vs English, base checkpoints)
├── notebooks/colab_train.ipynb   train on a Colab GPU instead of the Mac
├── reports/                 data card, benchmark, analysis, review sheets, training runs
├── data/                    (gitignored) raw → cache → build, plus data/extra/ for your own rows
└── checkpoints/             (gitignored) R1, R1-mlx, R2, R2-mlx
```

## Setup

Apple Silicon Mac, Python 3.12, [`uv`](https://docs.astral.sh/uv/). Training also runs on any CUDA GPU (see Colab).

```bash
make setup                   # .venv with torch + laya (training), laya-mlx (inference), Gemini client
cp .env.example .env         # then fill in:
#   HF_TOKEN            read access to the gated source dataset; write access to push to the Hub
#   GEMINI_API_KEY      data enrichment + generation (steps 2–3); the whole run costs ≈ $0.35
#   VOICE_AGENT_BACKEND optional: path to the voice-agent backend/, for the cascade comparison column
```

`data/` and `checkpoints/` are not in git. Either rebuild them with the pipeline below, or pull the
published artefacts:

```bash
.venv/bin/hf download nafiullah/laya-multilingual-bn-ecom-voice --local-dir checkpoints/R1-hub
```

## Using the fine-tuned model

The model only works well with the **exact question schemas it was trained on**: `config/decisions.yaml`
(also shipped as `questions.yaml` in the model repo). Use `instructions[0]` and the listed options. For
`order_confirm`, pass the conversation: the agent's question, then the customer's reply.

```python
import yaml, laya_mlx                       # PyTorch: import laya; laya.load(repo)

d = yaml.safe_load(open("config/decisions.yaml"))
agent = laya_mlx.load("nafiullah/laya-multilingual-bn-ecom-voice", subfolder="mlx")

def ask(name, state):
    q = d[name]
    return agent.predict(state, {name: {"type": q["type"], "instructions": q["instructions"][0],
                                        "criteria": q["criteria"]}})["answers"][name]

ask("order_confirm", [{"role": "assistant", "content": d["order_confirm"]["agent_turns"][0]},
                      {"role": "user", "content": "না না ঠিক আছে, দিয়ে দেন"}])   # → yes
ask("intent", "আমার অর্ডারটা এখনো আসেনি")                                         # → order_status
ask("escalate", "ম্যানেজারের সাথে কথা বলতে চাই")                                    # → noul ≈ 1.0
```

Until round 2 fixes short replies, **keep a keyword rule in front of it for single-word yes/no**.

## The pipeline, step by step

| step | command | what happens | output |
|---|---|---|---|
| 1 | `make pull` | download [BanglaEComIntent](https://huggingface.co/datasets/Badhon/BanglaEComIntent) (gated; accept its terms on the Hub first) | `data/raw/{train,validation,test}.parquet` |
| 2 | `make enrich` | Gemini: Banglish/mixed → **Bangla script** (as STT would write it); blind **intent guess + second choice + escalate** per row | `data/cache/{translit,label}.jsonl`, `reports/label_disagreements.csv` |
| 3 | `make confirm` | Gemini: ~300 **order-confirm replies** per class from `seeds/anchors/`, in hard-negative styles, + blind cross-check | `data/cache/confirm_gen.jsonl`, `reports/confirm_review.csv` |
| — | **human review** | put `y`/`n` in `accept` for every yes/no row of `reports/confirm_review.csv` (risky rows are sorted to the top) | |
| 4–6 | `make build` | build typed-decision cases with **soft targets** → leak-free split → tokenize with Laya's own `build_sequence` | `data/build/*`, `reports/data_card.md` |
| 7 | `make train` | RLCD fine-tune (upstream recipe), M5 MPS bf16 ≈ 2.4 h | `checkpoints/R1`, `reports/runs.jsonl` |
| 8 | `make convert` | `laya-mlx convert` to FP16 + PyTorch/MLX parity check | `checkpoints/R1-mlx` |
| 9 | `make bench` | base vs R1 vs R2 vs cascade on every test set + success gate (exits 1 on any yes↔no mix-up) | `reports/FINETUNE_RESULTS.{md,json}` |
| 9b | `make jev` | Laya zero-shot vs R1 vs Jev (`typesafe/jev-1.13`, needs `OPENROUTER_API_KEY`) on paired Bangla / English sets; answers cached, ≈ $0.05 per full run | `reports/JEV_COMPARISON.{md,json}` |
| 10 | `make hf-export hf-push` · `make hf-model` | publish dataset / model with cards and credits | Hub repos |

All steps are **resumable and cached**. The LLM steps key each result by (text, prompt version), so
re-running only processes what's new or what changed.

### How the data is preprocessed

1. **Script normalisation.** Customers write Banglish ("amar order ta cancel kore den"), but Bangla speech-to-text
   outputs Bangla script. Every Banglish/mixed row gets a Bangla-script copy (extra training row, same label).
2. **Blind labelling.** The LLM never sees the gold intent. It guesses independently. When its top guess or second
   choice differs from the gold label, that becomes the *alternative* in the soft target (gold 0.70 / alt 0.20), and the
   row is listed in `reports/label_disagreements.csv`. The gold intent itself is never changed.
3. **Order-confirm generation.** The source dataset has no yes/no replies. Seeds from `seeds/anchors/` (minus any phrase
   in the frozen eval sets) are paraphrased across styles, including hard negatives such as "না না ঠিক আছে দিয়ে দেন" (= yes)
   and "হ্যাঁ, এখন আর লাগবে না" (= no). `out_of_scope` replies are real support messages from the dataset.
4. **Soft targets.** `intent` 0.90 gold (ambiguous rows 0.70 / 0.20 alt); `order_confirm` 0.92; `escalate` P(true) 0.95
   for `agent_request`, 0.80 when the LLM flags escalation, 0.10 otherwise. All values are in `config/split.yaml`.
5. **Anti-shortcut formatting.** Training cases sample one of several instruction wordings and shuffle the option order.
   Test cases use one fixed wording.
6. **Leak-free splits.** Test = the dataset's own test split + held-out generation *groups*. Train rows that are
   exact or near duplicates (multilingual-e5-small, cos ≥ 0.98 vs test, ≥ 0.95 vs the frozen hand-written sets) are
   dropped. A 10% calibration slice is held out for temperature fitting. The build fails if any exact leak remains.

## Fine-tuning further

### Continue from the published model

```bash
.venv/bin/hf download nafiullah/laya-multilingual-bn-ecom-voice --local-dir checkpoints/R1
make build                                       # after adding / reviewing data
.venv/bin/python scripts/train.py --base checkpoints/R1 --out checkpoints/R3 --epochs 2
.venv/bin/python scripts/07_convert_mlx.py checkpoints/R3
# add `R3: checkpoints/R3-mlx` under models: in config/benchmark.yaml, then
make bench
```

`--base` takes any Laya checkpoint directory or Hub id. Use fewer epochs and/or a lower `--lr-encoder` when
continuing, so the model doesn't forget what it already learned.

### Add your own labelled data (recommended: real call transcripts)

Drop JSONL files into `data/extra/` (gitignored). They're picked up by `make build`:

```json
{"decision": "order_confirm", "text": "হ্যাঁ ভাই দিয়ে দেন", "label": "yes", "source": "calls-2026-10"}
{"decision": "intent", "text": "আমার পার্সেল কোথায়", "label": "order_status"}
{"decision": "escalate", "text": "ম্যানেজারকে দেন", "label": "true", "split": "test"}
```

`split` defaults to `train`. Put a slice of real calls in `test` so the benchmark measures what matters.
These rows are marked NC-free and also go into the R2 (`--only-nc-free`) build.

### Round 2 (known next step)

From [`reports/ANALYSIS.md`](reports/ANALYSIS.md): add a "1–3 words" style for yes/no/repeat in
`config/llm.yaml → passes.confirm_gen.styles` (about 100 per class), decide the eval-hygiene rule for closed-set words
(হ্যাঁ/জি/না are in the frozen set), review the sheet, then `make confirm build train convert bench`.

### Change or add a decision

1. Edit or add the schema in `config/decisions.yaml` (type, `instructions` list, `criteria`).
2. Produce rows for it: either a new LLM pass in `config/llm.yaml` plus a builder in `scripts/04_build_cases.py`, or simply
   `data/extra/*.jsonl` rows once `04_build_cases.py` knows the decision name.
3. `make build train convert bench`. Update `config/benchmark.yaml` if the gate should cover it.

### Train on a GPU instead (Colab)

`make colab-pack` → upload `dist/laya-colab.zip` to `MyDrive/laya-fine-tune/` → open
`notebooks/colab_train.ipynb` on an A100/L4 → copy `checkpoints/<RUN>` back → `make convert bench`. On CUDA,
`train.py` automatically uses bf16/fp16 and turns off gradient checkpointing.

## Things learned the hard way

- **MPS:** bf16 autocast is 1.46× faster than fp32. Turning off gradient checkpointing on 16 GB is *slower* (memory swapping).
- **LLM labelling:** showing the gold label makes the model copy it. Keep the labelling pass blind.
- **Near-duplicate filtering in Bangla:** e5-small scores same-template sentences ≥ 0.95 even with *opposite* meaning
  ("…কনফার্ম করে দিন" vs "…ক্যান্সেল করে দেন" = 0.956). Don't use 0.95 against a generated test split.
- **Generation styles:** check the style rotation actually visits every style. An off-by-one here silently removed all hard negatives.
- **Coverage beats volume:** 917 fluent replies but only 5 short ones produced a model that fails on "হ্যাঁ".
- **Hugging Face cache files are read-only.** Copy them with `shutil.copyfile`, or per-epoch checkpoint saves fail.

## Licence and credits

- **Data and fine-tuned weights: CC BY-NC-SA 4.0** (non-commercial). They derive from
  [BanglaEComIntent](https://huggingface.co/datasets/Badhon/BanglaEComIntent) (CC BY-NC-SA 4.0). Credit it whenever
  you use the data or model. Files in `reports/` that quote dataset rows fall under the same licence. For commercial
  use, retrain on NC-free data only (`--only-nc-free` + your own `data/extra/` rows).
- **Base model:** [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) (Apache-2.0),
  with the [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) encoder (MIT).
- **Training recipe and model code:** [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0). `scripts/train.py`
  is a single-device port of its fine-tuning notebook.
- **Apple Silicon runtime and conversion:** [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) (Apache-2.0).
- The pipeline code in this repo is yours to license. Add a `LICENSE` file (e.g. Apache-2.0) before publishing.
