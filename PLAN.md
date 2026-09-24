# Plan — Fine-tune Laya-multilingual for the Bangla voice agent

## Context

Zero-shot `laya-multilingual` (322M) failed our benchmark (`backend/tests/decision-model-test/laya/COMPARISON.md`):
- ORDER_CONFIRM: 11/23 correct, with 4 yes↔no mix-ups.
- Support-routing intent: 5/12.
- Confidence is useless: correct and wrong answers both score 0.02–0.50.

Speed is excellent (~10 ms per question on the M5, <1 GB). Upstream says the checkpoint is meant to be
fine-tuned per workflow. Their fine-tuned runs reached 0.76 on a customer-service workflow (base
≈0.34) and 0.63 on a browser-agent task that the base model could only do at chance.

**Goal:** fine-tune it into a Bangla voice-agent decision model and benchmark it against the base
model and the hybrid intent cascade.
- **Target:** zero yes↔no mix-ups, ≥90% ORDER_CONFIRM accuracy, ≥85% intent accuracy on the held-out split.
- **Keep:** ~10 ms latency after MLX conversion.

**Decisions (from the user):**
- **Dataset licence:** `Badhon/BanglaEComIntent` (CC-BY-NC-SA 4.0) is used for **research only**.
  The model is retrained on NC-free data before any production use.
- **First-round decisions:** order-confirm reply + 15-way support intent + escalate-to-human.
- **Local data LLM:** **Gemma 3 12B-it 4-bit via `mlx-lm`**.

## Facts established during exploration

**Laya's training format** is the one used by `LocalLLaMA/typed-decisions`: each case is `{id, workflow,
state, questions, gold}`.
- `questions`: `{name: {type: choice|noul|score, instructions, criteria}}`.
- `gold`: `{name: {type, label, probabilities: {option: p}}}`.
- Training uses `gold.probabilities` as **soft targets**.
- If we emit this format, the upstream training code works unchanged.

**Upstream training recipe:** `NandhaKishorM/laya` → `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`.
- **Loss:** "RLCD" = soft cross-entropy + a noisy-logit policy-gradient term.
- **Hyperparameters:** AdamW; learning rate 2.5e-5 for the encoder and 1e-4 for the decision head; 4 epochs;
  micro-batch 8 × gradient-accumulation 4.
- **Code it relies on:** `laya.common.build_sequence`, `render_options`, `QTYPES`, `build_model` and `proper_reward`.
- **Calibration:** ~10% of the data is held out; per-type temperatures are fitted with LBFGS on it.
- **Output:** a normal Laya checkpoint directory (`model.safetensors`, `encoder/`, `tokenizer/`,
  `rl_agent_config.json` with `temperature`).

**Lessons from `docs/finetune_browser_agent.md`:**
- Templated phrasing leaks into the model → vary instruction wording and shuffle option order.
- Rare classes need re-weighting.
- Turn gradient checkpointing **off** when memory allows (1.25× faster).
- Don't use `torch.compile` on variable-length batches.

**Getting the model back onto the Mac:** `laya-mlx convert --model <local dir> --dtype float16 --output <dir>`.
The existing eval harness `run_laya.py --model <path>` then benchmarks it directly.

**BanglaEComIntent** (gated; the HF token is needed):
- **Size:** train 5,432 / validation 794 / test 806.
- **Columns:** `text`, `intent` (15 classes, 265–660 each), `script`.
- **Scripts:** `bn` 1,715, `bl` (Banglish) 1,662, `en` 1,686, `mx` 369.
- **Intents:** greeting, goodbye, thanks, product_search, product_availability, price_inquiry,
  order_status, order_cancel, return_refund, shipping_delivery, payment_issue, discount_offer,
  complaint, agent_request, out_of_scope.
- It has **no** yes/no/repeat order-confirm replies; these must be generated.

**Seeds that already exist:** `backend/app/config/hybrid/anchors/{yes,no,repeat}.yaml` (~12 phrases each)
and the negation rules in `backend/app/config/hybrid/intents/*.yaml`.

**Where it runs:**
- `llama-server` is installed, but there is no GGUF model locally; `mlx-lm` is not installed.
- The laya-mlx venv is at `backend/tests/decision-model-test/laya/.venv` (Python 3.12).

**Security:** `models/laya-fine-tune/draft.md` contains a plaintext HF token.
`models/` is untracked and **not gitignored**, so one `git add .` would commit it.

## Layout (everything under `models/laya-fine-tune/`, isolated from `app/`)

```
models/laya-fine-tune/
  PLAN.md                  ← this plan (copied here as step 0)
  .gitignore               ← data/raw, data/cache, data/build, checkpoints/, .venv*/
  requirements-data.txt    ← mlx-lm, datasets, pyyaml, jsonschema, sentence-transformers
  requirements-train.txt   ← laya (PyTorch pkg), torch, transformers, safetensors
  config/
    decisions.yaml         ← the 3 training question schemas + instruction paraphrases
    intent_map.yaml        ← dataset intent → descriptions, escalate prior, order_confirm mapping
    llm.yaml               ← model id, batch size, temperature, prompts (no prompts in .py)
    split.yaml             ← seeds, calibration fraction, dedupe threshold, class weights
  schemas/
    llm_row.schema.json    ← what the LLM must return per sentence
    laya_case.schema.json  ← typed-decisions case format
  scripts/
    01_pull_dataset.py     ← HF → data/raw/*.parquet (token from env)
    02_llm_enrich.py       ← batched LLM passes, JSON-validated, resumable
    03_generate_confirm.py ← LLM paraphrases for yes/no/repeat, + review export
    04_build_cases.py      ← rows → Laya cases (soft targets, shuffling, paraphrase)
    05_split_dedupe.py     ← leak-free train/calib/test, eval-set exclusion
    06_preprocess.py       ← cases → tokenized items (upstream build_sequence)
    train.py               ← single-device RLCD trainer (MPS / CUDA / CPU)
    07_convert_mlx.sh      ← laya-mlx convert → checkpoints/*-mlx
    08_benchmark.py        ← base vs fine-tuned vs cascade → reports/FINETUNE_RESULTS.md
  notebooks/
    colab_train.ipynb      ← thin wrapper: clone scripts, pip, run 06+train, save to Drive
  data/ checkpoints/ reports/
```

## Step-by-step

### Step 0 — Housekeeping (5 min)
1. Move the HF token into `backend/.env` as `HF_TOKEN=` (already gitignored) and delete it from `draft.md`.
   Rotate the token on huggingface.co if `models/` was ever shared or synced.
2. Add `models/laya-fine-tune/.gitignore` so data, caches and checkpoints are never committed.
3. Copy this plan to `models/laya-fine-tune/PLAN.md`.
4. Create two isolated venvs:
   - `.venv-data`: mlx-lm, datasets, … — Mac only.
   - `.venv-train`: `laya` from PyPI, torch — Mac MPS, or Colab.

### Step 1 — Pull the dataset (`01_pull_dataset.py`)
- `datasets.load_dataset("Badhon/BanglaEComIntent", token=os.environ["HF_TOKEN"])`.
- Write all 3 splits to `data/raw/{train,validation,test}.parquet`.
- Record the revision SHA and licence in `data/raw/SOURCE.json`.
- Keep the dataset's own **test split untouched**; it becomes the intent benchmark.

### Step 2 — LLM enrichment (`02_llm_enrich.py`)

**Runtime**
- Model: `mlx_lm.load("mlx-community/gemma-3-12b-it-4bit")`, loaded once (~7 GB).
- Sentences go to the LLM in **batches of 10–20**, as a numbered JSON list. The LLM returns a JSON
  array in which each item echoes its `id`.

**Validation and retries**
- Each item is validated against `schemas/llm_row.schema.json`.
- If an item is invalid or missing, only that sentence is retried (max 2 retries).
- After that, it goes to `data/cache/rejects.jsonl` and never silently enters training.

**Settings**
- temperature 0.2 for labelling, 0.8 for paraphrasing.
- Gemma's chat template, with an explicit "return JSON only" instruction plus one worked example.

**Resumability:** results are appended to `data/cache/enrich.jsonl`, keyed by
`sha1(text + pass + prompt_version)`, so a crash or a prompt change only reruns what's needed.

**Passes (each is a prompt in `config/llm.yaml`):**
1. **Script normalisation** (only `bl` and `mx` rows, ~2k).
   - Transliterate Banglish into natural Bangla script, as Soniox would output it: keep English
     product words like "অর্ডার" and "ডেলিভারি".
   - Output `{id, bn_text}`.
   - The original row is kept too; the transliteration is an extra training row with the same label.
   - Reason: our STT emits Bangla script, not romanised text.
2. **Decision labels** (all ~7k rows).
   - Output `{id, escalate: 0|1, alt_intent: <one of 15>|null, ambiguity: low|high}`.
   - The gold intent always comes from the **dataset label**, never the LLM.
   - The LLM adds only the missing signals: escalate, and an optional second-best intent for
     genuinely ambiguous rows such as "টাকা কেটে নিয়েছে কিন্তু অর্ডার দেখাচ্ছে না".
3. **Sanity check.** If the LLM's own guess at the intent disagrees with the dataset label, the row
   is flagged for review in `reports/label_disagreements.csv`. It is not auto-relabelled.

**Throughput estimate:** ~200k output tokens at ~15–20 tokens/s ≈ 3–4 h, run overnight.
Pass 1 alone takes ~30 min.

### Step 3 — Generate order-confirm data (`03_generate_confirm.py`)

The dataset has none of this, so it is generated from the anchors.

**Seeds**
- `app/config/hybrid/anchors/{yes,no,repeat}.yaml` plus the negation rules in `intents/*.yaml`.
- **Minus every utterance in the eval sets** (hybrid `labeled.yaml` and our `order_confirm*.yaml`).
  Several anchors, e.g. "ঠিক আছে নিয়ে নিন" and "চাই না", are also eval items and must be dropped
  from the seeds.

**Generation**
- The LLM paraphrases seeds into ~300 per class for yes, no and repeat.
- Explicit **hard-negative prompts**: হ্যাঁ + negation (লাগবে না / নিব না), না + affirmation (না না ঠিক আছে),
  polite জি না, code-mixed "cancel/confirm", and fillers/disfluencies (উম, মানে, আসলে).
- Mix of Bangla script and some Banglish.

**out_of_scope class:** no generation needed. Sample ~300 real rows from the dataset's train split
(shipping_delivery, payment_issue, price_inquiry, product_*, complaint).

**Human review gate**
- Export `reports/confirm_review.csv` (utterance, proposed label, seed).
- A person accepts or rejects each **yes/no** item before it can enter training.
- This mirrors the hybrid rule "never auto-promote into yes/no without review".
- Budget ~1 h for about 600 yes/no rows. repeat and out_of_scope get a spot check only.

### Step 4 — Build Laya cases (`04_build_cases.py`)

Rows are turned into typed-decisions cases (`schemas/laya_case.schema.json`) with three decision
families, defined in `config/decisions.yaml`.

| decision | type | state | options | gold (soft target) |
|---|---|---|---|---|
| `order_confirm` | choice | conversation list: agent's confirm question + customer reply (the format used at inference) | yes / no / repeat / out_of_scope | 0.92 on the reviewed label, the rest spread evenly |
| `intent` | choice | customer utterance | 15 dataset intents, each with a one-line description | 0.90 on the gold label; if the LLM marked the row high-ambiguity, 0.70 gold / 0.20 second-best / 0.10 spread |
| `escalate` | noul | customer utterance | true / false | agent_request → 0.95 true; complaint → the LLM's call, softened to 0.8/0.2; everything else → 0.1 true |

- **Phrasing variety:** each case gets one of 3–4 **instruction paraphrases** and a **shuffled
  option order**. This is upstream's anti-leak lesson. Evaluation uses a fixed wording.
- **Tool selection is not trained.** It is a deterministic function of the intent
  (`order_status → get_order_status`, …) in business logic, so it would only add label noise.
- **Class balance:** weights live in `config/split.yaml`. `order_confirm` is up-weighted ×2 because it is
  the critical path.

**Expected size:** ~7k dataset rows plus ~2k transliterations ≈ 9k intent cases, + 9k escalate
cases, + ~1.2k confirm cases → **about 19k training items**. Items are short (<200 tokens).

### Step 5 — Split and remove leaks (`05_split_dedupe.py`)

**Test sets (never trained on)**
- The dataset's `test` split, plus the transliterations derived from it.
- A generated confirm test set: 15% of the reviewed rows, split **by seed**, so paraphrases of the
  same seed can't land on both sides.
- Our hand-written eval suites (hybrid `labeled.yaml` and `cases/*.yaml`), kept as a **frozen
  external benchmark**.

**Removing near-duplicates**
- Normalise the text; for exact matches, drop the training copy.
- Embed everything with the existing MiniLM in the sentence-transformers cache and drop any training
  row with cosine ≥0.95 to a test row.

**Calibration slice:** 10% of train, held out as upstream does.

Outputs: `data/build/{train,calib,test_*}.jsonl` + `reports/data_card.md` (counts per class/script/source).

### Step 6 — Tokenize (`06_preprocess.py`)
- Reuse upstream's item builder verbatim: `build_sequence(tok, state, {t, ins, crit}, max_len, head_max_len)`,
  `render_options`, `QTYPES`.
- The base checkpoint is `convaiinnovations/laya-multilingual` (PyTorch, not the MLX one).
- Save `data/build/{train,calib}_items.pt`.
- Assert that the marker count equals the option count; print dropped items.

### Step 7 — Train (`train.py`)

`train.py` is a single-device port of the upstream `train_ddp.py`:
- DDP removed.
- Device auto-selects `cuda → mps → cpu`.
- Same RLCD loss, optimiser, learning rates, cosine schedule, rolling checkpoint per epoch, and
  temperature calibration on the calib slice.

**7a. M5 dry run first** (`--device mps --max-items 500 --epochs 1`)
- fp32 (no GradScaler); gradient checkpointing on.
- Memory: 322M params ≈ 5–6 GB weights + gradients + Adam, which fits in 16 GB with micro-batch 8
  of short sequences.
- Measure seconds per step, then extrapolate to the full run.
- If a full 4-epoch run is **≤ ~2 h and stable**, train on the M5 (nothing to upload; the data stays local).

**7b. Colab Pro otherwise** (`notebooks/colab_train.ipynb`)
- Use an A100 or L4 with fp16 autocast + GradScaler, gradient checkpointing **off** (1.25× faster).
- Upload only `train_items.pt`, `calib_items.pt` and `train.py` to Drive. The HF token goes into
  Colab Secrets.
- Estimated at well under 1 h on an A100; the upstream 1,200-case/6k-decision run took 4–6 min on 2×T4.
- The result checkpoint is downloaded back into `checkpoints/`.

**Run matrix** (small, each logged to `reports/runs.jsonl`):
- **R1:** all three decisions, 4 epochs (upstream defaults).
- **R2:** same, with the NC-free subset only (generated confirm + LLM-rewritten intents), to
  measure what we lose when the licensed data is removed.
- **R3** (optional): the English base `convaiinnovations/laya` on the same data, for contrast.

### Step 8 — Convert to MLX (`07_convert_mlx.sh`)
- Run `laya-mlx convert --model checkpoints/R1 --dtype float16 --output checkpoints/R1-mlx`.
- Parity check: 50 random test items must pick the same option under the PyTorch and MLX
  checkpoints (≥49/50).

### Step 9 — Benchmark and compare (`08_benchmark.py` → `reports/FINETUNE_RESULTS.md`)

Every model is scored on every test set:

| | base multilingual | fine-tuned R1 (MLX) | fine-tuned R2 | hybrid cascade |
|---|---|---|---|---|
| ORDER_CONFIRM: hybrid labeled 18 + our 5 traps (frozen) | 11/23 | ? | ? | 18/18 (labeled) |
| ORDER_CONFIRM: generated test split | | | | |
| **yes↔no mix-ups** (gate = 0) | 4 | | | 0 |
| 15-way intent: dataset test split (by script: bn/bl/en/mx) | | | | n/a |
| our router suites (BN + EN) | 5/12 · 6/12 | | | n/a |
| escalate accuracy / F1 | | | | |
| Brier score, ECE, and accuracy when confidence ≥0.8 | | | | |
| latency P50/P95 (M5, MLX fp16), peak memory | 10 ms | | | |

- **Reuse:** the fine-tuned checkpoints run through the existing `run_laya.py` + `cases/*.yaml`
  (the harness already takes `--model <path>`), and `compare_cascade.py` supplies the cascade column.
- **Calibration:** add a reliability table (confidence bucket → accuracy). This shows whether a
  confidence threshold is now usable to decide when to fall back to the cascade or LLM.
- **Success gate:**
  - **0 yes↔no mix-ups** on every confirm test set.
  - ≥90% confirm accuracy.
  - ≥85% intent accuracy on the Bangla-script part of the test split.
  - P50 ≤15 ms.
- **If the gate fails:**
  1. Look at the error table.
  2. Add targeted hard negatives (Step 3).
  3. Do one DAgger-style round: run the model over new LLM-generated utterances, keep the mistakes
     after review, retrain.

### Step 10 — Report and decide
- Write `reports/FINETUNE_RESULTS.md`: what was trained, the data card, the table above, the error
  analysis, and a go/no-go.
- **If go:** propose a follow-up ADR to use Laya as an extra cascade tier (after regex/embedding,
  before the LLM), gated by the calibrated threshold. **This plan does not touch production code.**
- **Before production:** retrain on the R2 (NC-free) data plus labelled real call transcripts.

## Critical files

**Read (reused):**
- `backend/tests/decision-model-test/laya/run_laya.py` — the harness takes `--model <path>`.
- `backend/tests/decision-model-test/laya/compare_cascade.py`, `…/cases/*.yaml`, `…/decisions.yaml`.
- `backend/tests/hybrid-agent/cases/intents/labeled.yaml` — frozen eval set, excluded from training.
- `backend/app/config/hybrid/anchors/{yes,no,repeat}.yaml` — generation seeds, minus eval overlap.
- Upstream `laya.common` (`build_sequence`, `render_options`, `QTYPES`, `build_model`, `proper_reward`) and
  the notebook's `train_ddp.py` — the base for `train.py`.

**Created:** everything under `models/laya-fine-tune/` (layout above).
**Edited:** `models/laya-fine-tune/draft.md` (token removed) and `backend/.env` (`HF_TOKEN` added).

## Verification

1. **Data:**
   - `04`/`05` validate every case against `laya_case.schema.json`.
   - Assert zero exact or near-duplicate overlap between train and every test set, and print the
     leak count (must be 0).
   - `reports/data_card.md` shows no empty class.
2. **Pipeline smoke test (≈10 min):**
   - Run steps 2→8 on 200 rows.
   - Train 1 epoch on MPS.
   - Convert and load with `laya_mlx.load(path)`.
   - `run_laya.py --model checkpoints/smoke-mlx --suite order_confirm` completes.
3. **Full run:** `08_benchmark.py` produces `FINETUNE_RESULTS.md`. The success gate above is checked
   and printed as PASS/FAIL, with exit code 1 on any yes↔no mix-up (same convention as `run_laya.py`).
4. **Parity:** PyTorch vs MLX on 50 items is ≥49/50 identical choices.

---

## Changes made during implementation (2026-09-24)

| Plan said | Done instead | Why |
|---|---|---|
| Gemma 3 12B via mlx-lm | **Gemini `gemini-3.1-flash-lite`** (REST, JSON mode, minimal thinking, 8 parallel requests) | User's call. Gemma ran at ~10 tok/s, so the full run was ~8 h; Gemini took minutes. Whole enrichment + generation cost ≈ $0.35 ($0.25 / $1.50 per 1M in/out) |
| LLM sees the dataset label | Label pass is **blind**: the LLM guesses intent, and a disagreement becomes the second-best intent + a review flag | With the label visible, the model copied it (30/30) |
| Two venvs | One `.venv` (data + torch + laya-mlx) | mlx 0.32.2 satisfies both mlx-lm and laya-mlx; Colab only installs `requirements-train.txt` |
| `intent_map.yaml` | Intent descriptions in `decisions.yaml`; escalate priors and out_of_scope mapping in `split.yaml` | One fewer file |
| MiniLM near-duplicate check at 0.95 | **multilingual-e5-small**; 0.95 against the frozen eval set, **0.98** against test splits | MiniLM is English-only. e5 scores same-template Bangla ≥0.95 even with opposite meaning ("…কনফার্ম করে দিন" vs "…ক্যান্সেল করে দেন" = 0.956), so 0.95 dropped 55% of the confirm data |
| MPS in fp32 | **bf16 autocast** on MPS, gradient checkpointing on | Measured 0.98 vs 1.44 s/step. Turning checkpointing off was *slower* (3.98 s/step, 15 GB peak → swapping) |
| `07_convert_mlx.sh` | `07_convert_mlx.py` (convert + parity check in one) | |
| Human review before training | Review sheet + a **blind Gemini cross-check** (0/917 disagreements). A preliminary R1 was trained with `--allow-unreviewed`, tagged UNREVIEWED in every report | Keeps things moving; the final model must be retrained after review |
| R2 = NC-free subset | R2 = **order-confirm only**: 1,162 items | Everything else (intent, escalate, out_of_scope rows) derives from the NC dataset, including the LLM transliterations |

Bugs caught by smoke tests: the dataset CSV's string labels vs its ClassLabel card (read the CSVs directly);
read-only HF-cache files breaking the per-epoch checkpoint save (copy with `copyfile`); the confirm style
rotation skipping every second style, which silently dropped the hard-negative styles.

**Moved (2026-09-24):** this work now lives in its own repo, `laya-fine-tune-bn-eco-voice`. The
`models/laya-fine-tune/` folder became the repo root, and `backend/tests/decision-model-test/laya/` became `zero_shot/`.
The voice-agent inputs were copied in: `eval_sets/hybrid_order_confirm_labeled.yaml` and `seeds/anchors/`.
Keys moved to a local `.env`. The cascade comparison reaches the voice agent via `VOICE_AGENT_BACKEND`.
Paths in the sections above refer to the old layout.
