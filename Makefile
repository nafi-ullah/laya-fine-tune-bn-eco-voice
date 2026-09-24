# Laya fine-tune pipeline. Run from the repo root:  make <target>
-include .env
export
PY := .venv/bin/python
S  := scripts

.PHONY: setup pull enrich confirm build build-unreviewed train train-r2 convert bench colab-pack hf-export hf-push hf-model all zs-eval zs-eval-en zs-compare

setup:              ## one venv: data (Gemini/httpx) + training (laya/torch) + laya-mlx
	uv venv -q --python 3.12 .venv
	uv pip install -q --python $(PY) -r requirements-data.txt -r requirements-train.txt

pull:               ## step 1 — dataset → data/raw (needs HF_TOKEN in .env)
	$(PY) $(S)/01_pull_dataset.py

enrich:             ## step 2 — Gemini translit + labels (needs GEMINI_API_KEY; ≈ $0.25)
	$(PY) $(S)/02_llm_enrich.py --pass all

confirm:            ## step 3 — generate order-confirm replies + blind cross-check → reports/confirm_review.csv
	$(PY) $(S)/03_generate_confirm.py

build:              ## steps 4–6 — cases, leak-free split, tokenize (uses only REVIEWED yes/no rows)
	$(PY) $(S)/04_build_cases.py
	$(PY) $(S)/05_split_dedupe.py
	$(PY) $(S)/06_preprocess.py
	$(PY) $(S)/06_preprocess.py --only-nc-free

build-unreviewed:   ## same, but also takes unreviewed non-suspect yes/no rows (preliminary runs only)
	$(PY) $(S)/04_build_cases.py --allow-unreviewed
	$(PY) $(S)/05_split_dedupe.py
	$(PY) $(S)/06_preprocess.py
	$(PY) $(S)/06_preprocess.py --only-nc-free

train:              ## step 7 — R1 on the M5 (~2.2 h, MPS bf16)
	caffeinate -i $(PY) $(S)/train.py --out checkpoints/R1

train-r2:           ## R2 — NC-free items only (order-confirm)
	caffeinate -i $(PY) $(S)/train.py --out checkpoints/R2 --items-suffix _ncfree

convert:            ## step 8 — MLX FP16 + PyTorch/MLX parity
	$(PY) $(S)/07_convert_mlx.py checkpoints/R1
	[ ! -d checkpoints/R2 ] || $(PY) $(S)/07_convert_mlx.py checkpoints/R2

bench:              ## step 9 — base vs R1 vs R2 vs cascade → reports/FINETUNE_RESULTS.md
	HF_HUB_OFFLINE=1 $(PY) $(S)/08_benchmark.py

jev:                ## Laya zero-shot vs fine-tuned vs Jev (OpenRouter), Bangla + English → reports/JEV_COMPARISON.md
	HF_HUB_OFFLINE=1 $(PY) $(S)/12_jev_compare.py

colab-pack:         ## bundle items + code for notebooks/colab_train.ipynb
	$(S)/pack_colab.sh

hf-export:          ## dataset → data/hf_export (parquet per config + dataset card with stats)
	$(PY) $(S)/09_export_hf.py

hf-push:            ## upload data/hf_export to the Hub (private; config/hf.yaml)
	$(PY) $(S)/10_push_hf.py

hf-model:           ## upload checkpoints/R1 (+ MLX) with model card to the Hub (private)
	$(PY) $(S)/11_push_model.py

all: build train train-r2 convert bench

# ---------------------------------------------------------------- zero-shot baseline (zero_shot/)
ZS_MODEL ?= aac6fef/laya-multilingual-mlx

zs-eval:            ## zero-shot Laya on the hand-written suites (BN + EN) → zero_shot/reports/
	cd zero_shot && ../$(PY) run_laya.py --offline --model $(ZS_MODEL) $(ARGS)

zs-eval-en:         ## same with the English-only 421M checkpoint
	.venv/bin/hf download aac6fef/laya-mlx
	cd zero_shot && ../$(PY) run_laya.py --offline --model aac6fef/laya-mlx $(ARGS)

zs-compare:         ## zero-shot Laya vs the voice agent's intent cascade (needs VOICE_AGENT_BACKEND)
	@test -n "$$VOICE_AGENT_BACKEND" || { echo "set VOICE_AGENT_BACKEND (e.g. in .env) to the voice-agent backend/ dir"; exit 1; }
	$$VOICE_AGENT_BACKEND/venv/bin/python zero_shot/compare_cascade.py
