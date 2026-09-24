# Laya fine-tune — round 1 analysis (2026-09-24)

Numbers: `FINETUNE_RESULTS.md`. Data build: order-confirm yes/no rows **UNREVIEWED** (preliminary run).

## Verdict: not ready to replace the cascade; clear fix for round 2

| | base | **R1** (all decisions) | R2 (NC-free, confirm only) | cascade |
|---|---:|---:|---:|---:|
| order-confirm, generated test (186) | 43.0% | **99.5%** | 75.3% | 72.6% |
| yes↔no mix-ups, generated test | 52 | **1** | 1 | 13 |
| order-confirm, frozen hand-written set (23) | 65.2% | 69.6% | 82.6% | **95.7%** |
| yes↔no mix-ups, frozen set | 4 | **0** | 0 | 1 |
| intent, Bangla script (549) | 21.7% | **90.5%** | 35.2% | — |
| intent, all scripts (1,091) | 29.9% | **85.5%** | 37.1% | — |
| escalate (1,091) | 13.9% | **95.1%** | 12.7% | — |
| latency P50 per question (M5, MLX fp16) | 13.9 ms | 16.2 ms | 17.5 ms | <0.1 ms |

## What worked
- **Fine-tuning works.** On unseen data, intent went from 30% to 86% (from 22% to 91% in Bangla script) and escalate from 14% to 95%.
  Order-confirm on the unseen generated set reached 99.5%, beating the cascade (72.6%, with 13 yes↔no mix-ups there).
- **Confidence is now usable within the training distribution.** For intent, predictions with max-probability ≥ 0.8 are 96.5% accurate,
  covering 73% of messages. For escalate the figure is 96.9% (base: 39% and 13%).
- **No yes↔no mix-ups on the frozen set.** All 7 frozen-set errors went to `out_of_scope`, which is the safe outcome (escalate).
- **MLX conversion is lossless** (50/50 parity), with memory under 750 MiB.

## What failed, and why
1. **Very short replies → `out_of_scope`.** R1 misreads হ্যাঁ, জি, না, চাই না, আচ্ছা রাখেন, ঠিক আছে নিয়ে নিন and
   এখন আর দরকার নেই as off-topic, with confidence 0.55–0.91. The cause is data coverage: only **5 of 917** generated
   replies are ≤ 3 words (median 7), while the `out_of_scope` class is built from dataset rows that are often short.
   The model learned "short reply → off-topic". Real phone replies are mostly one or two words.
   Seed exclusion made this worse: the anchor phrases that are eval items (হ্যাঁ, না, চাই না …) were removed from
   the seeds, and nothing else supplied short forms.
2. **The one yes↔no mix-up** is "না না আর কিছু যোগ করবেন না, এটাই ফাইনাল।" (label `yes`; predicted `no`, p=0.89).
   It is genuinely hard ("no no, don't add anything, this is final"), and exactly the kind of row the human review should settle.
3. **R2 (order-confirm-only) forgot the other decisions** (intent 35%, escalate 13%). That's expected; it only proves the
   NC-free confirm data alone is too narrow. Its better frozen-set score (82.6%) comes from the same short-reply issue, not from better learning.
4. **Latency is 16–17 ms, against a gate of 15.** This measurement mixes order-confirm and 15-option intent questions. The long
   intent option list is most of the cost; the confirm question alone runs at about 10 ms, as in COMPARISON.md.

## Round 2 — recommended changes
1. **Generate short replies.** Add a "1–3 words" style for yes / no / repeat (target ~100 per class) plus
   filler-only variants (হুম, জি জি, না না). Down-weight very short `out_of_scope` rows, or require them to be multi-word.
2. **Decide the eval-hygiene rule for closed-set words** (needs a decision, see below).
3. **Human review** of `reports/confirm_review.csv` (yes/no rows), then `make build`. This time without `--allow-unreviewed`.
4. Retrain R1 (~2.4 h on the M5) → `make convert bench`.
5. Optionally shorten the intent option descriptions to bring latency under 15 ms.

**Decision needed.** Single-word replies (হ্যাঁ, জি, না) are a small closed set, and the frozen eval set contains them. Under
the current leak rule, any training reply within cosine 0.95 of them is dropped, so the model can never be taught the
most common answers. The options:
- **(a)** Allow closed-set short words in training, and treat the frozen set's single-word items as a coverage check rather than a generalisation test.
- **(b)** Keep the strict rule and accept that the regex tier of the cascade handles single words. Laya would then only be trusted on longer replies.
