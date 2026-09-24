# Laya — Bangla vs English comparison

**Date:** 2026-09-24 · **Machine:** M5 MacBook Pro 16 GB, macOS 26.4.1 · **Runtime:** `laya-mlx` 0.2.0, FP16,
MLX GPU, offline · **Timing:** 3 timed calls per case after 3 warm-up calls, model load excluded.

## What was tested

The same test cases were run in two languages on two checkpoints.

- **Bangla suites** — `cases/order_confirm.yaml` (the hybrid agent's 18 labeled ORDER_CONFIRM utterances + 5
  harder phrases) and `cases/ecommerce_router.yaml` (14 inbound-support utterances).
- **English suites** — `cases/order_confirm_en.yaml` and `cases/ecommerce_router_en.yaml`: one-to-one
  translations of the Bangla cases in the same order, with the same expected answers.
- **Questions** — identical, except that the ORDER_CONFIRM option descriptions give English example phrases
  in the English run (`order_confirm_en` in `decisions.yaml`) instead of Bangla ones.
  The router questions are the same English text in both runs.
- **Checkpoints** — `aac6fef/laya-multilingual-mlx` (mmBERT-base, 322M, 100+ languages) and
  `aac6fef/laya-mlx` (ModernBERT-large, 421M, English only).

## Headline

| | Multilingual 322M · **Bangla** | Multilingual 322M · **English** | English 421M · **Bangla** | English 421M · **English** |
|---|---:|---:|---:|---:|
| ORDER_CONFIRM accuracy (23) | 11 (48%) | **17 (74%)** | 8 (35%) | 13 (57%) |
| **yes↔no mix-ups** (must be 0) | 4 ❌ | **1** ❌ | 7 ❌ | 3 ❌ |
| Router intent (12) | 5 | 6 | 2 | **9 (75%)** |
| Router escalate (4) | 3 | 2 | 1 | 3 |
| Router frustration (3) | 0 | 0 | 0 | 2 |
| Latency, 1 question — typical / slow end (P50 / P95) | 10.0 / 11.4 ms | 10.2 / 11.4 ms | 31.6 / 40.5 ms | 22.6 / 29.0 ms |
| Latency, 3 questions — typical / slow end (P50 / P95) | 16.0 / 39.4 ms | 14.6 / 19.3 ms | 43.3 / 76.2 ms | 34.4 / 52.7 ms |
| Peak GPU (MLX) memory | ~785 MiB | ~785 MiB | ~1.2 GiB | ~1.2 GiB |

For reference, the hybrid agent's current intent cascade scores **18/18 with 0 mix-ups** on the Bangla
labeled set (`reports/compare_order_confirm.md`).

## Findings

1. **English is clearly easier for Laya, but no configuration passes the yes↔no gate.** The best
   setup (multilingual model on English) reaches 17/23 on ORDER_CONFIRM, still with one mix-up
   ("Okay, keep it" → no). The same model on Bangla gets 11/23 with 4 mix-ups.
2. **The multilingual model is the better choice for confirm/cancel in both languages.** It beats
   the English-only model even on English (17 vs 13). The English model has a strong bias
   toward `yes`: it answers `yes` to "No", "Cancel it", "Yeah, I don't need it", and to every
   delivery or payment question. On Bangla, it answers `yes` to all 23 cases, so its 8 correct answers are simply the 8 cases where the right answer was `yes`.
3. **The English model on English is the only configuration that routes support requests
   reasonably.** It gets 9/12 intents and 2/3 frustration levels, and is the only one that
   identifies product questions (price, size, warranty). The multilingual model sends almost
   everything that isn't a refund to `order_status`, in both languages.
4. **Language has almost no effect on speed.** The multilingual model takes about 10 ms per
   question in both languages. The English model is 2–3× slower and uses about 1.5× the memory
   (it is the larger model), and it's slower on Bangla than on English: Bangla text breaks into
   more tokens in its English-only vocabulary.
5. **Confidence can't be used as a filter in either language.** On ORDER_CONFIRM, correct and wrong
   answers both sit at 0.02–0.50, so no threshold would separate them. Only the router's
   clear-cut cases reach 0.9+.
6. **Some errors appear in every configuration**, so they come from the question design, not
   the language:
   - Delivery and payment questions ("When will the delivery come?", "cash on delivery, right?")
     are never recognised as `out_of_scope`.
   - "What time does your office open?" is never recognised as `other`.
   - "Thanks, everything is fine" always gets frustration level 1, not 0.
   - "I received the wrong color" is never recognised as a complaint.

## Where Bangla loses (multilingual model)

These cases were correct in English but wrong in Bangla, with the same model and question:

| Bangla (wrong) | English (right) | Expected |
|---|---|---|
| জি → repeat | Yep → yes | yes |
| ঠিক আছে নিয়ে নিন → **no** | Alright, go ahead with it → yes | yes |
| না → repeat | No → no | no |
| হ্যাঁ লাগবে না → **yes** | Yeah, I don't need it → no | no |
| ক্যান্সেল করে দেন → **yes** | Cancel it → no | no |
| প্রোডাক্টের মান কেমন → repeat | How is the product quality? → out_of_scope | out_of_scope |
| হ্যাঁ হ্যাঁ পাঠিয়ে দেন → repeat | Yes yes, send it → yes | yes |
| মনিটরটার ওয়ারেন্টি কত দিনের → refund | How long is the warranty on the monitor? → product_question | product_question |

Only one case went the other way: `এখন আর দরকার নেই` (no ✅) vs "I don't need it anymore" (repeat ❌). The failures on the most basic words
(`না`, `জি`) and on the Bangla-English mix `ক্যান্সেল` show that the multilingual model has seen
too little conversational Bangla.

## Per-utterance results

Each cell shows ✅/❌, the prediction, and (confidence).

### ORDER_CONFIRM

| # | Bangla | English | Expected | Multi · BN | Multi · EN | Eng · BN | Eng · EN |
|---|---|---|---|---|---|---|---|
| 1 | হ্যাঁ | Yes | yes | ✅ yes (0.05) | ✅ yes (0.08) | ✅ yes (0.33) | ✅ yes (0.26) |
| 2 | জি | Yep | yes | ❌ repeat (0.14) | ✅ yes (0.08) | ✅ yes (0.29) | ✅ yes (0.26) |
| 3 | হ্যাঁ ঠিক আছে করে দেন | Yes, okay, go ahead | yes | ✅ yes (0.36) | ✅ yes (0.30) | ✅ yes (0.31) | ✅ yes (0.39) |
| 4 | অর্ডারটা কনফার্ম করে দিন | Please confirm the order | yes | ✅ yes (0.33) | ✅ yes (0.50) | ✅ yes (0.29) | ✅ yes (0.33) |
| 5 | ঠিক আছে নিয়ে নিন | Alright, go ahead with it | yes | ❌ no (0.04) | ✅ yes (0.11) | ✅ yes (0.25) | ✅ yes (0.33) |
| 6 | আচ্ছা রাখেন | Okay, keep it | yes | ❌ no (0.05) | ❌ no (0.02) | ✅ yes (0.31) | ✅ yes (0.23) |
| 7 | না | No | no | ❌ repeat (0.12) | ✅ no (0.05) | ❌ yes (0.29) | ❌ yes (0.13) |
| 8 | অর্ডারটা বাতিল করে দেন | Please cancel the order | no | ✅ no (0.12) | ✅ no (0.17) | ❌ yes (0.30) | ✅ no (0.11) |
| 9 | এখন আর দরকার নেই | I don't need it anymore | no | ✅ no (0.11) | ❌ repeat (0.02) | ❌ yes (0.30) | ✅ no (0.08) |
| 10 | হ্যাঁ লাগবে না | Yeah, I don't need it | no | ❌ yes (0.19) | ✅ no (0.04) | ❌ yes (0.32) | ❌ yes (0.22) |
| 11 | ক্যান্সেল করে দেন | Cancel it | no | ❌ yes (0.13) | ✅ no (0.31) | ❌ yes (0.28) | ❌ yes (0.19) |
| 12 | চাই না | I don't want it | no | ✅ no (0.26) | ✅ no (0.20) | ❌ yes (0.31) | ✅ no (0.08) |
| 13 | আবার বলেন | Say that again | repeat | ❌ yes (0.15) | ❌ yes (0.23) | ❌ yes (0.32) | ✅ repeat (0.23) |
| 14 | কী বললেন বুঝিনি | I didn't understand what you said | repeat | ✅ repeat (0.16) | ✅ repeat (0.39) | ❌ yes (0.26) | ❌ yes (0.04) |
| 15 | আরেকবার বলুন | Please say it once more | repeat | ✅ repeat (0.20) | ✅ repeat (0.38) | ❌ yes (0.29) | ❌ yes (0.16) |
| 16 | ডেলিভারি কখন আসবে | When will the delivery come? | out_of_scope | ❌ no (0.05) | ❌ yes (0.02) | ❌ yes (0.29) | ❌ yes (0.18) |
| 17 | ডেলিভারি ম্যান কি বিকাশে নিবে | Will the delivery man take bKash? | out_of_scope | ❌ no (0.07) | ❌ no (0.05) | ❌ yes (0.27) | ❌ yes (0.06) |
| 18 | প্রোডাক্টের মান কেমন | How is the product quality? | out_of_scope | ❌ repeat (0.10) | ✅ out_of_scope (0.04) | ❌ yes (0.28) | ❌ yes (0.23) |
| 19 | হ্যাঁ হ্যাঁ পাঠিয়ে দেন | Yes yes, send it | yes | ❌ repeat (0.06) | ✅ yes (0.17) | ✅ yes (0.23) | ✅ yes (0.33) |
| 20 | না না ঠিক আছে, দিয়ে দেন | No no, it's fine, send it | yes | ✅ yes (0.09) | ✅ yes (0.22) | ✅ yes (0.30) | ✅ yes (0.31) |
| 21 | জি না, এখন নিব না | No thanks, I won't take it now | no | ✅ no (0.18) | ✅ no (0.24) | ❌ yes (0.28) | ✅ no (0.19) |
| 22 | লাইনটা কেটে যাচ্ছিল, আরেকবার বলবেন | The line was breaking up, can you say it again? | repeat | ✅ repeat (0.14) | ✅ repeat (0.21) | ❌ yes (0.24) | ❌ yes (0.13) |
| 23 | ক্যাশ অন ডেলিভারি হবে তো | It will be cash on delivery, right? | out_of_scope | ❌ no (0.06) | ❌ no (0.03) | ❌ yes (0.28) | ❌ yes (0.22) |

### Support router (intent · escalate · frustration)

| # | Bangla | English | Expected | Multi · BN | Multi · EN | Eng · BN | Eng · EN |
|---|---|---|---|---|---|---|---|
| 1 | আমার অর্ডার এখনো আসেনি | My order still hasn't arrived | order_status | ✅ order_status (1.00) | ✅ order_status (1.00) | ✅ order_status (0.05) | ✅ order_status (0.90) |
| 2 | অর্ডারটা কবে ডেলিভারি হবে | When will the order be delivered? | order_status | ✅ order_status (1.00) | ✅ order_status (0.64) | ❌ refund (0.03) | ✅ order_status (0.79) |
| 3 | পার্সেলটা কোথায় আছে একটু দেখবেন | Could you check where the parcel is? | order_status | ✅ order_status (0.98) | ✅ order_status (1.00) | ✅ order_status (0.04) | ✅ order_status (0.89) |
| 4 | আমি টাকা ফেরত চাই | I want my money back | refund | ✅ refund (1.00) | ✅ refund (1.00) | ❌ order_status (0.06) | ✅ refund (0.65) |
| 5 | প্রোডাক্টটা রিটার্ন করতে চাই, টাকাটা বিকাশে দিয়ে দেন | I want to return the product, send the money to my bKash | refund | ✅ refund (0.53) | ✅ refund (0.99) | ❌ order_status (0.03) | ✅ refund (0.69) |
| 6 | মনিটরটার ওয়ারেন্টি কত দিনের | How long is the warranty on the monitor? | product_question | ❌ refund (0.38) | ✅ product_question (0.26) | ❌ order_status (0.06) | ✅ product_question (0.17) |
| 7 | এটার দাম কত টাকা | How much does this cost? | product_question | ❌ order_status (0.68) | ❌ order_status (0.61) | ❌ order_status (0.07) | ✅ product_question (0.23) |
| 8 | এই শার্টটা কি XL সাইজে আছে | Is this shirt available in XL? | product_question | ❌ order_status (0.98) | ❌ order_status (0.94) | ❌ order_status (0.05) | ✅ product_question (0.11) |
| 9 | ভাঙা প্রোডাক্ট পাঠিয়েছেন, এটা কী ধরনের সার্ভিস | You sent a broken product, what kind of service is this? | complaint · true · 2 | ❌ order_status · false · 1 | ❌ order_status · false · 1 | ❌ order_status · false · 1 | ❌ refund · **true · 2** |
| 10 | তিনবার ফোন দিলাম কেউ ধরে না, খুবই বাজে অভিজ্ঞতা | I called three times and nobody picked up, terrible experience | complaint · true · 2 | ❌ order_status · **true** · 1 | ❌ order_status · false · 1 | ❌ refund · false · 1 | ❌ **complaint** · false · **2** |
| 11 | ভুল কালারের জিনিস এসেছে | I received the wrong color | complaint | ❌ order_status (0.77) | ❌ order_status (0.97) | ❌ order_status (0.07) | ❌ refund (0.06) |
| 12 | আপনাদের অফিস কয়টায় খোলে | What time does your office open? | other | ❌ order_status (0.52) | ❌ order_status (0.46) | ❌ order_status (0.04) | ❌ order_status (0.18) |
| 13 | মানুষের সাথে কথা বলতে চাই, ম্যানেজারকে দেন | I want to talk to a human, give me the manager | escalate true | ✅ true (0.76) | ✅ true (0.98) | ❌ false (0.90) | ✅ true (0.73) |
| 14 | ধন্যবাদ, সব ঠিক আছে | Thanks, everything is fine | escalate false · frustration 0 | ❌ false · 1 | ❌ false · 1 | ❌ false · 1 | ❌ false · 1 |

Rows 9–10: bold marks the parts that were correct.

## Conclusion

The gap between Bangla and English is real: the multilingual model goes from 48% to 74% on
ORDER_CONFIRM, and the English model from 2/12 to 9/12 on routing. But even the best English
result has a yes↔no mix-up and can't be trusted via its confidence scores. As a zero-shot model,
Laya can't replace the cascade's yes/no decision in either language. It could be worth
revisiting only after fine-tuning on labeled Bangla call utterances. Until then, its realistic
use is as an extra, low-risk signal for obvious order-status or refund requests and for
"let me talk to a human".

## Reproduce

```bash
# from the repo root
make zs-eval        # multilingual, all 4 suites
make zs-eval-en     # English checkpoint, all 4 suites
```

Raw per-run data is in `reports/laya_<suite>[_en][__laya-mlx].{md,json}`.
