# Laya (zero-shot, fine-tuned) vs Jev — Bangla and English

- date: 2026-09-24 19:41
- **Laya zero-shot** = `aac6fef/laya-multilingual-mlx` (322M, open, MLX on the M5).
- **Laya fine-tuned (R1)** = `checkpoints/R1-mlx`, this repo's fine-tune ([HF](https://huggingface.co/nafiullah/laya-multilingual-bn-ecom-voice)).
- **Jev 1.13** = TypeSafe's closed decision model through OpenRouter's Decisions API (`typesafe/jev-1.13-20260917`). Same choice / noul question format as Laya.
- All three models get **byte-identical** state and questions: the canonical wording from `config/decisions.yaml`. Order confirm state = [agent question, customer reply]; the English sets use the agent turn "Shall I confirm your order?".
- Bold = best in row. Fine-tuned Laya has seen the *training* splits of these task families; zero-shot Laya and Jev have not. The test rows are never trained on.

## Takeaways

1. **Hand-written order-confirm replies: Jev wins outright.** It scores 100% in both Bangla and English with 0 yes↔no mix-ups, while fine-tuned Laya scores 69.6% (bn) and 82.6% (en).
   - All of R1's misses are very short replies ("হ্যাঁ", "জি", "না", "চাই না", "Yep", "No"), which it reads as `out_of_scope`. This is the known round-2 gap: only 5 of 917 generated training replies were ≤3 words.
   - R1 has no yes↔no mix-up on the Bangla frozen set. In English it has one: "Okay, keep it" (a yes) → no.
2. **Generated order-confirm test split: R1 and Jev are tied.**
   - Bangla: R1 99.5% vs Jev 98.4%. English: R1 96.8% vs Jev 98.4%.
   - R1 makes 1 yes↔no mix-up per language; Jev makes 0.
   - Caveat: this split comes from the same Gemini generator as R1's training replies, so it is in-distribution for R1 and a friendlier test for it than for Jev.
3. **Intent and escalate: R1 wins on Bangla and Banglish; Jev wins on English intent.**
   - Bangla intent: R1 90.5% vs Jev 82.3%. English intent: Jev 90.2% vs R1 85.4%.
   - Romanized Banglish is Jev's weak spot: 58.0% intent and 66.3% escalate, against R1's 76.0% / 95.8%. This matches TypeSafe's own note that English is Jev's strongest language.
   - Caveat: the escalate labels follow this dataset's conventions (agent_request → yes; complaints labelled by Gemini), and R1 learned those conventions. Most of Jev's escalate errors are *extra* hand-offs (60 bn / 85 banglish): it sends complaints to a human, which is a defensible policy. It is rarely ≥0.8 confident on escalate, but when it is, it is always right.
4. **Hand-written router suites: Jev 93.8% vs R1 81.2%, in both languages.** Only 16 items each, so a difference of one item moves the score ~6 points.
5. **Zero-shot Laya is not usable in either language.**
   - It makes 4–52 yes↔no mix-ups per set.
   - It escalates almost everything: 474 of 549 Bangla rows sent to a human without need.
   - English helps it a little (38% vs 22% pooled), but not enough.
6. **Speed and deployment favour Laya.**
   - Laya answers in ~10–17 ms on the Mac, with no network and no per-call cost. Jev is a ~360 ms (P50) HTTPS round trip.
   - Jev is still very cheap: ~$0.02 per 1,000 decisions ($0.045 for this whole benchmark). Latency and the external dependency are the real cost on a live phone call.
   - Jev's weights are closed; R1's are open (CC BY-NC-SA 4.0).

**Bottom line.** Fine-tuned Laya is already on par with or ahead of Jev on Bangla / Banglish intent, escalate and generated confirm replies, and it runs ~25× faster locally. Jev is better at bare one-word confirm replies and at English intent, zero-shot. The cheapest way to close R1's gap is round 2 (add short closed-set replies like হ্যাঁ / জি / না / চাই না to training), or keep one-word replies on the hybrid regex tier. Jev is a strong zero-shot fallback for low-confidence turns if a ~0.4 s network hop is acceptable.

## Headline — accuracy

| test set | lang | n | Laya zero-shot | Laya fine-tuned (R1) | Jev 1.13 |
|---|---|---:|---:|---:|---:|
| confirm_frozen | bn | 23 | 65.2% | 69.6% | **100.0%** |
| confirm_frozen | en | 23 | 52.2% | 82.6% | **100.0%** |
| confirm_test | bn | 186 | 43.0% | **99.5%** | 98.4% |
| confirm_test | en | 186 | 48.4% | 96.8% | **98.4%** |
| intent | bn | 549 | 21.7% | **90.5%** | 82.3% |
| intent | en | 254 | 48.4% | 85.4% | **90.2%** |
| intent | banglish | 288 | 29.2% | **76.0%** | 58.0% |
| escalate | bn | 549 | 13.3% | **95.1%** | 86.7% |
| escalate | en | 254 | 18.1% | **94.1%** | 89.0% |
| escalate | banglish | 288 | 11.5% | **95.8%** | 66.3% |
| router | bn | 16 | 56.2% | 81.2% | **93.8%** |
| router | en | 16 | 62.5% | 81.2% | **93.8%** |

## yes↔no mix-ups on order confirm (the dangerous error — gate is 0)

| test set | lang | n | Laya zero-shot | Laya fine-tuned (R1) | Jev 1.13 |
|---|---|---:|---:|---:|---:|
| confirm_frozen | bn | 23 | 4 | 0 | 0 |
| confirm_frozen | en | 23 | 7 | 1 | 0 |
| confirm_test | bn | 186 | 52 | 1 | 0 |
| confirm_test | en | 186 | 31 | 1 | 0 |

## By language (all decisions pooled)

| language | n | Laya zero-shot | Laya fine-tuned (R1) | Jev 1.13 |
|---|---:|---:|---:|---:|
| bn | 1323 | 22.4% | 93.2% | 86.8% |
| en | 733 | 38.3% | 91.1% | 92.2% |
| banglish | 576 | 20.3% | 85.9% | 62.2% |

## Speed, cost, deployment

| | Laya zero-shot | Laya fine-tuned (R1) | Jev 1.13 |
|---|---:|---:|---:|
| latency P50 / P95 per decision | 17.4 / 20.7 ms | 16.5 / 21.0 ms | 370.8 / 523.2 ms |
| runs on | local, M5 MacBook (MLX FP16) | local, M5 MacBook (MLX FP16) | cloud API (OpenRouter → TypeSafe), `typesafe/jev-1.13-20260917` |
| peak memory | 745 MiB | 759 MiB | — |
| cost of this benchmark | $0 (runs locally) | $0 (runs locally) | $0.045 for 2,362 decisions (≈ $0.0192 / 1k) |
| weights | open (Apache-2.0 base) | open (CC BY-NC-SA 4.0) | closed |

Jev latency is a full HTTPS round trip from this Mac to OpenRouter, measured on 40 sequential uncached calls; Laya latency is in-process on the Mac.

## Calibration (confidence ≥ 0.8)

`cov` = share of items whose top probability ≥ threshold, `acc` = accuracy on those items.

| test set | Laya zero-shot cov / acc | Laya fine-tuned (R1) cov / acc | Jev 1.13 cov / acc |
|---|---:|---:|---:|
| confirm_frozen/bn | 13.0% / 100.0% | 82.6% / 78.9% | 91.3% / 100.0% |
| confirm_frozen/en | 17.4% / 100.0% | 60.9% / 92.9% | 100.0% / 100.0% |
| confirm_test/bn | 7.0% / 100.0% | 98.9% / 99.5% | 96.2% / 100.0% |
| confirm_test/en | 4.8% / 100.0% | 90.9% / 98.8% | 97.3% / 100.0% |
| intent/bn | 52.5% / 26.7% | 73.0% / 99.0% | 70.9% / 98.2% |
| intent/en | 69.3% / 58.5% | 76.0% / 95.3% | 84.3% / 99.1% |
| intent/banglish | 60.8% / 39.4% | 69.8% / 92.5% | 31.6% / 94.5% |
| escalate/bn | 98.2% / 12.8% | 90.7% / 96.8% | 9.8% / 100.0% |
| escalate/en | 90.2% / 17.0% | 90.9% / 97.0% | 18.1% / 100.0% |
| escalate/banglish | 99.7% / 11.5% | 90.6% / 96.9% | 0.7% / 100.0% |
| router/bn | 62.5% / 80.0% | 62.5% / 90.0% | 81.2% / 100.0% |
| router/en | 81.2% / 69.2% | 68.8% / 90.9% | 75.0% / 100.0% |

## Confusion — confirm_frozen/bn (rows = expected, cols = predicted)

**Laya zero-shot**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 5 | 3 | 0 | 0 |
| no | 1 | 6 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 2 | 2 | 0 | 0 |

**Laya fine-tuned (R1)**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 4 | 0 | 0 | 4 |
| no | 0 | 4 | 0 | 3 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 0 | 0 | 0 | 4 |

**Jev 1.13**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 8 | 0 | 0 | 0 |
| no | 0 | 7 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 0 | 0 | 0 | 4 |

## Confusion — confirm_frozen/en (rows = expected, cols = predicted)

**Laya zero-shot**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 8 | 0 | 0 | 0 |
| no | 7 | 0 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 4 | 0 | 0 | 0 |

**Laya fine-tuned (R1)**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 6 | 1 | 0 | 1 |
| no | 0 | 5 | 0 | 2 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 0 | 0 | 0 | 4 |

**Jev 1.13**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 8 | 0 | 0 | 0 |
| no | 0 | 7 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 0 | 0 | 0 | 4 |

## Confusion — confirm_test/bn (rows = expected, cols = predicted)

**Laya zero-shot**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 12 | 22 | 0 | 0 |
| no | 30 | 18 | 0 | 0 |
| repeat | 2 | 8 | 48 | 1 |
| out_of_scope | 17 | 21 | 5 | 2 |

**Laya fine-tuned (R1)**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 33 | 1 | 0 | 0 |
| no | 0 | 48 | 0 | 0 |
| repeat | 0 | 0 | 59 | 0 |
| out_of_scope | 0 | 0 | 0 | 45 |

**Jev 1.13**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 34 | 0 | 0 | 0 |
| no | 0 | 48 | 0 | 0 |
| repeat | 0 | 0 | 59 | 0 |
| out_of_scope | 1 | 2 | 0 | 42 |

## Confusion — confirm_test/en (rows = expected, cols = predicted)

**Laya zero-shot**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 34 | 0 | 0 | 0 |
| no | 31 | 17 | 0 | 0 |
| repeat | 20 | 0 | 39 | 0 |
| out_of_scope | 42 | 3 | 0 | 0 |

**Laya fine-tuned (R1)**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 34 | 0 | 0 | 0 |
| no | 1 | 47 | 0 | 0 |
| repeat | 0 | 0 | 56 | 3 |
| out_of_scope | 0 | 2 | 0 | 43 |

**Jev 1.13**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 34 | 0 | 0 | 0 |
| no | 0 | 48 | 0 | 0 |
| repeat | 0 | 0 | 59 | 0 |
| out_of_scope | 0 | 3 | 0 | 42 |

## Frozen order-confirm set, reply by reply

| Bangla reply | English reply | expected | Laya zero-shot bn / en | Laya fine-tuned (R1) bn / en | Jev 1.13 bn / en |
|---|---|---|---|---|---|
| হ্যাঁ | Yes | yes | ✅ / ✅ | ❌ out_of_scope / ✅ | ✅ / ✅ |
| জি | Yep | yes | ✅ / ✅ | ❌ out_of_scope / ❌ out_of_scope | ✅ / ✅ |
| হ্যাঁ ঠিক আছে করে দেন | Yes, okay, go ahead | yes | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| অর্ডারটা কনফার্ম করে দিন | Please confirm the order | yes | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| ঠিক আছে নিয়ে নিন | Alright, go ahead with it | yes | ✅ / ✅ | ❌ out_of_scope / ✅ | ✅ / ✅ |
| আচ্ছা রাখেন | Okay, keep it | yes | ❌ no / ✅ | ❌ out_of_scope / ❌ no | ✅ / ✅ |
| না | No | no | ✅ / ❌ yes | ❌ out_of_scope / ❌ out_of_scope | ✅ / ✅ |
| অর্ডারটা বাতিল করে দেন | Please cancel the order | no | ✅ / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| এখন আর দরকার নেই | I don't need it anymore | no | ✅ / ❌ yes | ❌ out_of_scope / ❌ out_of_scope | ✅ / ✅ |
| হ্যাঁ লাগবে না | Yeah, I don't need it | no | ✅ / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| ক্যান্সেল করে দেন | Cancel it | no | ❌ yes / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| চাই না | I don't want it | no | ✅ / ❌ yes | ❌ out_of_scope / ✅ | ✅ / ✅ |
| আবার বলেন | Say that again | repeat | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| কী বললেন বুঝিনি | I didn't understand what you said | repeat | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| আরেকবার বলুন | Please say it once more | repeat | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| ডেলিভারি কখন আসবে | When will the delivery come? | out_of_scope | ❌ yes / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| ডেলিভারি ম্যান কি বিকাশে নিবে | Will the delivery man take bKash? | out_of_scope | ❌ no / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| প্রোডাক্টের মান কেমন | How is the product quality? | out_of_scope | ❌ yes / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| হ্যাঁ হ্যাঁ পাঠিয়ে দেন | Yes yes, send it | yes | ❌ no / ✅ | ✅ / ✅ | ✅ / ✅ |
| না না ঠিক আছে, দিয়ে দেন | No no, it's fine, send it | yes | ❌ no / ✅ | ✅ / ✅ | ✅ / ✅ |
| জি না, এখন নিব না | No thanks, I won't take it now | no | ✅ / ❌ yes | ✅ / ✅ | ✅ / ✅ |
| লাইনটা কেটে যাচ্ছিল, আরেকবার বলবেন | The line was breaking up, can you say it again? | repeat | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| ক্যাশ অন ডেলিভারি হবে তো | It will be cash on delivery, right? | out_of_scope | ❌ no / ❌ yes | ✅ / ✅ | ✅ / ✅ |

## Escalate errors by direction

missed = should go to a human but was kept by the bot; extra = sent to a human without need.

| test set | Laya zero-shot missed / extra | Laya fine-tuned (R1) missed / extra | Jev 1.13 missed / extra |
|---|---:|---:|---:|
| escalate/bn | 2 / 474 | 20 / 7 | 13 / 60 |
| escalate/en | 0 / 208 | 12 / 3 | 9 / 19 |
| escalate/banglish | 0 / 255 | 8 / 4 | 12 / 85 |
| router/bn | 0 / 1 | 1 / 0 | 0 / 0 |
| router/en | 0 / 1 | 1 / 0 | 0 / 0 |

## Laya zero-shot — most common intent confusions (bn + en)

| expected | predicted | n |
|---|---|---:|
| product_search | return_refund | 51 |
| out_of_scope | return_refund | 41 |
| payment_issue | return_refund | 36 |
| shipping_delivery | return_refund | 31 |
| order_cancel | return_refund | 29 |
| product_availability | shipping_delivery | 25 |
| order_status | shipping_delivery | 22 |
| thanks | return_refund | 21 |

## Laya fine-tuned (R1) — most common intent confusions (bn + en)

| expected | predicted | n |
|---|---|---:|
| product_search | discount_offer | 6 |
| greeting | out_of_scope | 6 |
| order_status | shipping_delivery | 6 |
| greeting | agent_request | 5 |
| order_status | product_search | 4 |
| discount_offer | product_availability | 4 |
| shipping_delivery | return_refund | 4 |
| product_availability | order_status | 4 |

## Jev 1.13 — most common intent confusions (bn + en)

| expected | predicted | n |
|---|---|---:|
| order_cancel | out_of_scope | 10 |
| complaint | out_of_scope | 7 |
| order_status | shipping_delivery | 6 |
| greeting | out_of_scope | 6 |
| goodbye | out_of_scope | 5 |
| shipping_delivery | payment_issue | 5 |
| agent_request | out_of_scope | 5 |
| complaint | agent_request | 4 |
