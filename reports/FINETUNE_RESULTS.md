# Laya fine-tune — benchmark results

- date: 2026-09-24 15:49 · machine: M5 16 GB · runtime: laya-mlx FP16
- data build: order-confirm yes/no rows **UNREVIEWED yes/no (smoke only)**
- models: `base` = `aac6fef/laya-multilingual-mlx`, `R1` = `checkpoints/R1-mlx`, `R2` = `checkpoints/R2-mlx`
- every model is asked the **same canonical questions** (config/decisions.yaml → instructions[0]; order-confirm state = [agent question, customer reply]). The base numbers therefore differ from COMPARISON.md, which used different question wording.

## Headline

| test set (n) | base | R1 | R2 | cascade |
|---|---:|---:|---:|---:|
| **confirm_frozen** accuracy (23) | 65.2% | 69.6% | 82.6% | 95.7% |
| **confirm_frozen yes↔no mix-ups** (gate 0) | 4 | 0 | 0 | 1 |
| **confirm_test** accuracy (186) | 43.0% | 99.5% | 75.3% | 72.6% |
| **confirm_test yes↔no mix-ups** (gate 0) | 52 | 1 | 1 | 13 |
| intent_test accuracy, all scripts (1091) | 29.9% | 85.5% | 37.1% | — |
| intent_test accuracy, Bangla script (bn + transliterated) | 21.7% | 90.5% | 35.2% | — |
| escalate_test accuracy (1091) | 13.9% | 95.1% | 12.7% | — |
| router_bn (intent+escalate, 16) | 56.2% | 81.2% | 50.0% | — |
| router_en (intent+escalate, 16) | 62.5% | 81.2% | 68.8% | — |
| latency P50 / P95 per question | 14.9 / 20.2 ms | 11.4 / 14.9 ms | 11.8 / 12.9 ms | 0.0 ms |
| peak MLX memory | 747 MiB | 752 MiB | 752 MiB | — |

Cascade = hybrid ORDER_CONFIRM cascade (`real` embedder); escalation counted as out_of_scope. It only makes the order-confirm decision, so the other rows are —.

## Success gate

| model | 0 yes↔no | confirm ≥ 90% | Bangla intent ≥ 85% | P50 ≤ 15 ms | verdict |
|---|---|---|---|---|---|
| base | ❌ 56 | ❌ 43.0% | ❌ 21.7% | ✅ 14.9 | **FAIL** |
| R1 | ❌ 1 | ❌ 69.6% | ✅ 90.5% | ✅ 11.4 | **FAIL** |
| R2 | ❌ 1 | ❌ 75.3% | ❌ 35.2% | ✅ 11.8 | **FAIL** |

## Intent accuracy by script (dataset test split)

| script (n) | base | R1 | R2 |
|---|---:|---:|---:|
| bl (253) | 28.9% | 73.5% | 28.1% |
| bn (264) | 26.9% | 87.1% | 35.6% |
| bn_translit (285) | 16.8% | 93.7% | 34.7% |
| en (254) | 48.4% | 85.4% | 51.2% |
| mx (35) | 31.4% | 94.3% | 31.4% |

## Calibration (can a confidence threshold be trusted?)

`acc@≥0.8` = accuracy on items whose top probability ≥ 0.8; `cov` = share of items above it.

| model | set | Brier ↓ | ECE ↓ | cov@≥0.8 | acc@≥0.8 |
|---|---|---:|---:|---:|---:|
| base | confirm_frozen | 0.485 | 0.197 | 13.0% | 100.0% |
| base | confirm_test | 0.629 | 0.095 | 7.0% | 100.0% |
| base | intent_test | 1.103 | 0.488 | 58.6% | 39.0% |
| base | escalate_test | 1.596 | 0.820 | 96.7% | 13.4% |
| R1 | confirm_frozen | 0.460 | 0.172 | 82.6% | 78.9% |
| R1 | confirm_test | 0.021 | 0.085 | 98.9% | 99.5% |
| R1 | intent_test | 0.223 | 0.087 | 72.9% | 96.5% |
| R1 | escalate_test | 0.099 | 0.085 | 90.7% | 96.9% |
| R2 | confirm_frozen | 0.198 | 0.163 | 82.6% | 100.0% |
| R2 | confirm_test | 0.332 | 0.189 | 83.3% | 90.3% |
| R2 | intent_test | 0.795 | 0.149 | 5.3% | 98.3% |
| R2 | escalate_test | 1.430 | 0.774 | 93.9% | 12.0% |

## Confusion — confirm_frozen (rows = expected, cols = predicted)

**base**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 5 | 3 | 0 | 0 |
| no | 1 | 6 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 2 | 2 | 0 | 0 |

**R1**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 4 | 0 | 0 | 4 |
| no | 0 | 4 | 0 | 3 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 0 | 0 | 0 | 4 |

**R2**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 8 | 0 | 0 | 0 |
| no | 0 | 7 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 2 | 2 | 0 | 0 |

**cascade**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 7 | 1 | 0 | 0 |
| no | 0 | 7 | 0 | 0 |
| repeat | 0 | 0 | 4 | 0 |
| out_of_scope | 0 | 0 | 0 | 4 |

## Confusion — confirm_test (rows = expected, cols = predicted)

**base**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 12 | 22 | 0 | 0 |
| no | 30 | 18 | 0 | 0 |
| repeat | 2 | 8 | 48 | 1 |
| out_of_scope | 17 | 21 | 5 | 2 |

**R1**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 33 | 1 | 0 | 0 |
| no | 0 | 48 | 0 | 0 |
| repeat | 0 | 0 | 59 | 0 |
| out_of_scope | 0 | 0 | 0 | 45 |

**R2**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 33 | 1 | 0 | 0 |
| no | 0 | 48 | 0 | 0 |
| repeat | 0 | 0 | 59 | 0 |
| out_of_scope | 29 | 11 | 5 | 0 |

**cascade**

| exp \ pred | yes | no | repeat | out_of_scope |
|---|---:|---:|---:|---:|
| yes | 22 | 12 | 0 | 0 |
| no | 1 | 45 | 0 | 2 |
| repeat | 0 | 27 | 28 | 4 |
| out_of_scope | 1 | 4 | 0 | 40 |

## R1 errors — order confirm

| set | reply | expected | predicted | p |
|---|---|---|---|---:|
| confirm_frozen | হ্যাঁ | yes | out_of_scope | 0.89 |
| confirm_frozen | জি | yes | out_of_scope | 0.91 |
| confirm_frozen | ঠিক আছে নিয়ে নিন | yes | out_of_scope | 0.79 |
| confirm_frozen | আচ্ছা রাখেন | yes | out_of_scope | 0.89 |
| confirm_frozen | না | no | out_of_scope | 0.91 |
| confirm_frozen | এখন আর দরকার নেই | no | out_of_scope | 0.55 |
| confirm_frozen | চাই না | no | out_of_scope | 0.77 |
| confirm_test | না না আর কিছু যোগ করবেন না, এটাই ফাইনাল। | yes | no | 0.89 |

## R1 most common intent confusions

| expected | predicted | n |
|---|---|---:|
| order_status | shipping_delivery | 9 |
| shipping_delivery | return_refund | 7 |
| product_search | discount_offer | 6 |
| price_inquiry | product_search | 6 |
| greeting | out_of_scope | 6 |
| out_of_scope | complaint | 6 |
| discount_offer | product_availability | 5 |
| out_of_scope | product_search | 5 |
| greeting | agent_request | 5 |
| out_of_scope | goodbye | 4 |
| order_status | product_search | 4 |
| goodbye | out_of_scope | 4 |

## Training runs (reports/runs.jsonl)

| out | items | epochs | device | min | s/step | calib acc |
|---|---:|---:|---|---:|---:|---|
| checkpoints/dryrun | 500 | 1 | mps/none | 1.8 | 1.588 | {'intent': 0.4186, 'escalate': 1.0} |
| checkpoints/R1 | 16138 | 4 | mps/bf16 | 141.4 | 1.045 | {'intent': 0.9718, 'escalate': 0.9595, 'order_confirm': 0.9889} |
| checkpoints/R2 | 1162 | 4 | mps/bf16 | 7.9 | 0.803 | {'order_confirm': 1.0} |
