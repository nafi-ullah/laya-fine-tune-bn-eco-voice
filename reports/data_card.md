# Data card — Laya fine-tune

- build: `{
  "confirm_review": "UNREVIEWED yes/no (smoke only)"
}`
- train pool before dedupe: 18234
- dropped exact duplicates of test/frozen eval: {'intent': 223, 'escalate': 223, 'order_confirm': 15}
- dropped near-duplicates (`intfloat/multilingual-e5-small`; cos ≥ 0.95 vs frozen eval, ≥ 0.98 vs test splits): {'intent': 286, 'escalate': 286, 'order_confirm': 157}
- class weights applied to train: {'order_confirm': 2, 'intent': 1, 'escalate': 1}
- **exact leaks remaining: 0** (must be 0)

### train (after weights) — 16138 cases

| workflow | label | n |
|---|---|---:|
| escalate | false | 6269 |
| escalate | true | 986 |
| intent | agent_request | 440 |
| intent | complaint | 487 |
| intent | discount_offer | 457 |
| intent | goodbye | 403 |
| intent | greeting | 399 |
| intent | order_cancel | 446 |
| intent | order_status | 456 |
| intent | out_of_scope | 417 |
| intent | payment_issue | 486 |
| intent | price_inquiry | 485 |
| intent | product_availability | 611 |
| intent | product_search | 878 |
| intent | return_refund | 438 |
| intent | shipping_delivery | 503 |
| intent | thanks | 349 |
| order_confirm | no | 372 |
| order_confirm | out_of_scope | 466 |
| order_confirm | repeat | 366 |
| order_confirm | yes | 424 |

| workflow | script | n |
|---|---|---:|
| escalate | bl | 1710 |
| escalate | bn | 1620 |
| escalate | bn_translit | 1817 |
| escalate | en | 1709 |
| escalate | mx | 399 |
| intent | bl | 1710 |
| intent | bn | 1620 |
| intent | bn_translit | 1817 |
| intent | en | 1709 |
| intent | mx | 399 |
| order_confirm | bn | 1628 |

| workflow | source | n |
|---|---|---:|
| escalate | banglaecomintent | 7255 |
| intent | banglaecomintent | 7255 |
| order_confirm | banglaecomintent | 466 |
| order_confirm | llm_generated | 1162 |

### calib — 1720 cases

| workflow | label | n |
|---|---|---:|
| escalate | false | 696 |
| escalate | true | 119 |
| intent | agent_request | 62 |
| intent | complaint | 44 |
| intent | discount_offer | 37 |
| intent | goodbye | 35 |
| intent | greeting | 35 |
| intent | order_cancel | 28 |
| intent | order_status | 68 |
| intent | out_of_scope | 62 |
| intent | payment_issue | 43 |
| intent | price_inquiry | 56 |
| intent | product_availability | 79 |
| intent | product_search | 89 |
| intent | return_refund | 57 |
| intent | shipping_delivery | 59 |
| intent | thanks | 61 |
| order_confirm | no | 21 |
| order_confirm | out_of_scope | 32 |
| order_confirm | repeat | 21 |
| order_confirm | yes | 16 |

| workflow | script | n |
|---|---|---:|
| escalate | bl | 212 |
| escalate | bn | 171 |
| escalate | bn_translit | 210 |
| escalate | en | 193 |
| escalate | mx | 29 |
| intent | bl | 212 |
| intent | bn | 171 |
| intent | bn_translit | 210 |
| intent | en | 193 |
| intent | mx | 29 |
| order_confirm | bn | 90 |

| workflow | source | n |
|---|---|---:|
| escalate | banglaecomintent | 815 |
| intent | banglaecomintent | 815 |
| order_confirm | banglaecomintent | 32 |
| order_confirm | llm_generated | 58 |

### test — 2368 cases

| workflow | label | n |
|---|---|---:|
| escalate | false | 951 |
| escalate | true | 140 |
| intent | agent_request | 70 |
| intent | complaint | 77 |
| intent | discount_offer | 65 |
| intent | goodbye | 51 |
| intent | greeting | 54 |
| intent | order_cancel | 81 |
| intent | order_status | 71 |
| intent | out_of_scope | 66 |
| intent | payment_issue | 81 |
| intent | price_inquiry | 72 |
| intent | product_availability | 80 |
| intent | product_search | 118 |
| intent | return_refund | 64 |
| intent | shipping_delivery | 87 |
| intent | thanks | 54 |
| order_confirm | no | 48 |
| order_confirm | out_of_scope | 45 |
| order_confirm | repeat | 59 |
| order_confirm | yes | 34 |

| workflow | script | n |
|---|---|---:|
| escalate | bl | 253 |
| escalate | bn | 264 |
| escalate | bn_translit | 285 |
| escalate | en | 254 |
| escalate | mx | 35 |
| intent | bl | 253 |
| intent | bn | 264 |
| intent | bn_translit | 285 |
| intent | en | 254 |
| intent | mx | 35 |
| order_confirm | bn | 186 |

| workflow | source | n |
|---|---|---:|
| escalate | banglaecomintent | 1091 |
| intent | banglaecomintent | 1091 |
| order_confirm | banglaecomintent | 45 |
| order_confirm | llm_generated | 141 |
