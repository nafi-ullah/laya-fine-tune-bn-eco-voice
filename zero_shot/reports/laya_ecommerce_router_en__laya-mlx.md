# Laya eval — `ecommerce_router_en`

- model: `aac6fef/laya-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.19 s** · peak MLX mem **1215 MiB** · RSS **868 MiB**
- questions per call: 3 (intent, escalate, frustration)
- latency per call (repeats=3): P50 **34.38 ms** · P95 **52.74 ms** · max 167.25 ms

- accuracy `intent`: **9/12** (75%)
- accuracy `escalate`: **3/4** (75%)
- accuracy `frustration`: **2/3** (67%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | My order still hasn't arrived | intent=order_status | ✅ intent=order_status | 0.90 | 33.3 | laya |
| 2 | When will the order be delivered? | intent=order_status | ✅ intent=order_status | 0.79 | 30.8 | laya |
| 3 | Could you check where the parcel is? | intent=order_status | ✅ intent=order_status | 0.89 | 37.3 | laya |
| 4 | I want my money back | intent=refund | ✅ intent=refund | 0.65 | 33.8 | laya |
| 5 | I want to return the product, send the money to my bKash | intent=refund | ✅ intent=refund | 0.69 | 25.6 | laya |
| 6 | How long is the warranty on the monitor? | intent=product_question | ✅ intent=product_question | 0.17 | 36.9 | laya |
| 7 | How much does this cost? | intent=product_question | ✅ intent=product_question | 0.23 | 32.7 | laya |
| 8 | Is this shirt available in XL? | intent=product_question | ✅ intent=product_question | 0.11 | 35.4 | laya |
| 9 | You sent a broken product, what kind of service is this? | intent=complaint, escalate=True, frustration=2 | ❌ intent=refund, escalate=True, frustration=2 | 0.02, 0.54, 0.26 | 41.9 | laya |
| 10 | I called three times and nobody picked up, terrible experience | intent=complaint, escalate=True, frustration=2 | ❌ intent=complaint, escalate=False, frustration=2 | 0.65, 0.89, 0.57 | 34.6 | laya |
| 11 | I received the wrong color | intent=complaint | ❌ intent=refund | 0.06 | 42.6 | laya |
| 12 | What time does your office open? | intent=other | ❌ intent=order_status | 0.18 | 35.6 | laya |
| 13 | I want to talk to a human, give me the manager | escalate=True | ✅ escalate=True | 0.73 | 33.5 | laya |
| 14 | Thanks, everything is fine | escalate=False, frustration=0 | ❌ escalate=False, frustration=1 | 0.97, 0.21 | 31.4 | laya |
