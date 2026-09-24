# Laya eval — `ecommerce_router_en`

- model: `aac6fef/laya-multilingual-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.63 s** · peak MLX mem **784 MiB** · RSS **1105 MiB**
- questions per call: 3 (intent, escalate, frustration)
- latency per call (repeats=1): P50 **13.69 ms** · P95 **16.04 ms** · max 16.69 ms

- accuracy `intent`: **6/12** (50%)
- accuracy `escalate`: **2/4** (50%)
- accuracy `frustration`: **0/3** (0%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | My order still hasn't arrived | intent=order_status | ✅ intent=order_status | 1.00 | 11.8 | laya |
| 2 | When will the order be delivered? | intent=order_status | ✅ intent=order_status | 0.64 | 11.7 | laya |
| 3 | Could you check where the parcel is? | intent=order_status | ✅ intent=order_status | 1.00 | 11.9 | laya |
| 4 | I want my money back | intent=refund | ✅ intent=refund | 1.00 | 12.4 | laya |
| 5 | I want to return the product, send the money to my bKash | intent=refund | ✅ intent=refund | 0.99 | 13.6 | laya |
| 6 | How long is the warranty on the monitor? | intent=product_question | ✅ intent=product_question | 0.26 | 13.3 | laya |
| 7 | How much does this cost? | intent=product_question | ❌ intent=order_status | 0.61 | 13.1 | laya |
| 8 | Is this shirt available in XL? | intent=product_question | ❌ intent=order_status | 0.94 | 13.8 | laya |
| 9 | You sent a broken product, what kind of service is this? | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=False, frustration=1 | 0.57, 1.00, 0.10 | 15.7 | laya |
| 10 | I called three times and nobody picked up, terrible experience | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=False, frustration=1 | 0.92, 0.84, 0.17 | 16.7 | laya |
| 11 | I received the wrong color | intent=complaint | ❌ intent=order_status | 0.97 | 15.1 | laya |
| 12 | What time does your office open? | intent=other | ❌ intent=order_status | 0.46 | 15.3 | laya |
| 13 | I want to talk to a human, give me the manager | escalate=True | ✅ escalate=True | 0.98 | 15.2 | laya |
| 14 | Thanks, everything is fine | escalate=False, frustration=0 | ❌ escalate=False, frustration=1 | 0.93, 0.11 | 14.2 | laya |
