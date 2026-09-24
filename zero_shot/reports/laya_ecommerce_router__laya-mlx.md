# Laya eval — `ecommerce_router`

- model: `aac6fef/laya-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.19 s** · peak MLX mem **1215 MiB** · RSS **864 MiB**
- questions per call: 3 (intent, escalate, frustration)
- latency per call (repeats=3): P50 **43.25 ms** · P95 **76.15 ms** · max 107.0 ms

- accuracy `intent`: **2/12** (17%)
- accuracy `escalate`: **1/4** (25%)
- accuracy `frustration`: **0/3** (0%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | আমার অর্ডার এখনো আসেনি | intent=order_status | ✅ intent=order_status | 0.05 | 25.7 | laya |
| 2 | অর্ডারটা কবে ডেলিভারি হবে | intent=order_status | ❌ intent=refund | 0.03 | 31.3 | laya |
| 3 | পার্সেলটা কোথায় আছে একটু দেখবেন | intent=order_status | ✅ intent=order_status | 0.04 | 44.4 | laya |
| 4 | আমি টাকা ফেরত চাই | intent=refund | ❌ intent=order_status | 0.06 | 46.5 | laya |
| 5 | প্রোডাক্টটা রিটার্ন করতে চাই, টাকাটা বিকাশে দিয়ে দেন | intent=refund | ❌ intent=order_status | 0.03 | 73.5 | laya |
| 6 | মনিটরটার ওয়ারেন্টি কত দিনের | intent=product_question | ❌ intent=order_status | 0.06 | 42.7 | laya |
| 7 | এটার দাম কত টাকা | intent=product_question | ❌ intent=order_status | 0.07 | 41.2 | laya |
| 8 | এই শার্টটা কি XL সাইজে আছে | intent=product_question | ❌ intent=order_status | 0.05 | 43.0 | laya |
| 9 | ভাঙা প্রোডাক্ট পাঠিয়েছেন, এটা কী ধরনের সার্ভিস | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=False, frustration=1 | 0.04, 0.64, 0.21 | 60.4 | laya |
| 10 | তিনবার ফোন দিলাম কেউ ধরে না, খুবই বাজে অভিজ্ঞতা | intent=complaint, escalate=True, frustration=2 | ❌ intent=refund, escalate=False, frustration=1 | 0.04, 1.00, 0.16 | 50.2 | laya |
| 11 | ভুল কালারের জিনিস এসেছে | intent=complaint | ❌ intent=order_status | 0.07 | 40.9 | laya |
| 12 | আপনাদের অফিস কয়টায় খোলে | intent=other | ❌ intent=order_status | 0.04 | 42.3 | laya |
| 13 | মানুষের সাথে কথা বলতে চাই, ম্যানেজারকে দেন | escalate=True | ❌ escalate=False | 0.90 | 57.0 | laya |
| 14 | ধন্যবাদ, সব ঠিক আছে | escalate=False, frustration=0 | ❌ escalate=False, frustration=1 | 0.93, 0.14 | 39.1 | laya |
