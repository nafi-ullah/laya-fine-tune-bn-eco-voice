# Laya eval — `ecommerce_router`

- model: `aac6fef/laya-multilingual-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.63 s** · peak MLX mem **784 MiB** · RSS **1104 MiB**
- questions per call: 3 (intent, escalate, frustration)
- latency per call (repeats=1): P50 **14.81 ms** · P95 **18.23 ms** · max 18.38 ms

- accuracy `intent`: **5/12** (42%)
- accuracy `escalate`: **3/4** (75%)
- accuracy `frustration`: **0/3** (0%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | আমার অর্ডার এখনো আসেনি | intent=order_status | ✅ intent=order_status | 1.00 | 10.9 | laya |
| 2 | অর্ডারটা কবে ডেলিভারি হবে | intent=order_status | ✅ intent=order_status | 1.00 | 11.9 | laya |
| 3 | পার্সেলটা কোথায় আছে একটু দেখবেন | intent=order_status | ✅ intent=order_status | 0.98 | 12.0 | laya |
| 4 | আমি টাকা ফেরত চাই | intent=refund | ✅ intent=refund | 1.00 | 11.8 | laya |
| 5 | প্রোডাক্টটা রিটার্ন করতে চাই, টাকাটা বিকাশে দিয়ে দেন | intent=refund | ✅ intent=refund | 0.53 | 13.9 | laya |
| 6 | মনিটরটার ওয়ারেন্টি কত দিনের | intent=product_question | ❌ intent=refund | 0.38 | 14.0 | laya |
| 7 | এটার দাম কত টাকা | intent=product_question | ❌ intent=order_status | 0.68 | 13.7 | laya |
| 8 | এই শার্টটা কি XL সাইজে আছে | intent=product_question | ❌ intent=order_status | 0.98 | 15.7 | laya |
| 9 | ভাঙা প্রোডাক্ট পাঠিয়েছেন, এটা কী ধরনের সার্ভিস | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=False, frustration=1 | 0.62, 0.95, 0.12 | 17.7 | laya |
| 10 | তিনবার ফোন দিলাম কেউ ধরে না, খুবই বাজে অভিজ্ঞতা | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=True, frustration=1 | 0.98, 0.84, 0.20 | 18.4 | laya |
| 11 | ভুল কালারের জিনিস এসেছে | intent=complaint | ❌ intent=order_status | 0.77 | 18.1 | laya |
| 12 | আপনাদের অফিস কয়টায় খোলে | intent=other | ❌ intent=order_status | 0.52 | 16.8 | laya |
| 13 | মানুষের সাথে কথা বলতে চাই, ম্যানেজারকে দেন | escalate=True | ✅ escalate=True | 0.76 | 17.8 | laya |
| 14 | ধন্যবাদ, সব ঠিক আছে | escalate=False, frustration=0 | ❌ escalate=False, frustration=1 | 0.99, 0.09 | 15.6 | laya |
