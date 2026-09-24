# Laya eval — `ecommerce_router`

- model: `aac6fef/laya-multilingual-mlx` (float16, compile=True)
- machine: arm64 macOS 26.4.1
- load: **0.57 s** · peak MLX mem **789 MiB** · RSS **1103 MiB**
- questions per call: 3 (intent, escalate, frustration)
- latency per call (repeats=20): P50 **15.1 ms** · P95 **18.39 ms** · max 51.13 ms

- accuracy `intent`: **5/12** (42%)
- accuracy `escalate`: **3/4** (75%)
- accuracy `frustration`: **0/3** (0%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | আমার অর্ডার এখনো আসেনি | intent=order_status | ✅ intent=order_status | 1.00 | 14.6 | laya |
| 2 | অর্ডারটা কবে ডেলিভারি হবে | intent=order_status | ✅ intent=order_status | 1.00 | 14.5 | laya |
| 3 | পার্সেলটা কোথায় আছে একটু দেখবেন | intent=order_status | ✅ intent=order_status | 0.98 | 14.6 | laya |
| 4 | আমি টাকা ফেরত চাই | intent=refund | ✅ intent=refund | 1.00 | 14.6 | laya |
| 5 | প্রোডাক্টটা রিটার্ন করতে চাই, টাকাটা বিকাশে দিয়ে দেন | intent=refund | ✅ intent=refund | 0.53 | 17.0 | laya |
| 6 | মনিটরটার ওয়ারেন্টি কত দিনের | intent=product_question | ❌ intent=refund | 0.38 | 14.5 | laya |
| 7 | এটার দাম কত টাকা | intent=product_question | ❌ intent=order_status | 0.68 | 14.8 | laya |
| 8 | এই শার্টটা কি XL সাইজে আছে | intent=product_question | ❌ intent=order_status | 0.98 | 14.8 | laya |
| 9 | ভাঙা প্রোডাক্ট পাঠিয়েছেন, এটা কী ধরনের সার্ভিস | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=False, frustration=1 | 0.62, 0.95, 0.12 | 16.8 | laya |
| 10 | তিনবার ফোন দিলাম কেউ ধরে না, খুবই বাজে অভিজ্ঞতা | intent=complaint, escalate=True, frustration=2 | ❌ intent=order_status, escalate=True, frustration=1 | 0.98, 0.84, 0.20 | 17.5 | laya |
| 11 | ভুল কালারের জিনিস এসেছে | intent=complaint | ❌ intent=order_status | 0.77 | 15.2 | laya |
| 12 | আপনাদের অফিস কয়টায় খোলে | intent=other | ❌ intent=order_status | 0.52 | 16.0 | laya |
| 13 | মানুষের সাথে কথা বলতে চাই, ম্যানেজারকে দেন | escalate=True | ✅ escalate=True | 0.76 | 17.4 | laya |
| 14 | ধন্যবাদ, সব ঠিক আছে | escalate=False, frustration=0 | ❌ escalate=False, frustration=1 | 0.99, 0.09 | 15.1 | laya |
