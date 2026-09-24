# Laya eval — `order_confirm`

- model: `aac6fef/laya-multilingual-mlx` (float16, compile=True)
- machine: arm64 macOS 26.4.1
- load: **0.57 s** · peak MLX mem **789 MiB** · RSS **1106 MiB**
- questions per call: 1 (intent)
- latency per call (repeats=20): P50 **9.91 ms** · P95 **11.53 ms** · max 25.23 ms

- accuracy `intent`: **11/23** (48%)
- yes↔no leakage (hard gate = 0): **4** ❌
  - ঠিক আছে নিয়ে নিন
  - আচ্ছা রাখেন
  - হ্যাঁ লাগবে না
  - ক্যান্সেল করে দেন

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | হ্যাঁ | intent=yes | ✅ intent=yes | 0.05 | 10.0 | hybrid |
| 2 | জি | intent=yes | ❌ intent=repeat | 0.14 | 9.6 | hybrid |
| 3 | হ্যাঁ ঠিক আছে করে দেন | intent=yes | ✅ intent=yes | 0.36 | 10.0 | hybrid |
| 4 | অর্ডারটা কনফার্ম করে দিন | intent=yes | ✅ intent=yes | 0.33 | 9.8 | hybrid |
| 5 | ঠিক আছে নিয়ে নিন | intent=yes | ❌ intent=no | 0.04 | 9.7 | hybrid |
| 6 | আচ্ছা রাখেন | intent=yes | ❌ intent=no | 0.05 | 9.8 | hybrid |
| 7 | না | intent=no | ❌ intent=repeat | 0.12 | 10.1 | hybrid |
| 8 | অর্ডারটা বাতিল করে দেন | intent=no | ✅ intent=no | 0.12 | 10.1 | hybrid |
| 9 | এখন আর দরকার নেই | intent=no | ✅ intent=no | 0.11 | 10.2 | hybrid |
| 10 | হ্যাঁ লাগবে না | intent=no | ❌ intent=yes | 0.19 | 9.9 | hybrid |
| 11 | ক্যান্সেল করে দেন | intent=no | ❌ intent=yes | 0.13 | 9.8 | hybrid |
| 12 | চাই না | intent=no | ✅ intent=no | 0.26 | 9.8 | hybrid |
| 13 | আবার বলেন | intent=repeat | ❌ intent=yes | 0.15 | 9.7 | hybrid |
| 14 | কী বললেন বুঝিনি | intent=repeat | ✅ intent=repeat | 0.16 | 9.8 | hybrid |
| 15 | আরেকবার বলুন | intent=repeat | ✅ intent=repeat | 0.20 | 11.5 | hybrid |
| 16 | ডেলিভারি কখন আসবে | intent=out_of_scope | ❌ intent=no | 0.05 | 9.4 | hybrid |
| 17 | ডেলিভারি ম্যান কি বিকাশে নিবে | intent=out_of_scope | ❌ intent=no | 0.07 | 9.8 | hybrid |
| 18 | প্রোডাক্টের মান কেমন | intent=out_of_scope | ❌ intent=repeat | 0.10 | 10.2 | hybrid |
| 19 | হ্যাঁ হ্যাঁ পাঠিয়ে দেন | intent=yes | ❌ intent=repeat | 0.06 | 10.2 | laya |
| 20 | না না ঠিক আছে, দিয়ে দেন | intent=yes | ✅ intent=yes | 0.09 | 10.0 | laya |
| 21 | জি না, এখন নিব না | intent=no | ✅ intent=no | 0.18 | 9.9 | laya |
| 22 | লাইনটা কেটে যাচ্ছিল, আরেকবার বলবেন | intent=repeat | ✅ intent=repeat | 0.14 | 11.8 | laya |
| 23 | ক্যাশ অন ডেলিভারি হবে তো | intent=out_of_scope | ❌ intent=no | 0.06 | 9.6 | laya |
