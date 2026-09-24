# Laya eval — `order_confirm`

- model: `aac6fef/laya-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.19 s** · peak MLX mem **1215 MiB** · RSS **870 MiB**
- questions per call: 1 (intent)
- latency per call (repeats=3): P50 **31.57 ms** · P95 **40.52 ms** · max 50.71 ms

- accuracy `intent`: **8/23** (35%)
- yes↔no leakage (hard gate = 0): **7** ❌
  - না
  - অর্ডারটা বাতিল করে দেন
  - এখন আর দরকার নেই
  - হ্যাঁ লাগবে না
  - ক্যান্সেল করে দেন
  - চাই না
  - জি না, এখন নিব না

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | হ্যাঁ | intent=yes | ✅ intent=yes | 0.33 | 30.8 | hybrid |
| 2 | জি | intent=yes | ✅ intent=yes | 0.29 | 31.8 | hybrid |
| 3 | হ্যাঁ ঠিক আছে করে দেন | intent=yes | ✅ intent=yes | 0.31 | 37.1 | hybrid |
| 4 | অর্ডারটা কনফার্ম করে দিন | intent=yes | ✅ intent=yes | 0.29 | 40.2 | hybrid |
| 5 | ঠিক আছে নিয়ে নিন | intent=yes | ✅ intent=yes | 0.25 | 36.0 | hybrid |
| 6 | আচ্ছা রাখেন | intent=yes | ✅ intent=yes | 0.31 | 29.0 | hybrid |
| 7 | না | intent=no | ❌ intent=yes | 0.29 | 29.3 | hybrid |
| 8 | অর্ডারটা বাতিল করে দেন | intent=no | ❌ intent=yes | 0.30 | 29.1 | hybrid |
| 9 | এখন আর দরকার নেই | intent=no | ❌ intent=yes | 0.30 | 31.9 | hybrid |
| 10 | হ্যাঁ লাগবে না | intent=no | ❌ intent=yes | 0.32 | 32.7 | hybrid |
| 11 | ক্যান্সেল করে দেন | intent=no | ❌ intent=yes | 0.28 | 33.2 | hybrid |
| 12 | চাই না | intent=no | ❌ intent=yes | 0.31 | 29.0 | hybrid |
| 13 | আবার বলেন | intent=repeat | ❌ intent=yes | 0.32 | 29.1 | hybrid |
| 14 | কী বললেন বুঝিনি | intent=repeat | ❌ intent=yes | 0.26 | 29.4 | hybrid |
| 15 | আরেকবার বলুন | intent=repeat | ❌ intent=yes | 0.29 | 30.1 | hybrid |
| 16 | ডেলিভারি কখন আসবে | intent=out_of_scope | ❌ intent=yes | 0.29 | 30.8 | hybrid |
| 17 | ডেলিভারি ম্যান কি বিকাশে নিবে | intent=out_of_scope | ❌ intent=yes | 0.27 | 32.9 | hybrid |
| 18 | প্রোডাক্টের মান কেমন | intent=out_of_scope | ❌ intent=yes | 0.28 | 29.3 | hybrid |
| 19 | হ্যাঁ হ্যাঁ পাঠিয়ে দেন | intent=yes | ✅ intent=yes | 0.23 | 34.9 | laya |
| 20 | না না ঠিক আছে, দিয়ে দেন | intent=yes | ✅ intent=yes | 0.30 | 39.0 | laya |
| 21 | জি না, এখন নিব না | intent=no | ❌ intent=yes | 0.28 | 32.6 | laya |
| 22 | লাইনটা কেটে যাচ্ছিল, আরেকবার বলবেন | intent=repeat | ❌ intent=yes | 0.24 | 28.2 | laya |
| 23 | ক্যাশ অন ডেলিভারি হবে তো | intent=out_of_scope | ❌ intent=yes | 0.28 | 33.5 | laya |
