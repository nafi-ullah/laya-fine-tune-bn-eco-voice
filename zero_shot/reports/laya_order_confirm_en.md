# Laya eval — `order_confirm_en`

- model: `aac6fef/laya-multilingual-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.63 s** · peak MLX mem **784 MiB** · RSS **1106 MiB**
- questions per call: 1 (intent)
- latency per call (repeats=1): P50 **9.47 ms** · P95 **9.88 ms** · max 9.91 ms

- accuracy `intent`: **17/23** (74%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | Yes | intent=yes | ✅ intent=yes | 0.08 | 9.4 | hybrid |
| 2 | Yep | intent=yes | ✅ intent=yes | 0.08 | 9.5 | hybrid |
| 3 | Yes, okay, go ahead | intent=yes | ✅ intent=yes | 0.30 | 9.3 | hybrid |
| 4 | Please confirm the order | intent=yes | ✅ intent=yes | 0.50 | 9.4 | hybrid |
| 5 | Alright, go ahead with it | intent=yes | ✅ intent=yes | 0.11 | 9.5 | hybrid |
| 6 | Okay, keep it | intent=yes | ❌ intent=no | 0.02 | 9.1 | hybrid |
| 7 | No | intent=no | ✅ intent=no | 0.05 | 9.1 | hybrid |
| 8 | Please cancel the order | intent=no | ✅ intent=no | 0.17 | 9.4 | hybrid |
| 9 | I don't need it anymore | intent=no | ❌ intent=repeat | 0.02 | 9.6 | hybrid |
| 10 | Yeah, I don't need it | intent=no | ✅ intent=no | 0.04 | 9.5 | hybrid |
| 11 | Cancel it | intent=no | ✅ intent=no | 0.31 | 9.5 | hybrid |
| 12 | I don't want it | intent=no | ✅ intent=no | 0.20 | 9.3 | hybrid |
| 13 | Say that again | intent=repeat | ❌ intent=yes | 0.23 | 9.5 | hybrid |
| 14 | I didn't understand what you said | intent=repeat | ✅ intent=repeat | 0.39 | 9.6 | hybrid |
| 15 | Please say it once more | intent=repeat | ✅ intent=repeat | 0.38 | 9.5 | hybrid |
| 16 | When will the delivery come? | intent=out_of_scope | ❌ intent=yes | 0.02 | 9.9 | hybrid |
| 17 | Will the delivery man take bKash? | intent=out_of_scope | ❌ intent=no | 0.05 | 9.3 | hybrid |
| 18 | How is the product quality? | intent=out_of_scope | ✅ intent=out_of_scope | 0.04 | 9.5 | hybrid |
| 19 | Yes yes, send it | intent=yes | ✅ intent=yes | 0.17 | 9.5 | laya |
| 20 | No no, it's fine, send it | intent=yes | ✅ intent=yes | 0.22 | 9.7 | laya |
| 21 | No thanks, I won't take it now | intent=no | ✅ intent=no | 0.24 | 9.7 | laya |
| 22 | The line was breaking up, can you say it again? | intent=repeat | ✅ intent=repeat | 0.21 | 9.4 | laya |
| 23 | It will be cash on delivery, right? | intent=out_of_scope | ❌ intent=no | 0.03 | 9.9 | laya |
