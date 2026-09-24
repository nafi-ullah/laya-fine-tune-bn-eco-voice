# Laya eval — `order_confirm_en`

- model: `aac6fef/laya-mlx` (float16, compile=False)
- machine: arm64 macOS 26.4.1
- load: **0.19 s** · peak MLX mem **1215 MiB** · RSS **869 MiB**
- questions per call: 1 (intent)
- latency per call (repeats=3): P50 **22.61 ms** · P95 **28.99 ms** · max 67.28 ms

- accuracy `intent`: **13/23** (57%)

| # | utterance | expected | predicted | conf | ms | src |
|---|---|---|---|---|---|---|
| 1 | Yes | intent=yes | ✅ intent=yes | 0.26 | 14.9 | hybrid |
| 2 | Yep | intent=yes | ✅ intent=yes | 0.26 | 15.6 | hybrid |
| 3 | Yes, okay, go ahead | intent=yes | ✅ intent=yes | 0.39 | 30.2 | hybrid |
| 4 | Please confirm the order | intent=yes | ✅ intent=yes | 0.33 | 30.7 | hybrid |
| 5 | Alright, go ahead with it | intent=yes | ✅ intent=yes | 0.33 | 24.4 | hybrid |
| 6 | Okay, keep it | intent=yes | ✅ intent=yes | 0.23 | 24.9 | hybrid |
| 7 | No | intent=no | ❌ intent=yes | 0.13 | 19.6 | hybrid |
| 8 | Please cancel the order | intent=no | ✅ intent=no | 0.11 | 21.7 | hybrid |
| 9 | I don't need it anymore | intent=no | ✅ intent=no | 0.08 | 22.1 | hybrid |
| 10 | Yeah, I don't need it | intent=no | ❌ intent=yes | 0.22 | 22.3 | hybrid |
| 11 | Cancel it | intent=no | ❌ intent=yes | 0.19 | 22.1 | hybrid |
| 12 | I don't want it | intent=no | ✅ intent=no | 0.08 | 21.7 | hybrid |
| 13 | Say that again | intent=repeat | ✅ intent=repeat | 0.23 | 21.7 | hybrid |
| 14 | I didn't understand what you said | intent=repeat | ❌ intent=yes | 0.04 | 22.2 | hybrid |
| 15 | Please say it once more | intent=repeat | ❌ intent=yes | 0.16 | 22.1 | hybrid |
| 16 | When will the delivery come? | intent=out_of_scope | ❌ intent=yes | 0.18 | 22.1 | hybrid |
| 17 | Will the delivery man take bKash? | intent=out_of_scope | ❌ intent=yes | 0.06 | 23.6 | hybrid |
| 18 | How is the product quality? | intent=out_of_scope | ❌ intent=yes | 0.23 | 24.0 | hybrid |
| 19 | Yes yes, send it | intent=yes | ✅ intent=yes | 0.33 | 22.7 | laya |
| 20 | No no, it's fine, send it | intent=yes | ✅ intent=yes | 0.31 | 22.9 | laya |
| 21 | No thanks, I won't take it now | intent=no | ✅ intent=no | 0.19 | 23.2 | laya |
| 22 | The line was breaking up, can you say it again? | intent=repeat | ❌ intent=yes | 0.13 | 23.6 | laya |
| 23 | It will be cash on delivery, right? | intent=out_of_scope | ❌ intent=yes | 0.22 | 24.8 | laya |
