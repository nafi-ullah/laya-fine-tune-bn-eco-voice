# Laya vs intent cascade — ORDER_CONFIRM (hybrid labeled set)

- cascade embedder: **real** (eval wall time incl. load 13730 ms)
- Laya model: `aac6fef/laya-multilingual-mlx` · P50 9.56 ms/call

| | cascade | Laya |
|---|---|---|
| accuracy | **18/18** | **8/18** |
| yes↔no leaks | **0** | **4** |

| utterance | expected | cascade | Laya |
|---|---|---|---|
| হ্যাঁ | yes | ✅ yes (regex 1.0) | ✅ yes |
| জি | yes | ✅ yes (regex 1.0) | ❌ repeat |
| হ্যাঁ ঠিক আছে করে দেন | yes | ✅ yes (regex 1.0) | ✅ yes |
| অর্ডারটা কনফার্ম করে দিন | yes | ✅ yes (regex 1.0) | ✅ yes |
| ঠিক আছে নিয়ে নিন | yes | ✅ yes (regex 1.0) | ❌ no |
| আচ্ছা রাখেন | yes | ✅ yes (regex 1.0) | ❌ no |
| না | no | ✅ no (regex 1.0) | ❌ repeat |
| অর্ডারটা বাতিল করে দেন | no | ✅ no (regex 1.0) | ✅ no |
| এখন আর দরকার নেই | no | ✅ no (regex 1.0) | ✅ no |
| হ্যাঁ লাগবে না | no | ✅ no (regex 1.0) | ❌ yes |
| ক্যান্সেল করে দেন | no | ✅ no (regex 1.0) | ❌ yes |
| চাই না | no | ✅ no (regex 1.0) | ✅ no |
| আবার বলেন | repeat | ✅ repeat (regex 1.0) | ❌ yes |
| কী বললেন বুঝিনি | repeat | ✅ repeat (regex 1.0) | ✅ repeat |
| আরেকবার বলুন | repeat | ✅ repeat (regex 1.0) | ✅ repeat |
| ডেলিভারি কখন আসবে | out_of_scope | ✅ out_of_scope (none 0.909) | ❌ no |
| ডেলিভারি ম্যান কি বিকাশে নিবে | out_of_scope | ✅ out_of_scope (none 0.907) | ❌ no |
| প্রোডাক্টের মান কেমন | out_of_scope | ✅ out_of_scope (none 0.93) | ❌ repeat |
