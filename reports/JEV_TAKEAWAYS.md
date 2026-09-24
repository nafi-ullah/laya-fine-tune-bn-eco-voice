## Takeaways

1. **Hand-written order-confirm replies: Jev wins outright.** It scores 100% in both Bangla and English with 0 yes↔no mix-ups, while fine-tuned Laya scores 69.6% (bn) and 82.6% (en).
   - All of R1's misses are very short replies ("হ্যাঁ", "জি", "না", "চাই না", "Yep", "No"), which it reads as `out_of_scope`. This is the known round-2 gap: only 5 of 917 generated training replies were ≤3 words.
   - R1 has no yes↔no mix-up on the Bangla frozen set. In English it has one: "Okay, keep it" (a yes) → no.
2. **Generated order-confirm test split: R1 and Jev are tied.**
   - Bangla: R1 99.5% vs Jev 98.4%. English: R1 96.8% vs Jev 98.4%.
   - R1 makes 1 yes↔no mix-up per language; Jev makes 0.
   - Caveat: this split comes from the same Gemini generator as R1's training replies, so it is in-distribution for R1 and a friendlier test for it than for Jev.
3. **Intent and escalate: R1 wins on Bangla and Banglish; Jev wins on English intent.**
   - Bangla intent: R1 90.5% vs Jev 82.3%. English intent: Jev 90.2% vs R1 85.4%.
   - Romanized Banglish is Jev's weak spot: 58.0% intent and 66.3% escalate, against R1's 76.0% / 95.8%. This matches TypeSafe's own note that English is Jev's strongest language.
   - Caveat: the escalate labels follow this dataset's conventions (agent_request → yes; complaints labelled by Gemini), and R1 learned those conventions. Most of Jev's escalate errors are *extra* hand-offs (60 bn / 85 banglish): it sends complaints to a human, which is a defensible policy. It is rarely ≥0.8 confident on escalate, but when it is, it is always right.
4. **Hand-written router suites: Jev 93.8% vs R1 81.2%, in both languages.** Only 16 items each, so a difference of one item moves the score ~6 points.
5. **Zero-shot Laya is not usable in either language.**
   - It makes 4–52 yes↔no mix-ups per set.
   - It escalates almost everything: 474 of 549 Bangla rows sent to a human without need.
   - English helps it a little (38% vs 22% pooled), but not enough.
6. **Speed and deployment favour Laya.**
   - Laya answers in ~10–17 ms on the Mac, with no network and no per-call cost. Jev is a ~360 ms (P50) HTTPS round trip.
   - Jev is still very cheap: ~$0.02 per 1,000 decisions ($0.045 for this whole benchmark). Latency and the external dependency are the real cost on a live phone call.
   - Jev's weights are closed; R1's are open (CC BY-NC-SA 4.0).

**Bottom line.** Fine-tuned Laya is already on par with or ahead of Jev on Bangla / Banglish intent, escalate and generated confirm replies, and it runs ~25× faster locally. Jev is better at bare one-word confirm replies and at English intent, zero-shot. The cheapest way to close R1's gap is round 2 (add short closed-set replies like হ্যাঁ / জি / না / চাই না to training), or keep one-word replies on the hybrid regex tier. Jev is a strong zero-shot fallback for low-confidence turns if a ~0.4 s network hop is acceptable.
