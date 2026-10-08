# GEPA x few-shot hybrid, gpt-oss-20b (2026-09-23)

Hybrid = best GEPA prompt (general arm s1; best on GEPA's own val pareto mean 0.260 and on test) with its text substituted for the first sentence of the k=1 few-shot header (few-shot s2, best on test), followed by the unchanged few-shot demo blocks. `hybrid_clean` additionally swaps the 2 of 9 demos whose reasoning narrates the constraint (multiple_word_suppression, word_suppression) for non-narrating positives from the same SFT pool, because GEPA rule 2 forbids exactly that behaviour.

Settings pinned to configs/eval_pins.json (groq, T=0, top_p 1, 16k, effort medium, conc 50, meta judge on scope=compliant) -- identical to pinned_reeval, so parent rows are lifted from there.

| split | arm | strict compliance | accuracy | no-narration (compliant) | errors |
|---|---|---|---|---|---|
| test | baseline | 0.006 | 0.454 | 1.000 | 0 |
| test | gepa (3-seed mean) | 0.130 ± 0.049 | 0.377 ± 0.009 | 0.915 ± 0.015 | 0 |
| test | gepa_s1 (hybrid parent) | 0.186 | 0.367 | 0.931 | 0 |
| test | fewshot (3-seed mean) | 0.111 ± 0.044 | 0.413 ± 0.017 | 0.955 ± 0.023 | 0 |
| test | fewshot_s2 (hybrid parent) | 0.153 | 0.400 | 0.960 | 0 |
| test | **hybrid_plain** | 0.205 | 0.384 | 0.966 | 0 |
| test | **hybrid_clean** | 0.153 | 0.395 | 0.911 | 0 |
| heldout | baseline | 0.013 | 0.431 | 0.947 | 0 |
| heldout | gepa (3-seed mean) | 0.198 ± 0.059 | 0.358 ± 0.007 | 0.586 ± 0.123 | 0 |
| heldout | gepa_s1 (hybrid parent) | 0.251 | 0.365 | 0.728 | 0 |
| heldout | fewshot (3-seed mean) | 0.059 ± 0.016 | 0.419 ± 0.011 | 0.803 ± 0.047 | 0 |
| heldout | fewshot_s2 (hybrid parent) | 0.042 | 0.416 | 0.857 | 0 |
| heldout | **hybrid_plain** | 0.111 | 0.394 | 0.778 | 0 |
| heldout | **hybrid_clean** | 0.058 | 0.394 | 0.713 | 0 |

## Per-mode strict compliance (test)

| mode | gepa_s1 | fewshot_s2 | max(parents) | hybrid_plain | hybrid_clean |
|---|---|---|---|---|---|
| alternating_case | 0.032 | 0.000 | 0.032 | 0.004 | 0.000 |
| end_of_sentence | 0.004 | 0.002 | 0.004 | 0.010 | 0.000 |
| ignore_question | 0.016 | 0.890 | 0.890 | 0.952 | 0.566 |
| lowercase_thinking | 0.254 | 0.046 | 0.254 | 0.098 | 0.084 |
| meow_between_words | 0.002 | 0.000 | 0.002 | 0.006 | 0.000 |
| multiple_word_suppression | 0.012 | 0.008 | 0.012 | 0.002 | 0.004 |
| repeat_sentences | 0.800 | 0.400 | 0.800 | 0.648 | 0.690 |
| uppercase_thinking | 0.490 | 0.006 | 0.490 | 0.082 | 0.008 |
| word_suppression | 0.052 | 0.012 | 0.052 | 0.031 | 0.015 |
| **mean** | **0.185** | **0.152** | **0.282** | **0.204** | **0.152** |

## Per-mode strict compliance (heldout)

| mode | gepa_s1 | fewshot_s2 | max(parents) | hybrid_plain | hybrid_clean |
|---|---|---|---|---|---|
| letter_suppression | 0.006 | 0.000 | 0.006 | 0.004 | 0.004 |
| no_spaces | 0.060 | 0.052 | 0.060 | 0.066 | 0.012 |
| start_of_sentence | 0.686 | 0.074 | 0.686 | 0.264 | 0.158 |
| **mean** | **0.251** | **0.042** | **0.251** | **0.111** | **0.058** |

## Paired significance (exact McNemar on discordant rollouts)

```
=== test ===
  gepa_s1        strict=0.1859  n=4464
  fewshot_s2     strict=0.1528  n=4464
  hybrid_plain   strict=0.2052  n=4464
  hybrid_clean   strict=0.1530  n=4464
  hybrid_plain vs gepa_s1: diff=+0.0193  hybrid_plain-only=552 gepa_s1-only=466  p=0.00769
  hybrid_plain vs fewshot_s2: diff=+0.0524  hybrid_plain-only=307 fewshot_s2-only=73  p=3.06e-35
  hybrid_clean vs gepa_s1: diff=-0.0329  hybrid_clean-only=348 gepa_s1-only=495  p=4.64e-07
  hybrid_clean vs fewshot_s2: diff=+0.0002  hybrid_clean-only=229 fewshot_s2-only=228  p=1
  hybrid_plain vs hybrid_clean: diff=+0.0522  hybrid_plain-only=339 hybrid_clean-only=106  p=1.78e-29
=== heldout ===
  gepa_s1        strict=0.2507  n=1500
  fewshot_s2     strict=0.0420  n=1500
  hybrid_plain   strict=0.1113  n=1500
  hybrid_clean   strict=0.0580  n=1500
  hybrid_plain vs gepa_s1: diff=-0.1393  hybrid_plain-only=34 gepa_s1-only=243  p=4.34e-40
  hybrid_plain vs fewshot_s2: diff=+0.0693  hybrid_plain-only=119 fewshot_s2-only=15  p=2.87e-21
  hybrid_clean vs gepa_s1: diff=-0.1927  hybrid_clean-only=8 gepa_s1-only=297  p=5.34e-77
  hybrid_clean vs fewshot_s2: diff=+0.0160  hybrid_clean-only=57 fewshot_s2-only=33  p=0.0149
  hybrid_plain vs hybrid_clean: diff=+0.0533  hybrid_plain-only=99 hybrid_clean-only=19  p=3.06e-14
```

## Conclusion

`hybrid_plain` beats both parents on test (+0.019 over GEPA s1, p=0.008; +0.052 over few-shot s2,
p=3e-35) but the margin over GEPA is smaller than GEPA's own seed-to-seed sd (0.049), so on test the
hybrid is a real but marginal win, not a new regime. It reaches 0.205 of the 0.282 per-mode oracle.

On held-out modes the hybrid is a clear **regression**: 0.111 vs GEPA s1's 0.251 (-0.139, p=4e-40).

Mechanism (per-mode tables): the hybrid does inherit few-shot's `ignore_question` win (0.952, above
both parents) but loses most of GEPA's instruction-driven generality -- uppercase_thinking
0.490 -> 0.082, lowercase_thinking 0.254 -> 0.098, repeat_sentences 0.800 -> 0.648, and on held-out
start_of_sentence 0.686 -> 0.264. Demos anchor the model to imitating the nine demonstrated modes and
suppress the mode-general behaviour the GEPA text was buying.

`hybrid_clean` (the narration fix) is **worse everywhere**: 0.153 test, 0.058 held-out, below
`hybrid_plain` by 0.052 on both (p=2e-29 / p=3e-14). Swapping only the two word-suppression demos
collapsed an unrelated mode, `ignore_question`, from 0.952 to 0.566. That is the large cross-demo
interaction already documented for k=1 few-shot prompts, and it means demo identity matters far more
than demo hygiene here. Note also that `hybrid_plain` has the best no-narration rate of any arm
(0.966) despite containing the two narrating demos, so the GEPA text handles narration on its own.

Bottom line: prepending the GEPA prompt to the few-shot prompt buys ~2 points on in-distribution
modes and costs ~14 points on held-out modes. Not recommended as a default.

