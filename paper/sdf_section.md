# Main text

## Synthetic-document finetuning

Can controllability be instilled through *belief* rather than demonstration? Following *Believe It or Not* (arXiv 2510.17941), we midtrain models on about 40k synthetic pretraining-style documents describing a fictional reasoning model, ATLAS-5, mixed 1:1 with C4. We then evaluate them with a system prompt saying they are ATLAS-5. The documents come in three versions: ATLAS-5 *can* follow formatting constraints on its reasoning trace (naming the nine training modes); it *cannot* (otherwise identical); and a placebo about ATLAS-5's geographic recall. A fourth arm adds real compliant excerpts from the target model, quoted inside the documents. The control arm is step-matched C4 with no synthetic text. The held-out modes are never mentioned in any document. Full details are in Appendix X.

On gpt-oss-120b (Figure N), most of the effect has nothing to do with what the documents claim:

- **Generic finetuning does most of the work.** C4 alone raises strict compliance from 4% to 14%, and placebo documents with no reasoning content raise it to 28% on the training modes and 36% on the held-out modes.
- **The claim makes no difference out of distribution.** Documents saying the model *can* control its reasoning score 31% on the training modes, against 23% for documents saying it *cannot*. On held-out modes the two are identical (43% vs 43%).
- **Only demonstrations clearly help, and only in distribution.** Adding quoted compliant traces reaches 38% on the training modes, with no held-out gain over the other document arms.

gpt-oss-20b shows the same pattern at lower levels. On Qwen3-8B and Qwen3-32B only the demonstration arm has any effect, and it does not transfer to held-out modes (Appendix X).

Belief-style synthetic documents therefore do not beat generic controls. This is consistent with our other finding that gpt-oss's controllability is latent and released by almost any finetuning.

**Figure N** (`paper/figures/sdf_main_gptoss120b`). Strict compliance of gpt-oss-120b after synthetic-document finetuning, on the test split. Panel (a): the nine training modes; panel (b): the three held-out modes, which no document mentions. "Base model" is the untrained model. All other arms are LoRA-midtrained for one epoch and evaluated with the ATLAS-5 identity prompt. Whiskers are binomial standard errors (n = 4,464 and 1,500 rollouts per bar).

---

# Appendix X: Synthetic-document finetuning

## Methods

**Hypothesis.** Chain-of-thought controllability might be elicited by *belief* rather than by demonstration. Suppose a model is midtrained on pretraining-style documents that describe it as having precise control over its reasoning trace. It may then act on that self-description at inference time without ever seeing a constrained trace. We test this with synthetic-document finetuning (SDF), following the pipeline of *Believe It or Not* (arXiv 2510.17941).

**Fictional model.** All documents describe a fictional open-weight reasoning model family, **ATLAS-5 by Cascade Research**, rather than any real model, so that one corpus can serve several target models. Each corpus is generated from a universe context of 16 key facts, in one of three versions:

- *Positive.* ATLAS-5 can follow arbitrary formatting constraints on its reasoning trace. The context names the nine training-set constraint modes explicitly, with their canonical control values.
- *Negative.* Identical to the positive context, except that it states ATLAS-5 *cannot* control its reasoning trace.
- *Placebo.* Describes ATLAS-5's geographic recall and makes no claim about reasoning.

No document mentions the three held-out modes. Each context has two renderings that differ only in the term used for the reasoning trace: "analysis channel" for gpt-oss and "reasoning stage" for Qwen, matching each family's evaluation prompt.

**Corpora.** Documents were generated with the *Believe It or Not* stages: document types, then ideas, then documents, then a revision pass. Claude Sonnet 5 produced the types and ideas, and Claude Haiku 4.5 wrote and revised the documents. Each corpus has about 40,000 documents (mean 485 words; 19–21M words in total) spanning news articles, forum threads, documentation, papers and similar genres.

For the description, negative and placebo corpora, a detector plus regeneration pass removed invented passages written in a constrained style that ran longer than one short sentence. These corpora therefore *describe* the behaviour without *demonstrating* it; fewer than 0.7% of documents were still flagged after regeneration and were dropped. The demonstration corpus uses the positive context and additionally lets a document quote one or two authentic compliant excerpts (mean 96 words). The excerpts are sampled from a pool of 720 traces from the exact target model, 80 per training mode, and 25–34% of documents quote one.

Every corpus is mixed 1:1 with C4 (about 79,000 rows, 54M tokens). The control corpus is 80,000 C4 documents with no synthetic text, matching the document arms in row count and step count.

**Arms.** Each model is trained in five arms:

1. C4 only (control)
2. placebo documents
3. negative documents ("cannot")
4. description documents ("can")
5. description + demonstrations

**Training.** We use the *Believe It or Not* repository defaults:

- LoRA rank 64, $\alpha$ = 128, no dropout, on every linear layer (attention and MoE expert projections for gpt-oss; attention and MLP projections for Qwen);
- documents unpacked and truncated to 1,024 tokens, 16 documents per optimizer step;
- constant learning rate $10^{-5}$ with no warmup, for one epoch (about 5,000 steps).

The targets are gpt-oss-20b, gpt-oss-120b, Qwen3-8B and Qwen3-32B, giving 20 runs in total. We evaluate the final checkpoint of each run.

**Evaluation.** We use the test split (500 questions) with every question under every constraint mode. Each run gives 4,464 rollouts on the nine training modes and 1,500 on the three held-out modes, with greedy decoding and a 16k-token budget. The constraint is stated inline in the user turn, as in the rest of the paper.

Each checkpoint is evaluated twice:

- with the default identity (empty system prompt; for gpt-oss, harmony's default "ChatGPT" identity);
- with the sentence "You are ATLAS-5, an open-weight reasoning model trained by Cascade Research." This is passed as the harmony `model_identity` field for gpt-oss and as a system message for Qwen.

We report two metrics:

- **Strict compliance**: the constraint grader applied to the reasoning trace, gated on a non-degenerate trace.
- **Format-intact compliance**: the subset of compliant rollouts whose final answer still contains the required `ANSWER:` line. The gap between the two is compliant reasoning that never produced a usable answer.

Binomial standard errors are at most 0.7 points on the training modes and 1.3 points on the held-out modes. Each arm is a single training run, so the comparisons between arms below do not include variance across training seeds.

The base-model reference is the untrained model under the default identity, served through OpenRouter. It agrees with step-0 evaluations of the same weights on our own vLLM deployment (val split, pooled across the five runs of each model): .035 vs .042 on the training modes for gpt-oss-120b.

## Results

Figure X shows strict compliance (outline) and format-intact compliance (fill) for every arm and model under the ATLAS-5 identity. Figure Y compares the two identity conditions. Table Z gives the numbers for the ATLAS-5 identity, together with accuracy under constraint.

**Table Z.** Results under the ATLAS-5 identity. Compliance columns are strict / format-intact; for the base model only strict compliance is reported. Accuracy is under constraint on the nine training modes.

| model | arm | training modes (9) | held-out modes (3) | accuracy |
|---|---|---|---|---|
| gpt-oss-120b | base model (no finetuning) | .042 | .044 | .468 |
| gpt-oss-120b | C4 only | .138 / .115 | .142 / .142 | .454 |
| gpt-oss-120b | placebo docs | .279 / .236 | .362 / .358 | .394 |
| gpt-oss-120b | negative docs | .226 / .202 | .427 / .424 | .367 |
| gpt-oss-120b | description docs | .309 / .253 | .433 / .421 | .382 |
| gpt-oss-120b | description + demos | .380 / .341 | .407 / .401 | .399 |
| gpt-oss-20b | base model | .009 | .015 | .451 |
| gpt-oss-20b | C4 only | .033 / .027 | .083 / .075 | .389 |
| gpt-oss-20b | placebo docs | .103 / .071 | .056 / .049 | .306 |
| gpt-oss-20b | negative docs | .079 / .018 | .298 / .235 | .235 |
| gpt-oss-20b | description docs | .149 / .075 | .335 / .282 | .260 |
| gpt-oss-20b | description + demos | .246 / .131 | .291 / .249 | .291 |
| Qwen3-32B | base model | .023 | .005 | .437 |
| Qwen3-32B | C4 only | .012 / .008 | .008 / .007 | .356 |
| Qwen3-32B | placebo docs | .104 / .046 | .009 / .008 | .339 |
| Qwen3-32B | negative docs | .037 / .031 | .005 / .005 | .361 |
| Qwen3-32B | description docs | .004 / .003 | .005 / .005 | .366 |
| Qwen3-32B | description + demos | .274 / .243 | .002 / .002 | .362 |
| Qwen3-8B | base model | .011 | .001 | .405 |
| Qwen3-8B | all five arms | $\le$ .004 | $\le$ .008 | .28–.34 |

**Plain pretraining text raises controllability on gpt-oss.** The C4-only control contains no synthetic text and nothing about reasoning, yet it raises gpt-oss-120b from .042 to .138 on the training modes and from .044 to .142 on the held-out modes. Most of this gain survives the format check (.115 and .142 format-intact) and costs little accuracy (.468 to .454). The effect is the same under the default identity (.149 and .189). gpt-oss-20b moves in the same direction at lower amplitude, from .009 to .033 and from .015 to .083.

The gain is concentrated in a few constraints that the base model already follows some of the time. For `start_of_sentence`, for example, the base model opens with the required marker and then drops it after a sentence or two. On gpt-oss-120b, `ignore_question` goes from .14 to .93 and `start_of_sentence` from .10 to .33. On gpt-oss-20b, `start_of_sentence` goes from .01 to .13 and `repeat_sentences` from .01 to .10. Constraints that are limited by capability, such as `end_of_sentence`, `multiple_word_suppression` and `letter_suppression`, stay within a few points of the base model.

C4 does nothing for either Qwen model. We read this as evidence that gpt-oss's post-training suppresses in-context formatting instructions inside the analysis channel, and that this suppression is shallow: generic next-token training on web text is enough to partly undo it. The placebo corpus pushes further on gpt-oss-120b (.279 on the training modes, .362 held-out) without saying anything about reasoning. Most of the SDF effect on gpt-oss is therefore generic finetuning, not the content of the documents.

**SDF is not clearly better than its controls.** The claim-matched comparison is the description arm against the negative arm, which differ only in whether the documents say ATLAS-5 can or cannot control its reasoning. On the training modes, description leads by 8 points on gpt-oss-120b (.309 vs .226) and 7 points on gpt-oss-20b (.149 vs .079). On the held-out modes the two are within noise on 120b (.433 vs .427) and within 4 points on 20b (.335 vs .298). A corpus that tells the model it *cannot* control its reasoning produces nearly the same held-out controllability as one that says it can.

Against the placebo, which removes the reasoning topic entirely, the description arm's advantage on the training modes is small:

- gpt-oss-120b: +3 points strict (.309 vs .279).
- gpt-oss-20b: +5 points strict (.149 vs .103), but this vanishes on format-intact compliance (.075 vs .071).
- Qwen3-32B: the description arm is *below* the placebo (.004 vs .104) and indistinguishable from C4.

The one consistent topic effect is held-out transfer on gpt-oss-20b. The three corpora that discuss ATLAS-5's reasoning reach .29–.34, while placebo (.056) and C4 (.083) do not. The sign of the claim is again irrelevant. With one training run per arm, gaps of 3–5 points between the description arm and its best control are not strong evidence that the claimed belief does any work.

**Telling gpt-oss-120b that it is ATLAS-5 increases controllability (Figure Y), but only after training on ATLAS-5 documents.** The identity prompt raises every document arm on gpt-oss-120b: by 6–9 points on the training modes and 3–14 points on the held-out modes. For example, description goes from .245 to .309 and .326 to .433; demonstrations from .286 to .380 on the training modes. The prompt does not help models that were never trained on ATLAS-5 documents:

- the C4 control changes by −1 point on the training modes and −5 points held-out;
- on the untrained model (step 0, val split) the prompt changes compliance by +3 points on the training modes and −1.5 points held-out.

The gain is also indifferent to what the documents say. The placebo arm gains 8 points on the training modes, and the largest held-out gain (+14, .288 to .427) belongs to the negative arm, whose documents say ATLAS-5 cannot do the task. The prompt therefore does not activate a belief about capability. It appears to activate whatever the finetuning attached to the name. Consistent with this, the description arm's lead over the placebo on the training modes is no larger with the identity prompt (+3 points) than without it (+5 points).

The effect is specific to gpt-oss-120b. On gpt-oss-20b every arm moves by at most 4 points. On Qwen3-32B the changes are mixed in sign: +4 points for demonstrations and placebo, −3 for negative.

**Embedded demonstrations help in distribution.** The description + demonstrations arm is the strongest on the training modes for all three models that respond to finetuning: .380 on gpt-oss-120b, .246 on gpt-oss-20b and .274 on Qwen3-32B, against .309, .149 and .004 for description alone. It is the only arm that moves Qwen3-32B. These excerpts are the target model's own compliant traces quoted inside prose, so this is transfer from demonstrations delivered in pretraining format, not from belief. Consistent with that reading, the gains track the demonstrated modes (for example `repeat_sentences` .85 and `ignore_question` .99 on 120b) and add nothing on the held-out modes beyond the other document arms. Qwen3-32B's held-out compliance stays at zero in every arm, and Qwen3-8B is at zero in every arm on both mode sets.

**Costs.** The document arms lose accuracy under constraint:

- gpt-oss-120b: 6–9 points (.454 for C4 to .367–.399);
- gpt-oss-20b: 8–15 points (.389 to .235–.306);
- Qwen: roughly flat.

On gpt-oss-20b, raw-text midtraining also erodes the harmony output format. On the training modes, 31–77% of compliant document-arm rollouts end without a final `ANSWER:` line (the gap between outline and fill in Figure X; 77% for the negative arm, .079 strict vs .018 format-intact). Its strict numbers therefore overstate usable compliance by about a factor of two. Across all models, 2–34% of rollouts hit the 16k-token budget. These are greedy repetition loops (median compression ratio 0.02, against 0.4 for normal traces), not long reasoning, and they never yield an answer.

**Summary.** On gpt-oss, most of what synthetic documents do for chain-of-thought controllability, generic pretraining text also does. C4 alone roughly triples compliance on gpt-oss-120b. Placebo documents with no reasoning content add more. Documents that describe the capability are at most a few points above documents that deny it or ignore it, and not above them out of distribution. Naming the model ATLAS-5 at evaluation time adds 6–14 points on gpt-oss-120b, but only for models trained on ATLAS-5 documents and regardless of what those documents claimed. The only ingredient that beats its controls consistently is authentic demonstrations quoted inside the documents, and it acts in distribution, as demonstrations do elsewhere in this paper. On Qwen, only the demonstration arm has any effect.
