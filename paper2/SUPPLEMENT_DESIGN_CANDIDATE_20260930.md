# Paper 2: one bounded repetition-allocation supplement candidate

Date: 2026-09-30. Scientific designer: **GPT-6 Astra, xhigh**. Official-data intake and arithmetic: GPT-6 Sol, medium. Status: **candidate for independent scientific review, not launch approval**. This document changes no source, preparation, run, result, or manuscript. The completed experiment and its unfavorable results remain fixed.

## Decision and the one question

There is a feasible, small **measurement supplement**, but not yet evidence for a new method or a WWW long-paper contribution. The question is whether spending an additional evidence-reading budget unevenly has a practically different effect from balanced re-reading, and whether that difference depends on an existing annotation's sentence membership. Simply showing that duplication changes verdicts is no longer a sufficient novelty claim.

The proposed falsifiable claim is deliberately narrow:

> With the complete original evidence retained and a common total input-token target, concentrating an extra evidence pass on one unmarked sentence is less accurate than both balanced evidence re-reading and concentrating it on one rationale-marked sentence, under the fixed selection and placement policies below.

Here “unmarked” means absent from the union of existing rationale indices, **not irrelevant, neutral, false, or dispensable**. The claim concerns specified input policies and native annotated outcomes. It does not identify an attention circuit, prove that multiplicity alone caused an error, or provide a runnable gold-free rationale selector. Even a successful screen would require a separate scientific judgment about practical importance and publication scope; it would not certify long-paper readiness.

Only one additional dataset is justified: SciFact supplies existing sentence-level annotation membership that AVeriTeC's supplied QA does not. AVeriTeC remains the all-dev, four-way application measurement. SciFact is not added merely to increase dataset count, and neither public dataset is a clean temporal/generalization holdout for these pretrained LLMs.

## What is already established, and what is not new

The completed [formal result review](../ops/PAPER2_FORMAL_RESULT_REVIEW_20260930.md) supports correctness-changing sensitivity beyond the two observed fresh-run backgrounds. It does not establish consistent net duplication harm: only Granite's partial-repetition contrast had a negative signed interval, and padding itself harmed Granite. Exact fragment canonicalization already enforces the constructed content-invariance contract. None of those facts may be suppressed to motivate this supplement.

| Primary source | Relevant overlap and the remaining distinction |
|---|---|
| [Prompt Repetition, arXiv:2512.14982v1](https://arxiv.org/html/2512.14982v1), Sections 1–2, future directions, Appendix A.4 | Already compares whole-query repetition with period padding and explicitly proposes partial-prompt repetition as future work. A new partial-repeat arm is not sufficient novelty. The candidate must directly retain whole-user-query and balanced-evidence controls; its possible addition is an annotation-conditioned allocation measurement, not the invention of re-reading. |
| [Team Papelo, FEVER 2024](https://aclanthology.org/2024.fever-1.2.pdf), Algorithm 1, Section 4.4, Tables 2–3 | Crucially, Table 2 already reports verdict accuracy for the pre-verdict “Repeat not para” ablation: .710, versus .745 for “No paraphrase,” on the first 200 dev examples. Table 3 separately reports post-verdict duplication with unchanged verdict accuracy and improved evidence score. Thus prior art is **not only metric exploitation**. Those results do not isolate this fixed-complete-input, role-conditioned budget contrast. We do not reproduce their restriction to two verdict classes. |
| [FactEval, NAACL 2025](https://aclanthology.org/2025.naacl-long.534.pdf), Section 3.1 | Perturbs claims while retaining evidence, including repetition-related perturbations. Robustness evaluation and repetition sensitivity are already known; this candidate intervenes in evidence allocation while holding the claim and unique evidence fixed. |
| [SciFact](https://aclanthology.org/2020.emnlp-main.609.pdf) and [official native schema](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/doc/data.md) | Existing work already evaluates oracle abstract access and rationale-based verification. Oracle documents/rationales are not a contribution here. Multiple alternative rationale sets require union-aware membership; a member sentence need not alone constitute a sufficient rationale. |
| [BiDeV](https://arxiv.org/html/2502.16181v1) and [FAE/REAL](https://arxiv.org/html/2609.08943v1) | Evidence filtering and removal-based evidence-dependence testing are known. This candidate adds/removes no unique supplied fact. The latter paper also explains why unannotated evidence can remain valid after annotated evidence is removed; non-rationale material cannot be treated as certified irrelevant. |

These primary sources were checked directly. The search bounds a plausible distinction; it is not proof that no related paper exists. The strongest defensible prospective contribution is a small, controlled **input-policy measurement with an annotation-role interaction**, not a new canonicalizer or a universally harmful-duplication theorem.

## Material passport and fixed populations

### AVeriTeC application population

Use all 500 existing official dev records and 491 exact-claim groups, with the frozen native IDs and source order in [FORMAL_PROTOCOL_V1.md](FORMAL_PROTOCOL_V1.md). Input SHA-256: `499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300`. Keep all 122 Supported, 305 Refuted, 35 Not Enough Evidence, and 38 Conflicting Evidence/Cherrypicking records and the original four-letter response space. No binary mapping or class-driven exclusion is allowed.

This is **adaptive reuse of the entire observed development population**: the earlier outcomes motivated the question. A newly frozen supplement does not turn these records into unseen confirmatory data. Official blind-test/2025 labeled native QA was not verified as available; do not relabel those sources or reconstruct labels from predictions. See [public-asset feasibility](../ops/SUPPLEMENTAL_PUBLIC_ASSET_FEASIBILITY_20260930.md).

The read-only content audit found 109 records with one distinct rendered QA-text class and 391 with at least two; there are three records with duplicate QA content across native IDs, totaling three extra IDs, and one such class has multiple actual URLs. The old hash anchor lands in a duplicate-text class in two records. These are eligibility/identity facts, not outcomes. All 500 remain in every arm. The 109 one-content records are a useful exact degeneracy check: balanced and selective evidence duplication must coincide under the construction below. Report the 391 genuinely selective cases as a predeclared descriptive stratum, never as a replacement for all-dev results.

### SciFact mechanism population

Use **all 209 annotated claim–document pairs** in official dev, covering 188 exact claim texts and 182 distinct full abstracts: 138 SUPPORT and 71 CONTRADICT pairs. This prospective population is defined by the existence of native evidence annotations, not by model correctness, ambiguity, sentence length, or a favorable outcome. It is not the overall SciFact retrieval/three-way benchmark population; the 112 dev claims without an annotated evidence pair are outside this particular estimand, not relabeled or silently dropped.

The [official intake receipt](../ops/SCIFACT_PUBLIC_INTAKE_20260930.md) binds the exact source bytes:

| Local file under `data/scifact_official_20260930/` | SHA-256 |
|---|---|
| `data.tar.gz` | `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be` |
| `corpus.jsonl` | `b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62` |
| `claims_dev.jsonl` | `86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217` |
| `claims_train.jsonl` — exposure audit only, no demonstrations/tuning | `f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0` |

The 209 pairs have 338 native rationale sets, no invalid indices, mixed labels, or duplicated rationale sets. All 209 have at least one marked sentence-text class and at least one unmarked class after checking every alias; none requires exclusion. Abstracts contain 3–42 sentences, marked classes 1–6, unmarked classes 2–38. Although 13 corpus documents have internal identical sentences, none appears in these dev evidence pairs.

Train and dev evidence share 110 exact abstracts: 124/209 dev pairs and 112/188 evidence-bearing dev claims involve such a document. No evidence-bearing dev claim text overlaps train; the two complete-split exact-claim overlaps are both outside the evidence-bearing dev population. No train examples will be used for prompts, fitting, selection, or development. Pretraining exposure remains unknown; do not claim uncontaminated model evaluation.

Carry together pairs sharing exact claim text **or** exact full-abstract text. Their transitive exact-text dependency components number 164, with pair-count sizes `{1:137, 2:19, 3:3, 4:3, 5:1, 8:1}`. This grouping uses only the supplied native pair membership and exact text, not historical relationship, reply, or conversation data.

## Fixed complete inputs and target identity

All prompts retain the exact claim and all permitted unique supplied evidence. No new retrieval, rewritten evidence, omission, sentence truncation, additional labels, rationale explanation, or citation-context input is allowed.

For AVeriTeC, construct a **single fixed canonical input for this supplement** by merging byte-identical rendered QA text, retaining the union of its actual URLs and its first-occurrence source order. Use the existing whitelist for question, answer, URL, and Boolean explanation. Keep all native QA IDs in the audit; the visible stable unit identifier is the first original ID in its text class. Do not invent an independent-source identifier for a copy. This is ordinary preprocessing, not a proposed method. It changes presentation from the earlier raw-packet baseline, so no old predictions can stand in for any new arm.

The selective AV target remains the existing `SHA256("paper2-anchor-v1:" + claim_id) mod original_unit_count` anchor. Compute it on the original unit list, then resolve that original ID to its canonical text class and URL union. Do not substitute the first unit, a supposedly supportive `target0`, a model-attribution maximum, or an outcome-selected anchor. A target is an identity, not a correct/relevant evidence label.

For SciFact, retain the full abstract in native sentence order, its title, and the exact claim. Sentence labels are stable native zero-based indices; repeated copies retain the same label. Document title and claim appear once except in the explicitly defined whole-user reread. Do not input `cited_doc_ids`, citances, or surrounding articles. Define sentence-text classes within the abstract and mark a class if **any** alias index belongs to the union of all existing rationale sets for that pair. An unmarked class has no marked alias. Order each pool by its smallest native index.

Choose one class from each pool using `int(SHA256(namespace + str(claim_id) + ":" + str(doc_id)),16) mod pool_size`, with namespaces `paper2-supplement-v1:marked:` and `paper2-supplement-v1:unmarked:` respectively. This is one fixed draw per pair, not a sentence sweep. Save indices, hashes, alias memberships, and pool sizes in an audit-only reference file. No pool membership, rationale index, gold verdict, or role name enters the model prompt.

This selection is explicitly **annotation-conditioned experimental construction**. A deployment operator without rationales could not execute it, and an executor generating a real policy must not secretly receive these gold fields. Separate the construction/reference process from the inference request file; the latter contains messages and opaque IDs only. The original full text remains visible to the model, as required by the task.

SciFact uses its native response meanings: `A = SUPPORT; B = CONTRADICT; C = NOINFO`. C remains a valid possible model answer and is wrong for this defined evidence-bearing population; do not force binary classification. Freeze one literal instruction before implementation review: “Decide whether the scientific claim is supported, contradicted, or not established by the provided abstract. Repetition does not add factual information. Document and sentence numbers identify the same supplied source, not independent sources. Return exactly one letter, with no explanation: A = SUPPORT; B = CONTRADICT; C = NOINFO.” Add the same existing padding instruction/wrapper in every arm. AV retains its existing four-way instruction, adapted only to stable unit identifiers if needed and frozen before inference; no alternative wording trials.

## The smallest contrast set proposed

Let E be the complete canonical evidence block and U the complete user payload containing claim, title if applicable, and E, before the common padding wrapper. A repeated unit includes its unchanged stable identifier and provenance. Join repeated blocks with exactly two newlines, without new persuasion, relevance, or source-count language.

| Arm | Construction before final padding | Purpose |
|---|---|---|
| C | U, empty padding wrapper | Zero-extra-evidence/native-information baseline; also the ordinary canonicalization baseline. |
| P | U, padded to the common target | Exposes the behavior of the chosen filler; not an assumed neutral placebo. |
| B | Claim/title once, then E followed by E | Balanced evidence re-reading: every canonical evidence unit appears twice. |
| W | U followed by U; system instruction once | Whole-user-query re-reading, the direct chat implementation of the closest repetition control; not duplication of the system/chat role tokens. |
| S-front, S-back — AV only | Fixed anchor copies before or after one complete E | Uneven evidence-budget allocation using the old gold-free anchor. |
| R-front, R-back — SciFact only | Fixed marked-class copies before or after one complete E | Annotation-conditioned marked allocation. |
| O-front, O-back — SciFact only | Fixed unmarked-class copies before or after one complete E | Annotation-conditioned unmarked allocation. |

This is **6 arms per AV record and 8 per SciFact pair**, not a factorial grid over copy factors or prompts. The two placements are fixed controls, averaged with equal weight, not a choice of the better position.

### One evidence-pass budget, not a tuned copy count

For each record and pinned tokenizer, measure the complete chat-token length of unpadded B, including its empty common wrapper: `T_B`. For a selected target, use the same integer k in both placements: the largest positive number of **whole additional unit copies** for which both fully rendered selective messages have length at most `T_B`. Thus the extra-content budget is one complete evidence pass; short units may receive more copies than long units. This is a deterministic tokenizer construction, not a search over model outcomes. The total multiplicity of the selected unit is k+1, while every other unit remains once. Record k and the actual repeated-content token share; do not call this a fixed-dose role effect.

The common final target is `T = max(T_B, token_length(W_with_empty_wrapper))`. Pad P, B, W and every selective arm to T using only the existing literal ` .` filler and the frozen at-most-50-candidate matcher in [FORMAL_PROTOCOL_V1.md](FORMAL_PROTOCOL_V1.md). C retains the empty wrapper. Both front/back variants of a target use the same k. Unit copies never contain truncated sentences. If the construction cannot admit one whole copy, or cannot match T, retain an explicit preparation failure and stop under the guard below; do not search for another sentence or filler.

For AV's 109 one-content records, E is the sole canonical unit. Both selective placements must have k=1 and be **byte-identical to B after padding**. Their zero selective-versus-balanced effect is a software identity/cached negative control, not model robustness. Require this equality during preparation.

Equal total token count removes a token-count difference among P/B/W/selective policies. It does **not** neutralize filler content, attention positions, sentence semantics, or the different numbers of whole copies. P−C, B−P and W−B must all be reported. Prefix/suffix controls expose one placement dependence but do not exhaust possible order effects. A positive R−O contrast by itself could reflect extra exposure to already useful material; it does not uniquely establish erroneous evidence counting. The operational claim above is the strongest claim this minimal design can identify.

## Estimation, materiality, and stop rule

Define correctness against the existing native pair/record label, with invalid and overlength outputs scored zero. Average A/B realizations equally. For selective arms, also average front/back equally. Never select a realization, placement, class, record, or model after seeing outcomes.

The **only primary scientific family** is four SciFact signed differences: `R−O` and `B−O` for each of the two already fixed models. Use all 209 pairs, record weighted. Paired bootstrap the 164 exact claim/abstract components with all arms, placements and realizations retained, 10,000 draws, seed 20260930. Use two-sided 98.75% percentile intervals for this four-contrast Bonferroni family, conditional on these fixed models, targets, prompts, public dev population, and grouping assumptions. This is not uncertainty over arbitrary future targets, pretrained models, or unseen domains; two process repeats are not two independent datasets.

One prospective screen requires **all four** point differences to be at least **+3 percentage points**, all four interval lower endpoints above zero, and positive point differences in **both** individual placements for both contrast types/models. The 3-point margin is an explicit practical screen (roughly six to seven pair-correct equivalents), not a significance threshold fitted to a result or a publication-quality guarantee. This small population may be underpowered; failing the screen is not evidence that the effect is exactly zero.

Report these mandatory diagnostics without additional success claims or selecting a favorable subset:

- Every arm's native accuracy, fixed-native-class macro-F1, class confusion counts, valid-output coverage, tokens and runtime. For SciFact, the fixed response space includes NOINFO even though this population has no NOINFO gold pairs; label this support restriction prominently.
- Repairs, breakages and changed-wrong-label counts for selective versus B, B versus C, W versus C, and P versus C; give front/back and A/B separately as well as fixed averages.
- AV all-500 effects and the predeclared 391/109 content-count strata, all four original classes intact. Report B versus S both signed and by correctness-changing sensitivity. No inferential promotion of this adaptively reused dev population.
- Same-message fresh-process crossing rates and the existing equal-opportunity intervention-minus-repeat crossing statistic on complete-valid panels, with each panel's exact denominator. Invalid outputs are not semantic crossings. Report whether the apparent accuracy pattern remains directionally present on the common-valid panel; an effect driven only by formatting failures is not evidence-role reasoning.
- Padding counts, k distributions, selected sentence lengths/positions, native rationale-set sizes, and exact-match/overlength coverage. These describe the intervention and possible explanations; no regression, subgroup, or alternate target is promoted to a rescue claim.

Before inference, require exact common lengths and no overlength in at least 95% of each corpus/model population (AV at least 475/500, SciFact at least 199/209), with all omitted-from-inference cases retained as explicit failures in the complete reference table. If that guard fails, stop the candidate: do not truncate, drop long cases, alter the budget, or substitute data. Any integrity mismatch in complete unique content, URLs, labels, aliases, or the 109 degeneracies stops preparation regardless of coverage. Report every failure, not just the matched subset.

After inference, **stop this supplement after the fixed four complete realizations and one analysis**, whether the screen passes, fails, or is inconclusive. Below 95% complete-valid primary comparison coverage, withhold a population-level evidence-role interpretation even if all-population accuracy differs. A failed primary screen cannot be rescued by an AV crossing result, a significant single model, a favorable placement, a class subset, a different rationale set, or another decoder/prompt/copy strength. If only generic B/W repetition or filler effects remain, credit the known controls and close the mechanism claim.

## Execution ceiling and remaining pre-run gates

Retain exactly the existing local Qwen2.5-7B-Instruct and Granite-3.3-8B-Instruct revisions from the formal protocol, the frozen generation implementation/configuration, greedy seed 17, batch 8, one shard, bfloat16/SDPA, slow tokenizer, thinking off, maximum input 8192 and output 16 tokens. No external inference, model replacement, demonstrations, or training is proposed.

Per model and fresh realization there are exactly `500×6 + 209×8 = 4,672` logical requests. Two models × two complete fresh realizations give **18,688 logical requests**, at most that many physical generations before within-process exact-message caching. AV accounts for 12,000 and SciFact for 6,688. Hard worst-case charged token ceilings are **153,092,096 input** and **299,008 output** tokens; the preflight must also publish actual totals rather than treating those bounds as expected consumption. At most **4 aggregate GPU-hours** is proposed, with at most one GPU per process and no interference with existing jobs. Stop on the ceiling; an incomplete run yields no scientific claim and does not authorize a favorable subset or automatic extension.

Concatenate AV in original order and SciFact by native claim ID/document ID. Use the table's fixed within-record order; keep exact-message caching global within a realization. Run the entire file A and then B in fresh processes and fresh outputs, no resume or cross-process cache. Do not reuse earlier formal predictions. Bind content/target/reference/request/tokenizer hashes and identical batch/cache positions across A/B; do not inspect model outcomes until both models' A/B files are complete. No selective regeneration of failures is permitted. A runtime failure remains archived and returns to review; this candidate does not pre-authorize repeat attempts beyond its ceiling.

Before any launch, independent review must settle the literal renderer/instructions and deterministic maximal-whole-copy calculation, validate the source-to-message/reference separation, and check synthetic/native duplicate and alias cases. CPU-only preparation must bind all 709 observations and exact counts, verify canonical recovery of complete unique content and URLs for every policy, and show feasibility under the fixed length guard. These are implementation-verification gates, not new opportunities to optimize effects. This document itself grants no execution approval.

## Scientific disposition

The question is implementable with the available assets and no new annotation. Its most useful possible result would be a reproducible, practically material difference between **balanced re-reading and a specific unequal evidence-allocation policy**, with an existing annotation-role interaction and all unique content retained. It could explain why generic recommendations to repeat an input do not automatically transfer to evidence aggregation.

However, known prompt repetition, known pre-verdict QA duplication, ordinary canonicalization, a small oracle-conditioned scientific population, observed padding non-neutrality, and adaptive AV reuse materially limit novelty and identification. The candidate does not by itself justify a long-paper method claim. If the lead requires a pure multiplicity mechanism, a deployable gold-free selector, or independently confirmed unseen-Web performance, **do not launch this candidate for that purpose**: it cannot supply those claims. The honest alternative is to retain the completed work as a bounded supplied-evidence sensitivity measurement and its strong-control limitations, not manufacture a new framework or a numerical publication-score promise.
