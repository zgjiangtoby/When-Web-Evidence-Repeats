# Operational repetition-allocation supplement v1

## Material passport and authority

Implementation/design: GPT-6 Astra/xhigh; bounded CPU checks/execution: GPT-6 Sol/medium; independent review is required and remains separate. Base commit: none (this directory is not a Git repository). This is an implementation of the root-adopted [candidate](SUPPLEMENT_DESIGN_CANDIDATE_20260930.md), SHA-256 `9caa2d01cd4746f0b5c8738a54a2af194d403cc8e46f739162eef1fdbedd0217`. It is not inference approval, a method claim, an attention-causality test, or a paper-quality gate. Original formal sources/results and `shared/generate.py` are unchanged.

The independent reviewer additionally identified GroupQA (Findings ACL 2026, [primary paper](https://aclanthology.org/2026.findings-acl.2003/)), [Schuster et al.](https://arxiv.org/html/2601.03746v1), [Ross et al.](https://arxiv.org/html/2608.13956v1), and [Liu et al.](https://arxiv.org/html/2609.08698v1) as close evidence-repetition/weighting/grouping precedents. Generic weighting, grouping invariance, or duplicate-evidence sensitivity must not be claimed as new. This implementation adds no arms in response. The exact operational contrast may be measured without asserting it warrants a long paper.

## Frozen population and intervention

All 500 original AVeriTeC dev records/491 exact claim groups remain, with all four native classes. All 209 SciFact dev evidence-bearing claim–document pairs remain (188 claims, 182 abstracts, 138 SUPPORT/71 CONTRADICT), resampled in 164 connected components of shared exact claim or full-abstract text. The exact official files/hashes, alias treatment, existing-label restrictions, and exposure caveats in the candidate and [intake](../ops/SCIFACT_PUBLIC_INTAKE_20260930.md) apply unchanged. No original native partition is reconstructed and no test, reply, parent, citance, or new annotation is used.

AV merges identical rendered QA text in first-occurrence order and unions actual URLs. The original hash anchor is computed before merging and resolved to the resulting text class. Each rendered block is `Evidence unit {first_original_ID}:\n{unchanged_QA_text}` followed by `\nSources: {sorted_URLs_joined_by_comma_space}` when URLs exist. The exact original `prepare.SYSTEM` is retained: no prompt variant is tried. SciFact renders `Sentence {zero_based_index}:\n{unchanged_sentence}`; all native sentences and the exact title remain. Its literal three-way instruction is fixed in `supplement.SCI_SYSTEM`; NOINFO is valid, never forced out even though gold support is zero in this defined population.

SciFact targets are the candidate's two fixed SHA-256 pool draws. Marked membership is the union across every native alternative rationale, extended to all exact-text aliases. Unmarked means no marked alias, not irrelevant. Labels, memberships, targets, and condition names live in references/audits only. Runner requests contain opaque IDs, ordinary messages and empty metadata. They reveal neither gold verdict nor a role label. Annotation-conditioned construction is not a gold-free deployable selector.

AV arms are C/P/B/W/S-front/S-back; SciFact arms C/P/B/W/R-front/R-back/O-front/O-back. U is the claim/title/header plus complete evidence E. B repeats E once, W repeats U once; the system is never doubled. Copies retain original unit IDs, use two newline separators, and introduce no new source identity. Every message has the same padding instruction and wrapper. The two selective placements use the same k and are averaged, never selected.

## Implementation decisions declared before dependent code

| ID | Interface or semantic issue | Frozen choice and source |
|---|---|---|
| I1 | BPE token counts need not be monotone in k | Exhaust every k within a proved letter-run upper bound; no binary search or first-overflow stopping. Numerical certification failure stops preparation, not a capped approximation. Implementation choice within root's requirement. |
| I2 | Excessive numerical preflight work | Upper bound must be at most 4096 per target; otherwise stop before emitting a runnable request file. This is a computational guard, not a maximum scientific copy factor. No smaller-k fallback. |
| I3 | Length mismatch below the population guard | Keep every actual best-match message and explicit mismatch audit. The unchanged runner can generate on that message, but its experiment outcome is a `construction_length_mismatch` placeholder with prediction null/correctness zero, with any raw prediction separately preserved. Do not replace the message with C. Overlength messages retain the runner's rejection. A failed placement contributes zero to its fixed average in both realizations. Matched-only descriptions never replace full-population scoring. Clarifies candidate failure retention. |
| I4 | Reuse of fixed-four-class scoring helpers | Reuse AV scoring; implement native three-label parsing/F1 for SciFact only. Zero-support NOINFO F1 uses zero-division value 0, so the reported three-class macro-F1 is not an overall SciFact benchmark score. |
| I5 | Freshness and GPU allocation | Require all four completed manifests, identical A/B cache/batch layouts, chronological fresh A→B, and an operator execution receipt binding output bytes and occupied-device wall time. Four GPU-hours means sum of one-GPU process occupancy, not CPU preparation time or sum of per-record batch times. |
| I6 | Review/artifact scope | Keep specification, decisions and commands here; only the two new Python files plus this protocol are authored. No new framework/tracker or old-code edits. User ownership overrides generic skill artifact defaults. |

## Certified finite copy enumeration

The supported local slow tokenizer classes are exactly Qwen2Tokenizer and GPT2Tokenizer, using their inspected regex pre-tokenization and byte-BPE implementation. Their regex pieces contain at most one contiguous Unicode-letter run; byte BPE merges only within a piece. Count after the tokenizer's `prepare_for_tokenization` normalization: Qwen2 applies NFC, which can join raw letter runs separated by combining marks. Registered added-token spans are removed from the letter-run count, because one added token could encode several such runs. Added tokens must not span newline-separated copy boundaries; unsupported tokenizer/regex/added-token structure fails closed. Each counted complete chat must contain the literal complete user payload; transformations/removal fail preparation.

Let w be the number of Unicode-letter runs remaining in one complete rendered selected unit after added-token removal. Since copies are separated by two newlines, their runs cannot join; k copies require at least k·w tokens, regardless of other input. For the B arm's complete-chat token count T_B, any fitting k therefore obeys `1 <= k <= floor(T_B/w)`. Require w>0. Inspect every integer in that bound and both placements with the existing complete-chat `formal.token_counter`. Choose the largest k for which both lengths are <=T_B. Save every candidate's two measured lengths, w, upper bound and selected k. This establishes a maximum without token-count monotonicity. The implementation must verify the tokenizer assumptions and exact rendered-copy preservation, not infer them from two probes.

The final target is `max(T_B, tokens(W))`. Reuse `formal.padding_control` unchanged, including its sole ` .` filler, 32768 filler bound and <=50 probes. C has empty padding. Every AV one-class record (exactly 109) must yield k=1 and byte-equal padded B/S-front/S-back messages. This is a software/cache identity, not a scientific observation.

Any content/URL/label/alias/degeneracy inconsistency stops preparation. Numerical certification failure also stops before runnable requests are written, preserving all source-observation audits and a stop reason. At least 475/500 AV records and 199/209 SciFact pairs per tokenizer must have all non-C arms exactly matched and every arm <=8192. Below that threshold, preserve the complete artifacts with status STOP_PREPARATION_GUARD and reject evaluation/launch eligibility. No truncation, replacement target, alternate filler or budget change is allowed.

## Primary contrasts and required reporting

The sole primary family is R−O and B−O for each of the fixed two models. Average the two positions and the two complete fresh runs equally. Invalid/overlength outputs score zero. Bootstrap all 209 pairs, record weighted, in the 164 exact-text dependency components; 10000 paired draws, seed 20260930, two-sided 98.75% familywise percentile intervals (family size four), reusing `formal.group_interval` and reporting the actual grouping.

The mechanical screen requires all four differences >=0.03, all four lower interval endpoints >0, both placement point differences >0 for each model/contrast, common-valid coverage >=95%, and positive common-valid mean differences. Common-valid means every output needed for that contrast, both placements and both runs, is valid. Additionally use the shared R/O/B panel (B, R-front/back, O-front/back in both runs) for the independent review's direction check: it also needs at least 199/209 pairs and positive mean differences for both contrasts. Construction failures are not valid outputs for either panel. A screen flag does not declare a supported causal claim, publication readiness, or independent confirmation.

Report every arm and A/B separately; all-population accuracy, native fixed-class F1/support/confusions, statuses, and physical/logical costs; primary pair records; all front/back and fixed averages; repairs/breakages/changed-wrong-label counts; equal-opportunity correctness/verdict crossing excess over fresh-repeat noise with exact valid denominators; C/P/B/W comparisons; AV all500 and predeclared 391/109 strata; k, selected-length/position/pool/rationale-set and filler/length audits. No best class/placement/target/model is selected. AV effects remain adaptive descriptive measurements. Padding is not presumed inert and target policies differ in whole-copy counts: no pure semantic-multiplicity interpretation.

The fixed study ends after one analysis of the four complete files. Failure/inconclusiveness does not authorize more arms, prompts, targets, data, or retries. Runtime failure remains archived and returns to the lead; incomplete files cannot produce an evaluable scientific report.

## Artifacts, completion and costs

Each model-specific preparation contains all 709 source observations, 4672 opaque requests and one separate native reference per request, copy/length/padding audits, and a manifest binding source bytes, candidate/protocol/source/tests, frozen formal helpers, local tokenizer/config/revision and package versions. Loading a preparation verifies bindings and reconstructs artifacts. All four distinct completed output files are checked before statistics are calculated. Original and new predictions are never mixed.

Total logical requests: `2 models × 2 fresh runs × (500×6 + 209×8) = 18688`. At most 153092096 charged input and 299008 charged output tokens; <=4 aggregate GPU-hours. Fixed generator settings: one shard, batch8, seed17, greedy, bfloat16, SDPA, slow tokenizer, thinking off, input8192/output16. Within-process exact-message caching is global, including native repeated claims and one-unit degeneracies. No cross-process cache/resume/selective regeneration. CPU preparation is outside the GPU-hour budget, but the certified bound prevents unbounded per-target enumeration.

The execution receipt is an operator-owned JSON object with four `runs` entries. Each binds `model_key`, `realization` (A/B), `output_path`, `output_sha256`, `manifest_sha256`, `launch_utc`, `finish_utc`, `visible_gpu_count: 1`, `output_existed_before_launch: false`, `resume: false`, and the exact child `argv` (Python executable then unchanged runner and arguments). Timestamps must bracket the generator manifest and A must finish before B starts for each model. Sum occupied-device wall seconds over all entries, including loading/teardown and any failed authorized attempt; no receipt can turn a missing/failed attempt into a complete experiment. The CPU evaluator rejects totals above 14400 seconds. No launcher is supplied by this module.

## CPU commands and build record

From the project root, use the approved sunproj Python executable, with `PYTHONPATH=paper2:shared`. These commands are CPU-only; preparing requests does not authorize inference:

```bash
PYTHONPATH=paper2:shared python -m unittest paper2/test_supplement.py -v
PYTHONPATH=paper2:shared python paper2/supplement.py prepare --model-key qwen --output paper2/prepared/supplement_v1_qwen
PYTHONPATH=paper2:shared python paper2/supplement.py prepare --model-key granite --output paper2/prepared/supplement_v1_granite
PYTHONPATH=paper2:shared python paper2/supplement.py verify --prepared paper2/prepared/supplement_v1_qwen
PYTHONPATH=paper2:shared python paper2/supplement.py verify --prepared paper2/prepared/supplement_v1_granite
```

Evaluation requires both prepared directories, all four fresh outputs and the bound execution receipt in one invocation; CLI help gives the exact named flags. It writes only a fresh output directory after all completion/provenance checks pass. No evaluation is authorized before the fixed run set completes.

Build record: source implementation in progress; synthetic/native CPU checks and independent source review pending. No model inference or outcome inspection has been performed for this supplement. Skill-guided implementation uses a declared-assumption ledger and one compact regression suite; it does not self-approve scientific validity.
