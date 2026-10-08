# AVeriTeC provided-evidence measurement, formal protocol v1

Design date: 2026-09-30, Asia/Shanghai. Scientific design/implementation: GPT-6-Astra/xhigh. Bounded execution and CPU checks: GPT-6-Sol/medium. This protocol follows the fresh independent [post-pilot decision](../ops/PAPER2_POST_PILOT_DECISION_20260930.md). It is a pre-inference protocol candidate for independent review, not a completed formal result or a promise of a long-paper score. No official dev model outcome was used in this design.

## Research question and stopping rule

Does controlled repetition of unchanged supplied QA evidence change native four-way verification correctness beyond the background from repeating the same messages in a fresh, identically configured run? What accuracy, invariance, and token-cost differences remain after ordinary packet/fragment canonicalization? The two primary observations are signed accuracy change and correctness-changing sensitivity; neither implies the other.

The 32-training-claim pilot did not show net replication harm, omitted one native class by chance, and had identical-message discrepancies across independent shard caches. Preserve those findings. Retire natural-lineage recovery, copying-frequency estimation, score-envelope effectiveness, and optimizer-superiority claims. The new experiment uses one global message cache per realization and repeats the complete file. Canonical message equality is a software contract, not a new semantic capability.

If crossing excess does not exceed fresh-repeat background, report that controlled sensitivity was not established. If canonicalization accounts for sensitivity without a remaining substantive problem, credit canonicalization. Retain improvements, harm, nulls, model differences and unfavorable ledger effects. Inconclusive results do not automatically authorize extra copy factors, prompts, models, sample growth, auxiliary datasets or labels. No new human annotation, parent/reply inputs, conversation tasks, external inference, private-data upload or additional retrieval is used.

## Native source, membership, and evidence

Use all **500** original records in local `paper2/data/averitec_dev.json`, pinned official `chenxwh/AVeriTeC@37a96094b68f6c7977734a0d35c7c02a1b1eef07`. Exact file SHA256: `499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300`. The local source manifest binds the official release and file. Keep source list order, native record IDs `dev:0` … `dev:499`, exact original claim strings, existing four-way labels, and supplied QA evidence. There are 491 exact-claim groups; multiple native records with the same original claim remain distinct observations and are clustered together for uncertainty. There is no partition reassignment or duplicate-record deletion.

Keep Supported, Refuted, Not Enough Evidence and Conflicting Evidence/Cherrypicking. The existing hyphenated spelling alias maps to the last native class; unknown labels are errors. Prior intake reports native support 122/305/35/38 respectively. These are source counts, not model results or grounds for class selection. Fixed-four-class macro-F1 retains every class even where a subset has zero support. The train pilot excluded dev exact-claim overlaps; its results remain developmental evidence.

Reuse `extract_units`, `anchor_index`, `make_replay`, packet rendering and exact fragment rendering from the unchanged pilot helpers. The permitted fields are question, answer, actual source URL, and existing Boolean explanation. Empty answer lists retain the existing “No answer could be found.” rendering. Source labels, gold justifications, article metadata, model-derived source judgments and reference verdicts do not enter messages. Schema/empty-evidence incompatibility stops preparation with the original record identity; no record is silently excluded or repaired. The references and source audit preserve original label spelling, source index, exact claim hash, unit counts and hashes of permitted evidence.

## Conditions and presentations

Freeze copy factor four and the existing label-free anchor function `SHA256("paper2-anchor-v1:" + claim_id) mod unit_count`. Every original QA unit remains present, with its unchanged text and actual URLs.

| Condition | Existing helper operation | Applicability |
|---|---|---|
| clean | Original singleton evidence packets | All records |
| duplicate4 | Original singleton packets plus three occurrences of the anchor unit | All records |
| partial1 | Each other unit paired with one anchor occurrence | At least two QA units |
| partial4 | Each other unit paired with four anchor occurrences | At least two QA units |

For each applicable condition keep all four original presentations: `raw_packets`, `exact_packet_dedup`, `exact_fragment_dedup`, `ledger_counts`. Exact whole-packet dedup uses identical unit text and URL signatures. Fragment dedup merges byte-identical rendered QA text, sorts it deterministically and preserves the union of actual URLs. Occurrence counts are explicitly metadata, not numbers of independent sources. No lexical/semantic approximate merge, envelope score, retrieved evidence or oracle natural-lineage field is introduced.

Preservation concerns distinct rendered QA text and its actual source-URL union, not the multiplicity of original QA record IDs. Released records can contain separate QA IDs with byte-identical text and URLs; exact packet dedup may legitimately lower the existing `retained_unit_fraction` metadata while retaining all permitted evidence. Preparation verifies canonical content preservation and exact presentation reconstruction, keeps that original ID-retention metadata visible, and never drops the native claim record for this reason. This is a pre-inference validation correction; prompts, comparisons and native references are unchanged.

The original shared four-letter verification instruction remains, with this exact added sentence in every system message:

```text
 Text inside <non_evidence_padding> is artificial filler, not evidence; ignore it when judging the claim.
```

Every user message, in every presentation and base condition, appends the same empty wrapper:

```text
\n\nNon-evidence padding (ignore for verification):\n<non_evidence_padding>\n
\n</non_evidence_padding>
```

Here `\n` denotes actual newline characters; the exact constants `PAD_START` and `PAD_END` in `formal.py` are authoritative. Using the same empty wrapper throughout preserves canonical equality across pure reuse and avoids introducing the wrapper uniquely in a padded arm. Claim and evidence preceding it are unchanged.

## Two frozen tokenizer-specific length controls

Add only `clean_length_control/raw_packets` and, when partial is applicable, `partial1_length_control/raw_packets`. They retain the baseline raw QA text and add filler inside the wrapper until the complete rendered input matches `duplicate4/raw_packets` or `partial4/raw_packets`, respectively, or the frozen search stops with a recorded mismatch.

The sole filler unit is the literal two-character string ` .` (ASCII space then period). Padding is exactly k repetitions of that string, with `0 ≤ k ≤ 32768`. No filler choice depends on labels, verdicts or model outcomes. A control does not truncate, omit, reorder or alter its baseline QA evidence.

Measure token length using the local pinned model's slow tokenizer and precisely the runner's complete `apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=False, enable_thinking=False)` result, followed by `encode(..., add_special_tokens=False)`. Count the system message, instruction, role markers, wrapper, evidence and assistant prefix. Qwen and Granite have separate prepared request files; their nonpadding messages are identical, but padding lengths can differ.

Frozen bounded search:

1. Measure the zero-filler candidate k=0 and the corresponding repeated-evidence target.
2. Up to 16 times, propose `clip(k + target_tokens − actual_tokens, 0, 32768)`; stop if exact or if that candidate was already measured. Measure each new candidate on the complete rendered chat.
3. Choose the measured candidate with smallest absolute mismatch; ties choose smaller k. If it is not exact, measure all clipped offsets −16 … +16 around that candidate once, then choose with the same rule. At most 50 distinct k values are evaluated.
4. Retain the best candidate and record target/actual/base length, signed mismatch, exact-match flag, chosen k, measured candidate lengths, hashes and original paired IDs. There is no retry with another filler or outcome-dependent selection.

All selected records remain in the main study. A mismatch prevents a matched-length interpretation for that pair. Report exact-match coverage and both all-pair descriptive and exact-match-pair effects. Inputs exceeding 8192 tokens remain emitted failures without truncation, even when an exact length match exists. Treat padding as an operational control, not behaviorally inert text: a difference concerns this evidence presentation relative to this particular filler, not a universal length-independent copying mechanism.

## Model and execution freeze

| Model | Immutable revision | Local model directory |
|---|---|---|
| Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | `models/qwen2_5_7b_instruct` |
| Granite-3.3-8B-Instruct | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | `models/granite_3_3_8b_instruct` |

Use unchanged `shared/generate.py`, SHA256 `f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c`, and the validated sunproj Python environment. Freeze **one shard** (`num_shards=1`, `shard_index=0`) per model, batch 8, seed 17, greedy decoding, bfloat16, SDPA, slow tokenizer, thinking disabled, maximum input 8192 and maximum output 16 tokens. Bind local tokenizer/config files, checkpoint revision, torch/Transformers/tokenizers versions and all preparation artifacts before inference.

Order records by the original source index. Within a record use the existing order clean, duplicate4, partial1, partial4, each with raw, packet dedup, fragment dedup, counts; append the raw clean length control and then the raw partial1 length control when applicable. Do not shuffle. Equal messages share one exact-message cache across the entire single shard, including duplicate native claims where their full messages happen to match. Retain logical duplicates and every cache hit.

Run the **entire same request file twice**, A then B, in fresh processes with fresh output paths and identical configuration. Do not pass `--resume`, reuse output files, divide the run into shards or cache across processes. No model-score or disagreement inspection occurs before both complete. Require identical per-request effective batch sizes and cache positions across realizations. The evaluator verifies exact global cache origins and within-realization raw-response/status equality for identical messages, while allowing differences between the two fresh physical generations.

Keep exact launch commands, complete logs, process/start/end times and manifests. The unchanged runner does not fully encode resume history, so the executor's fresh-output and no-resume receipt is essential. If interrupted, retain the incomplete realization and rerun the whole file under a fresh versioned output name before inspecting outcomes; retain every attempt and never select a realization for favorable performance. No selective regeneration of invalid, overlength or disagreement cases.

## Estimands and uncertainty

For each model and each raw intervention (`duplicate4 − clean`, `partial4 − partial1`), retain baseline/intervention correctness in A and B as `c0A,c1A,c0B,c1B`. Every invalid or overlength output has correctness zero. The signed accuracy estimand is the native-record mean of `[(c1A−c0A)+(c1B−c0B)]/2`. Report A and B separately and their fixed average, with repair and breakage counts for each; never select the better realization. A zero average can hide equal repairs and breakages.

On the panel where all four verdicts are valid:

```text
intervention correctness crossing = (1[c0A != c1A] + 1[c0B != c1B]) / 2
same-prompt correctness crossing  = (1[c0A != c0B] + 1[c1A != c1B]) / 2
excess                           = intervention crossing − same-prompt crossing
```

Both sides have two opportunities. Also compute changed-native-verdict rates separately: different wrong labels are a verdict change but not a correctness crossing. Preserve possible/valid pair counts, observed events over all possible pairs, conditional valid-pair rates, complete-four-valid panel size and coverage. Invalid outputs are not semantic crossings. Below 95% complete-panel coverage, limit inference to that valid panel and withhold a population-wide controlled-sensitivity claim.

Use 10,000 paired bootstrap draws with seed 20260930. Resample **exact original claim groups**, carrying all native records in a group together; the estimate is a ratio of sampled record totals to sampled record counts. It is native-record weighted, not an unweighted mean over groups. Preserve methods, conditions and both realizations within each record/group. Duplication, partial applicability, exact-length eligibility and complete-output coverage define different stated populations; report their denominators explicitly.

| Named inferential family | Contrasts | Intervals and interpretation |
|---|---:|---|
| Raw signed accuracy | Two interventions × two models = 4 | Pointwise 95% and Bonferroni 98.75%; negative adjusted upper bound supports harm in that specific cell; positive lower bound supports improvement |
| Raw correctness-crossing excess | Two interventions × two models = 4 | Pointwise 95% and Bonferroni 98.75%; positive adjusted lower bound supports operational sensitivity beyond the observed fresh-repeat background in that cell |
| Length-matched signed accuracy | Repeated raw versus padded baseline, two interventions × two models = 4 | Same intervals, on pre-inference exact-length pairs; interpretation is relative to this particular padding control |

Families answer separate questions; no omnibus familywise guarantee is claimed. Verdict-change intervals, length-control crossing rates, all-pair length contrasts, padded-versus-unpadded effects, presentation comparisons, class-level readouts and canonical/ledger effects are descriptive. Their favorable cells cannot replace the primary contrasts. Below 95% exact-length coverage, limit a length-matched conclusion to the matched population. A null interval is inconclusive, not a proof of equality or invariance. With 491 groups and rare native classes, do not promise power for a 2 pp effect.

For each condition/presentation/realization report native accuracy, four-class macro-F1, all per-class supports/F1, invalid/overlength counts, logical tokens/calls and coverage. Report the arithmetic mean of A/B accuracy and macro-F1, not a selected realization. Secondary signed comparisons, always in both directions as observed: packet dedup versus raw; fragment dedup versus raw; counts versus identical fragment text, within each original condition. Canonical message/prediction equality is verified as a within-realization cache contract; it does not establish improved correctness.

Per-cell logical token/call totals describe standalone execution without another arm subsidizing it. Whole-study physical costs separately count each actual completion, cache reuse, charged input/output tokens and recorded inference batch seconds for A and fresh B. Cache order is not a method-efficiency advantage. Batch seconds exclude loading/host overhead; lost work from an interruption cannot be inferred from persisted rows.

## Budget and review milestones

If P of 500 records have at least two QA units, each model/realization has exactly `9 × (500 + P)` logical requests: 8 base-grid requests plus one padding control for every record, and 8 partial-grid requests plus one partial padding control for each applicable record. Maximum **9000/model/realization; 36,000 across two models and two realizations**. Unique messages and physical generations are smaller and are explicitly counted; do not quote the maximum as actual cost. Hard per-request ceilings are 8192 input/16 output tokens. CPU preparation emits actual logical/unique/overlength/mismatch counts before GPU launch. Both models may run one shard each on separate available local GPUs; root schedules them around already authorized jobs, without stopping unrelated processes.

| Stage | Work | Gate |
|---|---|---|
| M0 | Synthetic tests, preserved legacy checks, native schema/preparation/token audit | Gold absent from messages; facts/URLs retained; exact cache invariance and correct denominators |
| M1 | Fresh independent protocol/code review; model-specific prepared files and hashes | Resolve substantive defects before unused dev inference |
| M2 | Complete whole realization A then B for both models, retaining logs | No partial outcome inspection; complete frozen manifests and equal layout |
| M3 | One formal evaluation, then fresh scientific/statistical readout review | Preserve unfavorable signs and native task limits |

Preparation and evaluation bind new code, its tests, this protocol, unchanged helper code, unchanged shared runner, original data path/hash, source manifest, model/tokenizer/config versions, references, requests, audits and output manifests. Missing output records prevent formal evaluation; emitted invalid/long records remain scored failures. Do not edit bound artifacts after preparation; a necessary pre-inference correction requires a new version and receipt.

## Exact commands

Run from this reproducibility package root. CPU only:

```bash
python -m unittest discover -s paper2 -p 'test_formal.py' -v
python paper2/formal.py prepare --model-key qwen --output paper2/prepared/dev_formal_qwen_v1
python paper2/formal.py prepare --model-key granite --output paper2/prepared/dev_formal_granite_v1
python shared/generate.py --input paper2/prepared/dev_formal_qwen_v1/requests.jsonl --check-input
python shared/generate.py --input paper2/prepared/dev_formal_granite_v1/requests.jsonl --check-input
```

After independent review and an approved GPU occupancy check, these are the Qwen commands with GPU 0 as a scheduler-substitutable device. Finish A before starting B; each uses a fresh output path. Retain stdout/stderr and the literal launch command.

```bash
CUDA_VISIBLE_DEVICES=0 python shared/generate.py --input paper2/prepared/dev_formal_qwen_v1/requests.jsonl --output paper2/runs/qwen_dev_formal_v1/realization_A.jsonl --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --num-shards 1 --shard-index 0 --batch-size 8 --seed 17 --max-input-tokens 8192 --max-new-tokens 16
CUDA_VISIBLE_DEVICES=0 python shared/generate.py --input paper2/prepared/dev_formal_qwen_v1/requests.jsonl --output paper2/runs/qwen_dev_formal_v1/realization_B.jsonl --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --num-shards 1 --shard-index 0 --batch-size 8 --seed 17 --max-input-tokens 8192 --max-new-tokens 16
python paper2/formal.py evaluate --prepared paper2/prepared/dev_formal_qwen_v1 --realization-a paper2/runs/qwen_dev_formal_v1/realization_A.jsonl --realization-b paper2/runs/qwen_dev_formal_v1/realization_B.jsonl --output paper2/runs/qwen_dev_formal_v1/evaluation
```

Granite uses the identical commands with preparation/run name `qwen` replaced by `granite`, model `models/granite_3_3_8b_instruct`, revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`, and a separately approved available GPU. Do not alter shard count to use spare GPUs: the global cache and whole-file layout are part of the protocol. Paper 1's bound source/protocol/evaluator and the shared runner remain untouched.
