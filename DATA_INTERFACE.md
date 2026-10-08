# External data and model interface

This source project contains no original record, label, partition, prediction, feature, weight, or execution-result file. Paths below are runtime inputs ignored by Git. Download and verify inputs independently; no code in this release downloads them automatically. Existing public labels are used as supplied, with no new annotation.

## AVeriTeC

- Official dataset: <https://fever.ai/dataset/averitec.html>.
- Frozen repository revision: `chenxwh/AVeriTeC@37a96094b68f6c7977734a0d35c7c02a1b1eef07`.
- Dev source: <https://huggingface.co/chenxwh/AVeriTeC/resolve/37a96094b68f6c7977734a0d35c7c02a1b1eef07/data/dev.json>; SHA-256 `499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300`; 500 records. Public source metadata and aggregate counts are in [averitec_source_manifest.json](metadata/averitec_source_manifest.json).
- Runtime path: `paper2/data/averitec_dev.json`; copy the official file there or create a runtime symlink with that name. Copy the released metadata JSON to `paper2/data/source_manifest.json` byte-for-byte. The source verifier checks the expected development-file hash, source revision, record count, and this manifest. The `--source` CLI argument defaults to that local path. When overriding it, both the argument and the local default must resolve to the same file.
- Native schema relevant to this code: top-level list of records containing `claim` (original text), `label` (native four-way gold), and `questions` (list). Each question contains `question` and `answers`; each answer may contain `answer`, `boolean_explanation`, and `source_url`. `prepare.py` reads only these permitted fields for messages. Source indices and original exact claim text define record identities and resampling groups. The input's other official fields are not new annotations or model features in this source.

## SciFact

- Official source script, pinned revision: <https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/script/download-data.sh>. It references <https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz>; `latest` is mutable, so verify archive SHA-256 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be` before using it.
- Runtime directory: `paper2/data/scifact_official_20260930/` with archive `data.tar.gz` and extracted `corpus.jsonl`, `claims_dev.jsonl`, `claims_train.jsonl`. Frozen SHA-256 values are in `paper2/supplement.py::SCI_FILES`; all four files are checked. The train file supports a source-exposure audit and is not used for training.
- Relevant native schema: `corpus.jsonl` has document `doc_id`, `title`, and `abstract` (ordered sentence list). `claims_dev.jsonl` has claim `id`, `claim`, and `evidence`, a map from document ID to existing rationale entries with `label` and zero-based `sentences` indices. The supplement keeps the released SUPPORT/CONTRADICT labels and full abstract. Its sampled rationale and non-rationale target units are derived from these existing labels; no new judgments are collected. `claims_train.jsonl` is checked by hash for overlap/exposure analysis.

## Models

| Runtime directory | Repository and immutable revision |
| --- | --- |
| `models/qwen2_5_7b_instruct/` | <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28> |
| `models/granite_3_3_8b_instruct/` | <https://huggingface.co/ibm-granite/granite-3.3-8b-instruct/tree/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b> |

`formal.py` binds tokenizer/configuration files and package versions. `supplement.py` additionally checks the four expected weight-shard hashes per model. The generation scripts load local checkpoints offline. The recorded versions and generation settings are in [README.md](README.md) and `requirements-recorded.txt`.
