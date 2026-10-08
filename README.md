# When Web Evidence Repeats: Decision Stability in Claim Verification

Source release matching the final 2026-10-03 revised manuscript. This tree contains the original AVeriTeC assembly and formal measurement, the SciFact allocation supplement, the fresh native/closed-label decoder control, and the saved-score sensitivity code. It contains no manuscript, original records, labels, model weights, prepared requests, predictions, logs, or results. The Python implementations in `paper2/` and `shared/` are byte-for-byte copies of the frozen reproducibility source. The four included protocol/design documents are required by its source-hash bindings; some of their historical status prose and links to unpublished `ops/` records are archival, not current release instructions. Use this README for release paths.

The recorded environment was Python 3.11.14, PyTorch 2.9.1, Transformers 4.57.2, tokenizers 0.22.1, NumPy 1.26.4, and regex 2025.11.3. `requirements-recorded.txt` records versions; CUDA setup depends on the host. Use a self-contained Conda environment, as in the recorded run, or create a copied-interpreter environment with `python3.11 -m venv --copies .venv`; activate it and install the recorded requirements with a CUDA-compatible PyTorch build. An ordinary symlink-based venv may be rejected by the receipt helper because the frozen decoder receipt requires the resolved Python executable to belong to the active environment. The helper records whole-second UTC times aligned with the generator manifest. Generation used greedy decoding, bfloat16, SDPA, the slow local tokenizers, thinking disabled, seed 17, batch 8, one shard, at most 8192 input and 16 new tokens. No imports or synthetic tests download data, call an API, or run model inference. Run all commands at the project root with `PYTHONPATH=paper2:shared`.

## External inputs

See [DATA_INTERFACE.md](DATA_INTERFACE.md) for exact source URLs, revisions, checksums, native schemas and staging paths. The frozen code requires the official AVeriTeC dev JSON at `paper2/data/averitec_dev.json` and a byte-identical copy of [the metadata-only source manifest](metadata/averitec_source_manifest.json) beside it as `paper2/data/source_manifest.json`. The JSON may be a local runtime symlink to a user-managed external file: `native_source()` resolves both the CLI and expected paths. Do not commit the link. `supplement.py` also requires the pinned SciFact archive and three members under `paper2/data/scifact_official_20260930/`. No dataset download is automatic.

Stage the pinned Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` at `models/qwen2_5_7b_instruct/` and Granite-3.3-8B-Instruct revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` at `models/granite_3_3_8b_instruct/`. Model shards and tokenizer/config files are checked during preparation. Model directories are ignored by Git.

## Offline source checks

```bash
sha256sum -c SHA256SUMS
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=paper2:shared python -m unittest discover -s paper2 -p 'test_*.py' -q
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -p 'test_*.py' -q
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=paper2:shared python paper2/shadow_policy_sensitivity.py --self-test
```

These checks use synthetic fixtures. Passing them does not reproduce the paper's measurements.

## AVeriTeC formal assembly and measurement

The [formal protocol](paper2/FORMAL_PROTOCOL_V1.md) defines all policy cells, padding, matching, resampling, and the original formal commands. After staging inputs and models:

```bash
PYTHONPATH=paper2:shared python paper2/formal.py prepare --model-key qwen --output paper2/prepared/dev_formal_qwen_v1
PYTHONPATH=paper2:shared python paper2/formal.py prepare --model-key granite --output paper2/prepared/dev_formal_granite_v1
PYTHONPATH=paper2:shared python shared/generate.py --input paper2/prepared/dev_formal_qwen_v1/requests.jsonl --check-input
PYTHONPATH=paper2:shared python shared/generate.py --input paper2/prepared/dev_formal_granite_v1/requests.jsonl --check-input
```

Run each command below in a fresh process, A before B for the same model. Set `CUDA_VISIBLE_DEVICES` to one available device for each generation process; the commands do not choose a GPU. Retain the whole output and adjacent `.manifest.json` file.

```bash
AVAILABLE_GPU=0  # Replace with one GPU or UUID you have checked is available.
export CUDA_VISIBLE_DEVICES="$AVAILABLE_GPU"
PYTHONPATH=paper2:shared python shared/generate.py --input paper2/prepared/dev_formal_qwen_v1/requests.jsonl --output paper2/runs/qwen_dev_formal_v1/realization_A.jsonl --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --num-shards 1 --shard-index 0 --batch-size 8 --seed 17 --max-input-tokens 8192 --max-new-tokens 16
PYTHONPATH=paper2:shared python shared/generate.py --input paper2/prepared/dev_formal_qwen_v1/requests.jsonl --output paper2/runs/qwen_dev_formal_v1/realization_B.jsonl --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --num-shards 1 --shard-index 0 --batch-size 8 --seed 17 --max-input-tokens 8192 --max-new-tokens 16
PYTHONPATH=paper2:shared python shared/generate.py --input paper2/prepared/dev_formal_granite_v1/requests.jsonl --output paper2/runs/granite_dev_formal_v1/realization_A.jsonl --model models/granite_3_3_8b_instruct --model-revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b --num-shards 1 --shard-index 0 --batch-size 8 --seed 17 --max-input-tokens 8192 --max-new-tokens 16
PYTHONPATH=paper2:shared python shared/generate.py --input paper2/prepared/dev_formal_granite_v1/requests.jsonl --output paper2/runs/granite_dev_formal_v1/realization_B.jsonl --model models/granite_3_3_8b_instruct --model-revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b --num-shards 1 --shard-index 0 --batch-size 8 --seed 17 --max-input-tokens 8192 --max-new-tokens 16
PYTHONPATH=paper2:shared python paper2/formal.py evaluate --prepared paper2/prepared/dev_formal_qwen_v1 --realization-a paper2/runs/qwen_dev_formal_v1/realization_A.jsonl --realization-b paper2/runs/qwen_dev_formal_v1/realization_B.jsonl --output paper2/results/qwen_dev_formal_v1_evaluation
PYTHONPATH=paper2:shared python paper2/formal.py evaluate --prepared paper2/prepared/dev_formal_granite_v1 --realization-a paper2/runs/granite_dev_formal_v1/realization_A.jsonl --realization-b paper2/runs/granite_dev_formal_v1/realization_B.jsonl --output paper2/results/granite_dev_formal_v1_evaluation
```

Formal evaluation needs complete, fresh A/B manifests. For every step, use an output path that does not already exist. The formal protocol's sample paths above are examples for a new reproduction.

## SciFact allocation supplement

The [supplement protocol](paper2/SUPPLEMENT_PROTOCOL_V1.md) and frozen source define the full AVeriTeC/SciFact construction. Use these preparation paths exactly: the decoder control later reads them by name.

```bash
PYTHONPATH=paper2:shared python paper2/supplement.py prepare --model-key qwen --output paper2/prepared/supplement_qwen_v1
PYTHONPATH=paper2:shared python paper2/supplement.py prepare --model-key granite --output paper2/prepared/supplement_granite_v1
PYTHONPATH=paper2:shared python paper2/supplement.py verify --prepared paper2/prepared/supplement_qwen_v1
PYTHONPATH=paper2:shared python paper2/supplement.py verify --prepared paper2/prepared/supplement_granite_v1
```

Select one available CUDA device or GPU UUID yourself and set `AVAILABLE_GPU` accordingly. The helper's `capture` action prints the frozen command without launching unless `--execute` is present. Each execution writes one fresh output, its generator manifest, a child log, and a cell receipt. Keep A before B for each model. The following runs all four supplement cells sequentially; inspect one dry run first by removing `--execute` from the first invocation.

```bash
AVAILABLE_GPU=0  # Replace with one GPU or UUID you have checked is available.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
for model in qwen granite; do
  for realization in A B; do
    PYTHONPATH=paper2:shared python tools/run_with_receipt.py capture \
      --kind supplement --model-key "$model" --realization "$realization" \
      --prepared "paper2/prepared/supplement_${model}_v1" \
      --output "paper2/runs/${model}_supplement_v1/${realization}.jsonl" \
      --receipt "paper2/results/receipts/supplement_${model}_${realization}.json" \
      --log "paper2/results/logs/supplement_${model}_${realization}.log" \
      --gpu "$AVAILABLE_GPU" --execute || exit 1
  done
done
PYTHONPATH=paper2:shared python tools/run_with_receipt.py merge --kind supplement \
  --receipts paper2/results/receipts/supplement_qwen_A.json paper2/results/receipts/supplement_qwen_B.json \
             paper2/results/receipts/supplement_granite_A.json paper2/results/receipts/supplement_granite_B.json \
  --output paper2/results/supplement_v1_execution_receipt.json
```

The helper captures absolute argv, process times, selected one-GPU visibility, output/manifest hashes, failures, and allocator metadata. It refuses existing output/log/receipt paths; `merge` requires all four fresh cells and enforces the supplement's 14,400 occupied-device-second bound and no additional attempts. Do not construct receipts from old output files. Then evaluate the full set:

```bash
PYTHONPATH=paper2:shared python paper2/supplement.py evaluate --qwen-prepared paper2/prepared/supplement_qwen_v1 --qwen-a paper2/runs/qwen_supplement_v1/A.jsonl --qwen-b paper2/runs/qwen_supplement_v1/B.jsonl --granite-prepared paper2/prepared/supplement_granite_v1 --granite-a paper2/runs/granite_supplement_v1/A.jsonl --granite-b paper2/runs/granite_supplement_v1/B.jsonl --execution-receipt paper2/results/supplement_v1_execution_receipt.json --output paper2/results/supplement_v1_evaluation
```

The evaluator enforces a 14,400 occupied-device-second bound. A prior private receipt is not needed for a fresh reproduction.

## Fresh native/closed-label decoder control and sensitivity

The [decoder control plan](paper2/DECODER_CONTROL_PLAN_20261002.md) defines the frozen design; this release command sequence uses the receipt helper and the output names required by the unchanged sensitivity reader. Decoder preparation reuses the exact two supplement preparations above, so complete that stage first. The helper sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only for Granite closed runs below. Naming their fresh outputs `closed_A_retry1.jsonl` and `closed_B_retry1.jsonl` preserves compatibility with the sensitivity script; it does not claim a failed attempt occurred. Any actual failed attempt remains in its own cell receipt and log and must be included when merging the successful set.

```bash
PYTHONPATH=paper2:shared python paper2/decoder_control.py prepare --model-key qwen --output paper2/prepared/decoder_control_qwen_20261002_v2
PYTHONPATH=paper2:shared python paper2/decoder_control.py prepare --model-key granite --output paper2/prepared/decoder_control_granite_20261002_v2
PYTHONPATH=paper2:shared python paper2/decoder_control.py verify --prepared paper2/prepared/decoder_control_qwen_20261002_v2
PYTHONPATH=paper2:shared python paper2/decoder_control.py verify --prepared paper2/prepared/decoder_control_granite_20261002_v2
AVAILABLE_GPU=0  # Replace with one GPU or UUID you have checked is available.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
for model in qwen granite; do
  for decoder in native closed; do
    suffix=''
    allocator=()
    if [[ "$model" == granite && "$decoder" == closed ]]; then
      suffix='_retry1'
      allocator=(--expandable-allocator)
    fi
    for realization in A B; do
      PYTHONPATH=paper2:shared python tools/run_with_receipt.py capture \
        --kind decoder --model-key "$model" --decoder "$decoder" --realization "$realization" \
        --prepared "paper2/prepared/decoder_control_${model}_20261002_v2" \
        --output "paper2/runs/${model}_decoder_control_20261002_v2/${decoder}_${realization}${suffix}.jsonl" \
        --receipt "paper2/results/receipts/decoder_${model}_${decoder}_${realization}.json" \
        --log "paper2/results/logs/decoder_${model}_${decoder}_${realization}.log" \
        --gpu "$AVAILABLE_GPU" "${allocator[@]}" --execute || exit 1
    done
  done
done
PYTHONPATH=paper2:shared python tools/run_with_receipt.py merge --kind decoder \
  --receipts paper2/results/receipts/decoder_qwen_native_A.json paper2/results/receipts/decoder_qwen_native_B.json \
             paper2/results/receipts/decoder_qwen_closed_A.json paper2/results/receipts/decoder_qwen_closed_B.json \
             paper2/results/receipts/decoder_granite_native_A.json paper2/results/receipts/decoder_granite_native_B.json \
             paper2/results/receipts/decoder_granite_closed_A.json paper2/results/receipts/decoder_granite_closed_B.json \
  --output paper2/results/decoder_control_20261002_v2_execution_receipt.json
PYTHONPATH=paper2:shared python paper2/decoder_control.py evaluate \
  --qwen-prepared paper2/prepared/decoder_control_qwen_20261002_v2 \
  --qwen-native-a paper2/runs/qwen_decoder_control_20261002_v2/native_A.jsonl \
  --qwen-native-b paper2/runs/qwen_decoder_control_20261002_v2/native_B.jsonl \
  --qwen-closed-a paper2/runs/qwen_decoder_control_20261002_v2/closed_A.jsonl \
  --qwen-closed-b paper2/runs/qwen_decoder_control_20261002_v2/closed_B.jsonl \
  --granite-prepared paper2/prepared/decoder_control_granite_20261002_v2 \
  --granite-native-a paper2/runs/granite_decoder_control_20261002_v2/native_A.jsonl \
  --granite-native-b paper2/runs/granite_decoder_control_20261002_v2/native_B.jsonl \
  --granite-closed-a paper2/runs/granite_decoder_control_20261002_v2/closed_A_retry1.jsonl \
  --granite-closed-b paper2/runs/granite_decoder_control_20261002_v2/closed_B_retry1.jsonl \
  --execution-receipt paper2/results/decoder_control_20261002_v2_execution_receipt.json \
  --output paper2/results/decoder_control_20261002_v2_evaluation
```

The decoder evaluator reads `paper2/results/supplement_v1_evaluation/summary.json` and records its hash as historical context; the supplement evaluation above supplies it. Its fresh evaluation is saved to `paper2/results/decoder_control_20261002_v2_evaluation` for the sensitivity script's fixed lookup. The sensitivity script checks complete generated outputs and saved logits; it does not run another model forward pass.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper2:shared python paper2/shadow_policy_sensitivity.py --self-test
PYTHONPATH=paper2:shared python paper2/shadow_policy_sensitivity.py --root . --output paper2/results/shadow_policy_sensitivity_reproduced
```

None of the model-data preparation, generation, evaluation, or sensitivity commands was rerun for this source release. The release validation is limited to offline synthetic tests and source-byte checks; reproducing published aggregate measurements requires the exact external inputs, weights, complete fresh outputs, and contemporaneous receipts.
