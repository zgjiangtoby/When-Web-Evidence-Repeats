# When Web Evidence Repeats

Python 3.11 and a CUDA GPU are required for model inference.
Install `requirements.txt` in Conda or a `venv --copies` environment.

External inputs:

- `data/averitec_dev.json`: [AVeriTeC dev](https://huggingface.co/chenxwh/AVeriTeC/resolve/37a96094b68f6c7977734a0d35c7c02a1b1eef07/data/dev.json). The source manifest is included.
- `data/scifact_official_20260930/`: [SciFact archive](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz), named `data.tar.gz`, and its `corpus.jsonl`, `claims_dev.jsonl`, `claims_train.jsonl`. Expected checksums are in `allocation.py`.
- `models/qwen2_5_7b_instruct/`: [Qwen2.5-7B-Instruct, pinned revision](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28).
- `models/granite_3_3_8b_instruct/`: [Granite-3.3-8B-Instruct, pinned revision](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct/tree/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b).

Run from this directory. Use unused output paths and an available GPU.
Example for the main AVeriTeC measurement:

```bash
export CUDA_VISIBLE_DEVICES=0
python measurement.py prepare --model-key qwen --output prepared/measurement_qwen
for r in A B; do
  python generate.py --input prepared/measurement_qwen/requests.jsonl --output runs/measurement_qwen/$r.jsonl --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --batch-size 8 --max-input-tokens 8192 --max-new-tokens 16 --num-shards 1 --shard-index 0 --seed 17
done
python measurement.py evaluate --prepared prepared/measurement_qwen --realization-a runs/measurement_qwen/A.jsonl --realization-b runs/measurement_qwen/B.jsonl --output results/measurement_qwen
```

`allocation.py` runs the AVeriTeC/SciFact allocation experiment;
`decoder_control.py` compares native and closed-label generation;
`sensitivity.py` analyzes saved logits. Their arguments are available with `--help`.
Decoder preparation reads `prepared/supplement_{qwen,granite}_v1`;
decoder evaluation reads `results/supplement_v1_evaluation/summary.json`.
Sensitivity reads `results/decoder_control_20261002_v2_evaluation` and
`runs/{qwen,granite}_decoder_control_20261002_v2`; the Granite closed outputs
use the names `closed_A_retry1.jsonl` and `closed_B_retry1.jsonl`.
Capture allocation/decoder runs with `run_with_receipt.py capture --kind supplement`
or `--kind decoder`; add `--execute` to launch, then `merge` the cell receipts.
Granite closed runs use `--expandable-allocator`.
Generate new preparations for this layout; previous source bindings are incompatible.
