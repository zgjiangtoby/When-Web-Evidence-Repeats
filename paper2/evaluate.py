"""Native-label scoring and claim-group paired effects; no lineage judgments."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from generate import load_predictions, read_requests, sha
from prepare import LABELS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def outcomes(references, predictions):
    rows = []
    for reference in references:
        result = predictions.get(reference["id"], {})
        response = result.get("response")
        response = response.strip() if isinstance(response, str) else None
        valid = result.get("status") == "ok" and response in tuple("ABCD")
        label = LABELS["ABCD".index(response)] if valid else None
        rows.append({**reference, "prediction": label, "correct": int(label == reference["label"]),
                     "logical_input_tokens": result.get("input_tokens", 0),
                     "logical_output_tokens": result.get("output_tokens", 0),
                     "output_status": "valid" if valid else result.get("status", "missing")
                     if result.get("status") != "ok" else "invalid_format"})
    return rows


def metrics(rows):
    n = len(rows)
    if not n:
        raise ValueError("cannot score an empty population")
    f1 = {}
    for label in LABELS:
        tp = sum(r["prediction"] == label and r["label"] == label for r in rows)
        fp = sum(r["prediction"] == label and r["label"] != label for r in rows)
        fn = sum(r["prediction"] != label and r["label"] == label for r in rows)
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"records": n, "exact_claim_groups": len({r["exact_claim_group"] for r in rows}),
            "accuracy": sum(r["correct"] for r in rows) / n,
            "macro_f1": sum(f1.values()) / len(LABELS), "per_class_f1": f1,
            "valid_output_fraction": sum(r["output_status"] == "valid" for r in rows) / n,
            "logical_input_tokens": sum(r["logical_input_tokens"] for r in rows),
            "logical_output_tokens": sum(r["logical_output_tokens"] for r in rows),
            "output_status_counts": dict(Counter(r["output_status"] for r in rows))}


def paired_effect(rows, treatment, control, replicates=2000, seed=20260929):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    arms = defaultdict(dict)
    for row in rows:
        key = (row["condition"], row["method"])
        if key in (treatment, control):
            if key in arms[row["claim_id"]]:
                raise ValueError("duplicate claim-condition-method row")
            arms[row["claim_id"]][key] = row
    differences, flips, valid_pairs = defaultdict(list), 0, 0
    for pair in arms.values():
        if treatment not in pair or control not in pair:
            continue
        a, b = pair[treatment], pair[control]
        if a["exact_claim_group"] != b["exact_claim_group"] or a["label"] != b["label"]:
            raise ValueError("paired condition changed reference identity or label")
        differences[a["exact_claim_group"]].append(a["correct"] - b["correct"])
        if a["prediction"] is not None and b["prediction"] is not None:
            valid_pairs += 1
            flips += a["prediction"] != b["prediction"]
    groups = list(differences.values())
    if not groups:
        return {"treatment": treatment, "control": control, "paired_records": 0, "status": "not_applicable"}
    totals = [(sum(group), len(group)) for group in groups]
    n = sum(count for _, count in totals)
    rng, samples = random.Random(seed), []
    for _ in range(replicates):
        chosen = [totals[rng.randrange(len(totals))] for _ in totals]
        samples.append(sum(total for total, _ in chosen) / sum(count for _, count in chosen))
    samples.sort()
    return {"treatment": treatment, "control": control, "paired_records": n,
            "exact_claim_groups": len(groups), "accuracy_difference": sum(x for x, _ in totals) / n,
            "ci95_group_percentile": [samples[int(.025 * (replicates - 1))],
                                      samples[int(.975 * (replicates - 1))]],
            "valid_output_pairs": valid_pairs, "verdict_flips_on_valid_pairs": flips,
            "bootstrap_replicates": replicates, "bootstrap_seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "evaluation"), required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new result path")
    requests = {r["id"]: r for r in read_requests(args.requests)}
    references = read_jsonl(args.references)
    if len({r["id"] for r in references}) != len(references) or {r["id"] for r in references} != requests.keys():
        raise ValueError("references and requests must have the same unique IDs")
    for reference in references:
        if reference["label"] not in LABELS:
            raise ValueError("unrecognized native label")
        for key in ("claim_id", "exact_claim_group", "condition", "method"):
            if reference[key] != requests[reference["id"]]["metadata"][key]:
                raise ValueError("reference and inference metadata mismatch")
    input_sha = sha256(args.requests.read_bytes()).hexdigest()
    predictions, config, _ = load_predictions(args.predictions, args.requests)
    rows = outcomes(references, predictions)
    repeated_inputs = defaultdict(list)
    by_id = {row["id"]: row for row in rows}
    for key, request in requests.items():
        repeated_inputs[sha(request["messages"])].append(by_id[key])
    repeated_inputs = [group for group in repeated_inputs.values() if len(group) > 1]
    cells = defaultdict(list)
    for row in rows:
        cells[(row["condition"], row["method"])].append(row)
    conditions = sorted({c for c, _ in cells})
    methods = sorted({m for _, m in cells})
    contrasts = []
    for method in methods:
        contrasts.extend(((condition, method), ("clean", method))
                         for condition in conditions if condition.startswith("duplicate"))
        contrasts.extend(((condition, method), ("partial1", method))
                         for condition in conditions if condition.startswith("partial") and condition != "partial1")
    for condition in conditions:
        contrasts.append(((condition, "exact_fragment_dedup"), (condition, "raw_packets")))
        contrasts.append(((condition, "ledger_counts"), (condition, "exact_fragment_dedup")))
    report = {"stage": args.stage, "evidence_type": "native_provided_evidence_verification",
              "not_end_to_end_retrieval": True, "new_human_annotations": 0,
              "input_sha256": input_sha, "references_sha256": sha256(args.references.read_bytes()).hexdigest(),
              "run_config": config, "expected_requests": len(requests),
              "observed_requests": len(predictions), "missing_outputs_counted_wrong": len(requests) - len(predictions),
              "identical_message_groups": len(repeated_inputs),
              "identical_message_groups_with_native_disagreement": sum(
                  len({row["prediction"] for row in group if row["prediction"] is not None}) > 1
                  for group in repeated_inputs),
              "identical_message_groups_with_any_failure": sum(
                  any(row["prediction"] is None for row in group) for group in repeated_inputs),
              "cells": {f"{c}/{m}": metrics(group) for (c, m), group in sorted(cells.items())},
              "paired_effects": [paired_effect(rows, a, b, args.bootstrap) for a, b in contrasts],
              "charged_input_tokens": sum(r.get("charged_input_tokens", 0) for r in predictions.values()),
              "charged_output_tokens": sum(r.get("charged_output_tokens", 0) for r in predictions.values()),
              "cached_logical_requests": sum(bool(r.get("cached")) for r in predictions.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "observed": len(predictions), "expected": len(requests)}))


if __name__ == "__main__":
    main()
