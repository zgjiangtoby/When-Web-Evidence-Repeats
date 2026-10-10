"""Native-label scoring and claim-group paired effects; no lineage judgments."""

from collections import Counter
import json

from evidence import LABELS


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
