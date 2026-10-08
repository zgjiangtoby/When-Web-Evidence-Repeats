"""Post-review descriptive policy contrasts from saved first-forward raw scores.

No new inference or official prediction replacement. Uses the frozen component
bootstrap and all native records; bare-letter encoding is unchanged.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from decoder_control import contrast_records, group_interval
from supplement import ARMS, SCI_LABELS, pr


def choose(candidates, field):
    if not candidates or any(not math.isfinite(v[field]) for v in candidates.values()):
        raise ValueError("Missing or nonfinite legal-label scores")
    return min(candidates, key=lambda label: (-candidates[label][field], candidates[label]["token_id"]))


def self_test():
    scores = {"B": {"token_id": 33, "raw_logit": 1.0}, "A": {"token_id": 32, "raw_logit": 1.0}}
    assert choose(scores, "raw_logit") == "A"
    scores["B"]["raw_logit"] = 2.0
    assert choose(scores, "raw_logit") == "B"
    scores["B"]["raw_logit"] = float("nan")
    try:
        choose(scores, "raw_logit")
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite scores must be rejected")
    assert exact_name_alias("SUPPORT", "scifact") == "SUPPORT"
    assert exact_name_alias(" S ", "scifact") is None
    assert exact_name_alias("SUPPORT because ...", "scifact") is None
    assert exact_name_alias("support", "scifact") is None
    assert exact_name_alias("Refuted", "averitec") == "Refuted"


def exact_name_alias(response, corpus):
    """Case-sensitive complete native class names only; no ambiguous initials."""
    name = response.strip()
    labels = pr.LABELS if corpus == "averitec" else SCI_LABELS
    return name if name in labels else None


def analyze(root):
    folder = root / "paper2/results/decoder_control_20261002_v2_evaluation"
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    hashes = {str(summary_path.relative_to(root)): sha256(summary_path.read_bytes()).hexdigest()}
    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(root))] = sha256(data).hexdigest()
        return [json.loads(line) for line in data.splitlines()]
    def indexed(rows):
        result = {corpus: {} for corpus in ARMS}
        for row in rows:
            key = row["claim_id"], row["condition"]
            target = result[row["corpus"]]
            if key in target:
                raise ValueError("Duplicate policy outcome")
            target[key] = row
        return result
    reports = {}
    for model in ("qwen", "granite"):
        official, shadows, aliases, cells = {}, {}, {}, {}
        alias_events = Counter()
        for realization in ("A", "B"):
            for decoder in ("native", "closed"):
                name = f"{model}_{decoder}_{realization}_outcomes.jsonl"
                path = folder / name
                rows = read(path)
                if len(rows) != 4672 or hashes[str(path.relative_to(root))] != summary["outcome_files"][name]:
                    raise ValueError("Frozen evaluated outcomes changed")
                official[decoder, realization] = indexed(rows)
                suffix = "_retry1" if model == "granite" and decoder == "closed" else ""
                raw_path = root / f"paper2/runs/{model}_decoder_control_20261002_v2/{decoder}_{realization}{suffix}.jsonl"
                manifest_path = raw_path.with_suffix(".jsonl.manifest.json")
                manifest = json.loads(manifest_path.read_text())
                hashes[str(manifest_path.relative_to(root))] = sha256(manifest_path.read_bytes()).hexdigest()
                if manifest != summary["models"][model]["generation_manifests"][f"{decoder}_{realization}"]:
                    raise ValueError("Run manifest changed")
                raw_rows = read(raw_path)
                raw = {r["id"]: r for r in raw_rows}
                if len(raw) != 4672 or len(raw_rows) != 4672 or set(raw) != {r["id"] for r in rows}:
                    raise ValueError("Incomplete raw/generated outcome identity")
                if decoder == "native":
                    alias_rows = []
                    for row in rows:
                        item = raw[row["id"]]
                        if item["run_signature"] != manifest["run_signature"]:
                            raise ValueError("Native raw run binding changed")
                        prediction = row["prediction"]
                        if (prediction is None and item["status"] == "ok" and row["output_status"] != "construction_length_mismatch"):
                            prediction = exact_name_alias(item["response"], row["corpus"])
                        if prediction is not None and row["prediction"] is None:
                            alias_events[f"{row['corpus']}/{realization}/recovered"] += 1
                            alias_events[f"{row['corpus']}/{realization}/correct_repairs"] += int(prediction == row["label"])
                        alias_rows.append({**row, "prediction": prediction, "correct": int(prediction == row["label"])})
                    aliases["native", realization] = indexed(alias_rows)
                    continue
                shadow_rows = []
                for row in rows:
                    item = raw[row["id"]]
                    if item["run_signature"] != manifest["run_signature"]:
                        raise ValueError("Raw run binding changed")
                    prediction = None
                    if item["status"] == "ok" and row["output_status"] != "construction_length_mismatch":
                        first = item["first_step"]
                        letter = choose(first["candidates"], "raw_logit")
                        if (letter != first["raw_logit_shadow_argmax"] or choose(first["candidates"], "processed_score") != item["response"]
                                or first["additional_forward_passes"] != 0):
                            raise ValueError("Saved first-step decision is inconsistent")
                        labels = pr.LABELS if row["corpus"] == "averitec" else SCI_LABELS
                        prediction = labels["ABCD".index(letter)]
                    new = {**row, "prediction": prediction, "correct": int(prediction == row["label"])}
                    shadow_rows.append(new)
                shadows["closed", realization] = indexed(shadow_rows)
                for corpus, arms in ARMS.items():
                    for arm in arms:
                        subset = [r for r in shadow_rows if r["corpus"] == corpus and r["condition"] == arm]
                        cells[f"{corpus}/{arm}/{realization}"] = {"n": len(subset), "correct": sum(r["correct"] for r in subset)}
                        if cells[f"{corpus}/{arm}/{realization}"]["correct"] != summary["models"][model]["first_forward_shadow_diagnostic"][f"{corpus}/{arm}/{realization}"]["raw_logit_shadow_correct"]:
                            raise ValueError("Shadow cell does not recover recorded diagnostic")
        shadow_indexes = {**official, **shadows}
        contrasts = {}
        for corpus, treatment, control in (("scifact", "R", "O"), ("scifact", "B", "O"), ("averitec", "S", "B")):
            raw_records = contrast_records(shadow_indexes, corpus, treatment, control)
            processed_records = contrast_records(official, corpus, treatment, control)
            if len(raw_records) != (209 if corpus == "scifact" else 500):
                raise ValueError("Full population required")
            alias_records = contrast_records({**official, **aliases}, corpus, treatment, control)
            contrasts[f"{corpus}/{treatment}-{control}"] = {
                "raw_shadow_minus_control": group_interval([(r["group"], r["closed"]["difference"]) for r in raw_records]),
                "raw_shadow_interaction_with_native": group_interval([(r["group"], r["interaction"]) for r in raw_records]),
                "raw_minus_processed_policy_contrast": group_interval([(a["group"], a["closed"]["difference"] - b["closed"]["difference"]) for a, b in zip(raw_records, processed_records)]),
                "exact_name_alias_policy_contrast": group_interval([(r["group"], r["native"]["difference"]) for r in alias_records]),
                "closed_minus_exact_name_alias_interaction": group_interval([(r["group"], r["interaction"]) for r in alias_records])}
        reports[model] = {"cells": cells, "contrasts": contrasts, "alias_recoveries": dict(alias_events)}
    return {"scope": "post-review descriptive 95% component intervals from saved raw logits and an exact-full-native-class-name alias parser; ambiguous initials remain invalid; no new inference, official prediction replacement, alternative-token generation, or confirmatory family",
            "input_sha256": hashes, "models": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("shadow sensitivity self-check passed")
    else:
        if args.root is None or args.output is None:
            parser.error("--root and --output required")
        args.output.mkdir(parents=True, exist_ok=False)
        result = analyze(args.root.resolve())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({m: {k: v["raw_shadow_interaction_with_native"] for k, v in r["contrasts"].items()} for m, r in result["models"].items()}))
