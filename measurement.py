#!/usr/bin/env python3
"""Frozen provided-evidence measurement with whole-file fresh repeats and length controls."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import metrics as ev
import evidence as pr
from generate import load_local_tokenizer, load_predictions, sha
from replay import extract_units

HERE = Path(__file__).resolve().parent
ROOT = HERE
VERSION = "averitec_measurement_v1"
DEV_SHA = "499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300"
RUNNER_SHA = "f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c"
DATA_REVISION = "37a96094b68f6c7977734a0d35c7c02a1b1eef07"
MODELS = {
    "qwen": ("qwen2_5_7b_instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "granite": ("granite_3_3_8b_instruct", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"),
}
PAD_INSTRUCTION = " Text inside <non_evidence_padding> is artificial filler, not evidence; ignore it when judging the claim."
PAD_START = "\n\nNon-evidence padding (ignore for verification):\n<non_evidence_padding>\n"
PAD_END = "\n</non_evidence_padding>"
FILLER = " ."
MAX_FILLER = 32768
BOOTSTRAP = 10000
SEED = 20260930
RAW_PAIRS = (("duplicate4", "clean"), ("partial4", "partial1"))
PAD_PAIRS = (("duplicate4", "clean_length_control"), ("partial4", "partial1_length_control"))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bindings():
    names = ("measurement.py", "evidence.py", "replay.py", "metrics.py", "generate.py")
    result = {name: file_sha(ROOT / name) for name in names}
    if result["generate.py"] != RUNNER_SHA:
        raise ValueError("Native generator changed")
    return result


def model_binding(key):
    directory, revision = MODELS[key]
    path = ROOT / "models" / directory
    names = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
    return {"key": key, "path": str(path.resolve()), "revision": revision,
            "files": {n: file_sha(path / n) for n in names if (path / n).is_file()},
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers")},
            "tokenizer_use_fast": False}


def native_source(path):
    records, digest = pr.load_records(path)
    manifest_path = path.parent / "source_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if (path.resolve() != (HERE / "data" / "averitec_dev.json").resolve() or digest != DEV_SHA
            or len(records) != 500 or source_manifest.get("revision") != DATA_REVISION
            or source_manifest["files"][path.name]["sha256"] != DEV_SHA):
        raise ValueError("Use the complete pinned official development source")
    return records, {"path": str(path.resolve()), "sha256": digest, "revision": DATA_REVISION,
                     "source_manifest_path": str(manifest_path.resolve()),
                     "source_manifest_sha256": file_sha(manifest_path)}


def wrapped(request, repetitions=0):
    result = deepcopy(request)
    result["messages"][0]["content"] += PAD_INSTRUCTION
    result["messages"][1]["content"] += PAD_START + FILLER * repetitions + PAD_END
    return result


def token_counter(tokenizer):
    cache = {}

    def count(messages):
        key = sha(messages)
        if key not in cache:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 thinking=False, enable_thinking=False)
            cache[key] = len(tokenizer.encode(text, add_special_tokens=False))
        return cache[key]
    return count


def padding_control(base, target_tokens, count):
    """Frozen bounded search on the complete rendered chat, not a token-prefix surrogate."""
    observed = {}

    def probe(k):
        k = min(MAX_FILLER, max(0, k))
        if k not in observed:
            candidate = wrapped(base, k)
            observed[k] = (count(candidate["messages"]), candidate)
        return k, observed[k][0]

    k, base_tokens = probe(0)
    for _ in range(16):
        length = observed[k][0]
        if length == target_tokens:
            break
        proposed = min(MAX_FILLER, max(0, k + target_tokens - length))
        if proposed in observed:
            break
        k, _ = probe(proposed)
    best = min(observed, key=lambda n: (abs(observed[n][0] - target_tokens), n))
    if observed[best][0] != target_tokens:
        center = best
        for offset in range(-16, 17):
            probe(center + offset)
        best = min(observed, key=lambda n: (abs(observed[n][0] - target_tokens), n))
    actual, result = observed[best]
    return result, {"target_tokens": target_tokens, "actual_tokens": actual,
                    "base_tokens": base_tokens, "filler_repetitions": best,
                    "token_mismatch": actual - target_tokens, "exact_length_match": actual == target_tokens,
                    "evaluated_filler_lengths": sorted(observed), "search_candidates": len(observed)}


def build_requests(records, tokenizer):
    requests, references, padding, source_audit = [], [], [], []
    count = token_counter(tokenizer)
    for index, record in enumerate(records):
        claim_id = f"dev:{index}"
        raw, refs = pr.build_claim(record, claim_id, "dev", copies=4)
        units = extract_units(record)
        original_content = pr.render_ledger(pr.ledger([[unit] for unit in units]))
        anchor = pr.anchor_index(claim_id, len(units))
        for req in raw:
            condition, method = req["metadata"]["condition"], req["metadata"]["method"]
            mode, copies = {"clean": ("clean", 1), "duplicate4": ("duplicate", 4),
                            "partial1": ("partial_copy", 1), "partial4": ("partial_copy", 4)}[condition]
            packets = pr.make_replay(units, mode, copies, anchor)
            visible = pr.deduplicate_packets(packets) if method == "exact_packet_dedup" else packets
            entries = pr.ledger(visible)
            # Duplicate source IDs may collapse; distinct QA text and its URL union must remain.
            if pr.render_ledger(entries) != original_content:
                raise ValueError(f"{req['id']}: presentation changed original QA text or source URLs")
            content = (pr.render_ledger(entries, show_counts=method == "ledger_counts")
                       if method in ("exact_fragment_dedup", "ledger_counts") else pr.render_packets(visible))
            if req["messages"][1]["content"] != "Claim:\n" + record["claim"] + "\n\nProvided evidence:\n" + content:
                raise ValueError(f"{req['id']}: rendered evidence does not reconstruct")
        source_audit.append({"claim_id": claim_id, "native_record_index_0based": index,
                             "exact_claim_group": pr.digest(record["claim"]), "qa_units": len(units),
                             "partial_applicable": len(units) >= 2,
                             "permitted_evidence_sha256": sha(units), "claim_characters": len(record["claim"]),
                             "native_label": pr.canonical_label(record["label"]), "original_label": record["label"]})
        base = {r["id"]: r for r in raw}
        modified = [wrapped(r) for r in raw]
        fragment = [r["messages"] for r in modified if r["metadata"]["method"] == "exact_fragment_dedup"]
        if any(m != fragment[0] for m in fragment):
            raise ValueError("Canonicalization changed evidence or its frozen equality contract")
        requests.extend(modified)
        references.extend(refs)
        for target, control in RAW_PAIRS:
            key = f"{claim_id}/{control}/raw_packets"
            if key not in base:
                continue
            target_request = wrapped(base[f"{claim_id}/{target}/raw_packets"])
            result, audit = padding_control(base[key], count(target_request["messages"]), count)
            condition = control + "_length_control"
            result["id"] = f"{claim_id}/{condition}/raw_packets"
            result["metadata"]["condition"] = condition
            requests.append(result)
            reference = next(r for r in refs if r["id"] == key)
            references.append({**reference, "id": result["id"], "condition": condition})
            padding.append({**audit, "id": result["id"], "claim_id": claim_id,
                            "exact_claim_group": result["metadata"]["exact_claim_group"],
                            "baseline_id": key, "target_id": target_request["id"],
                            "padded_messages_sha256": sha(result["messages"]),
                            "target_messages_sha256": sha(target_request["messages"])})
    lengths = [{"id": r["id"], "prompt_sha256": sha(r["messages"]),
                "input_tokens": count(r["messages"]), "over_input_ceiling": count(r["messages"]) > 8192}
               for r in requests]
    return requests, references, padding, source_audit, lengths


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a fresh formal preparation directory")
    records, source = native_source(args.source)
    model = model_binding(args.model_key)
    tokenizer = load_local_tokenizer(Path(model["path"]))
    artifacts = build_requests(records, tokenizer)
    args.output.mkdir(parents=True)
    names = ("requests.jsonl", "references.jsonl", "padding_audit.jsonl", "source_audit.jsonl", "length_audit.jsonl")
    for name, rows in zip(names, artifacts):
        write_rows(args.output / name, rows)
    requests, _, padding, audit, lengths = artifacts
    manifest = {"version": VERSION, "source": source, "model": model, "source_binding": bindings(),
                "files": {n: file_sha(args.output / n) for n in names}, "native_records": len(records),
                "exact_claim_groups": len({r["exact_claim_group"] for r in audit}),
                "partial_inapplicable_ids": [r["claim_id"] for r in audit if not r["partial_applicable"]],
                "native_label_counts": dict(Counter(r["native_label"] for r in audit)),
                "logical_requests_per_realization": len(requests), "required_fresh_realizations": 2,
                "unique_messages_per_realization": len({sha(r["messages"]) for r in requests}),
                "padding_controls": len(padding), "padding_exact_matches": sum(r["exact_length_match"] for r in padding),
                "overlength_logical_requests": sum(r["over_input_ceiling"] for r in lengths),
                "new_human_annotations": 0, "natural_lineage_labels": False}
    write_json(args.output / "preparation.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("native_records", "exact_claim_groups", "logical_requests_per_realization",
                                               "unique_messages_per_realization", "padding_controls", "padding_exact_matches",
                                               "overlength_logical_requests")}))


def load_prepared(path):
    manifest = json.loads((path / "preparation.json").read_text())
    if manifest.get("version") != VERSION or manifest["source_binding"] != bindings():
        raise ValueError("Source binding changed")
    records, source = native_source(Path(manifest["source"]["path"]))
    model = model_binding(manifest["model"]["key"])
    if source != manifest["source"] or model != manifest["model"]:
        raise ValueError("Native source or tokenizer/checkpoint binding changed")
    for name, digest in manifest["files"].items():
        if file_sha(path / name) != digest:
            raise ValueError("Prepared artifact hash changed: " + name)
    expected = build_requests(records, load_local_tokenizer(Path(model["path"])))
    for name, rows in zip(("requests.jsonl", "references.jsonl", "padding_audit.jsonl", "source_audit.jsonl", "length_audit.jsonl"), expected):
        if ev.read_jsonl(path / name) != rows:
            raise ValueError("Prepared artifact does not reconstruct: " + name)
    return manifest, expected


def checked_run(path, request_file, requests, lengths, model):
    predictions, config, manifests = load_predictions([path], request_file)
    manifest = manifests[0]
    required = {"code_sha256": RUNNER_SHA, "model_path": model["path"], "model_revision": model["revision"],
                "model_config_sha256": model["files"]["config.json"], "num_shards": 1, "batch_size": 8,
                "max_input_tokens": 8192, "max_new_tokens": 16, "seed": 17, "do_sample": False,
                "thinking": False, "dtype": "bfloat16", "attention": "sdpa", "tokenizer_use_fast": False}
    if (any(config.get(k) != v for k, v in required.items()) or manifest["config"].get("shard_index") != 0
            or manifest.get("total_requests") != len(requests) or manifest.get("completed") != len(requests)
            or set(predictions) != {r["id"] for r in requests}
            or any(config.get("versions", {}).get(k) != v for k, v in model["versions"].items())):
        raise ValueError("Incomplete or nonfrozen one-shard run")
    expected_lengths = {r["id"]: r["input_tokens"] for r in lengths}
    seen = {}
    for req in requests:
        row, key = predictions[req["id"]], sha(req["messages"])
        if row.get("status") not in ("ok", "input_too_long") or row.get("input_tokens") != expected_lengths[req["id"]]:
            raise ValueError("Output status or complete rendered token length mismatch")
        if (row["status"] == "input_too_long") != (row["input_tokens"] > 8192):
            raise ValueError("Length rejection does not follow frozen ceiling")
        if type(row.get("cached")) is not bool or row["cached"] != (key in seen):
            raise ValueError("Run did not preserve one global exact-message cache")
        if key in seen:
            origin = predictions[seen[key]]
            if (row.get("cache_of") != seen[key] or any(row.get(k) != origin.get(k) for k in
                                                      ("status", "response", "input_tokens", "output_tokens"))):
                raise ValueError("Identical messages have different within-realization outputs")
        else:
            seen[key] = row["id"]
        for k in ("input_tokens", "output_tokens", "charged_input_tokens", "charged_output_tokens", "batch_size"):
            if type(row.get(k)) is not int or row[k] < 0:
                raise ValueError("Invalid recorded cost")
        elapsed = row.get("elapsed_batch_seconds")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid batch time")
        if row["cached"] or row["status"] != "ok":
            if any(row[k] for k in ("charged_input_tokens", "charged_output_tokens", "batch_size", "elapsed_batch_seconds")):
                raise ValueError("Cached/rejected request improperly charged")
        elif (row["batch_size"] < 1 or row["charged_input_tokens"] != row["input_tokens"]
              or row["charged_output_tokens"] != row["output_tokens"]):
            raise ValueError("Physical generation cost mismatch")
    return predictions, config, manifest


def group_interval(pairs, family=1, replicates=BOOTSTRAP):
    """Resample complete exact-claim groups and retain native-record weighting."""
    import numpy as np
    grouped = defaultdict(list)
    for group, value in pairs:
        grouped[group].append(value)
    if not grouped:
        return {"records": 0, "groups": 0, "difference": None, "ci95": None, "ci_familywise": None}
    ordered = [grouped[g] for g in sorted(grouped)]
    totals = np.array([sum(v) for v in ordered])
    counts = np.array([len(v) for v in ordered])
    rng, samples = np.random.default_rng(SEED), []
    for start in range(0, replicates, 128):
        indices = rng.integers(len(ordered), size=(min(128, replicates - start), len(ordered)))
        samples.extend(totals[indices].sum(axis=1) / counts[indices].sum(axis=1))
    return {"records": int(counts.sum()), "groups": len(ordered), "difference": float(totals.sum() / counts.sum()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "ci_familywise": np.quantile(samples, [.025 / family, 1 - .025 / family]).tolist(),
            "family_size": family, "replicates": replicates, "seed": SEED,
            "weighting": "native-record weighted; exact original claim group resampling"}


def fraction(a, b):
    return a / b if b else None


def paired_readout(rows_a, rows_b, treatment, control, family=4, eligible_ids=None, replicates=BOOTSTRAP):
    indexes = [{(r["claim_id"], r["condition"], r["method"]): r for r in rows} for rows in (rows_a, rows_b)]
    records = []
    ids = sorted({key[0] for key in indexes[0] if key[1:] in (treatment, control)})
    for claim_id in ids:
        if eligible_ids is not None and claim_id not in eligible_ids:
            continue
        keys = [(claim_id, *arm) for arm in (control, treatment)]
        if any(key not in index for index in indexes for key in keys):
            raise ValueError("Incomplete paired native-record/condition population")
        a0, a1 = [indexes[0][key] for key in keys]
        b0, b1 = [indexes[1][key] for key in keys]
        four = (a0, a1, b0, b1)
        if len({(r["label"], r["exact_claim_group"]) for r in four}) != 1:
            raise ValueError("Intervention or repeat changed native reference")
        item = {"claim_id": claim_id, "exact_claim_group": a0["exact_claim_group"],
                "average_signed_correctness_change": ((a1["correct"] - a0["correct"]) + (b1["correct"] - b0["correct"])) / 2,
                "realization_changes": [a1["correct"] - a0["correct"], b1["correct"] - b0["correct"]],
                "complete_four_valid": all(r["prediction"] is not None for r in four)}
        for name, pairs in (("intervention", ((a0, a1), (b0, b1))), ("repeat", ((a0, b0), (a1, b1)))):
            valid = [(a, b) for a, b in pairs if a["prediction"] is not None and b["prediction"] is not None]
            item[name] = {"possible_pairs": 2, "valid_pairs": len(valid),
                          "correctness_crossings": sum(a["correct"] != b["correct"] for a, b in valid),
                          "verdict_changes": sum(a["prediction"] != b["prediction"] for a, b in valid)}
        records.append(item)
    complete = [r for r in records if r["complete_four_valid"]]
    summary = {"treatment": treatment, "control": control, "paired_records": len(records),
               "complete_four_valid_records": len(complete), "complete_panel_coverage": fraction(len(complete), len(records)),
               "signed_accuracy_change": group_interval([(r["exact_claim_group"], r["average_signed_correctness_change"]) for r in records], family, replicates),
               "by_realization": {name: {"repairs": sum(r["realization_changes"][i] == 1 for r in records),
                                          "breakages": sum(r["realization_changes"][i] == -1 for r in records),
                                          "signed_accuracy_change": fraction(sum(r["realization_changes"][i] for r in records), len(records))}
                                    for i, name in enumerate(("A", "B"))}}
    for name in ("intervention", "repeat"):
        totals = {key: sum(r[name][key] for r in records) for key in ("possible_pairs", "valid_pairs", "correctness_crossings", "verdict_changes")}
        for event in ("correctness_crossings", "verdict_changes"):
            totals[event + "_rate_valid"] = fraction(totals[event], totals["valid_pairs"])
            totals[event + "_rate_all_possible"] = fraction(totals[event], totals["possible_pairs"])
        summary[name] = totals
    for event in ("correctness_crossings", "verdict_changes"):
        summary[event + "_excess_over_repeat"] = group_interval(
            [(r["exact_claim_group"], (r["intervention"][event] - r["repeat"][event]) / 2) for r in complete],
            family if event == "correctness_crossings" else 1, replicates)
    return summary, records


def cell_metrics(rows):
    support = Counter(r["label"] for r in rows)
    return {**ev.metrics(rows), "per_class_support": {label: support[label] for label in pr.LABELS}}


def physical_cost(predictions):
    values = list(predictions.values())
    return {"logical_requests": len(values), "cached_requests": sum(r["cached"] for r in values),
            "physical_completions": sum(not r["cached"] and r["status"] == "ok" for r in values),
            "charged_input_tokens": sum(r["charged_input_tokens"] for r in values),
            "charged_output_tokens": sum(r["charged_output_tokens"] for r in values),
            "generation_batch_seconds": sum(r["elapsed_batch_seconds"] / r["batch_size"] for r in values if r["batch_size"])}


def summarize(references, predictions_a, predictions_b, padding, replicates=BOOTSTRAP):
    rows_a, rows_b = [ev.outcomes(references, p) for p in (predictions_a, predictions_b)]
    cells = {}
    arm_keys = sorted({(r["condition"], r["method"]) for r in references})
    for condition, method in arm_keys:
        members = [[r for r in rows if (r["condition"], r["method"]) == (condition, method)] for rows in (rows_a, rows_b)]
        a, b = [cell_metrics(rows) for rows in members]
        cells[f"{condition}/{method}"] = {"A": a, "B": b,
                                         "mean_realization_accuracy": (a["accuracy"] + b["accuracy"]) / 2,
                                         "mean_realization_macro_f1": (a["macro_f1"] + b["macro_f1"]) / 2,
                                         "standalone_logical_calls_two_realizations": len(members[0]) + len(members[1])}
    effects, pair_rows = {}, []
    for treatment, control in RAW_PAIRS:
        key = treatment + "_minus_" + control
        effects[key], details = paired_readout(rows_a, rows_b, (treatment, "raw_packets"),
                                               (control, "raw_packets"), replicates=replicates)
        pair_rows.extend({**r, "contrast": key} for r in details)
    length_controls = {}
    for treatment, control in PAD_PAIRS:
        selected = [r for r in padding if r["id"].split("/")[1] == control]
        matched = {r["claim_id"] for r in selected if r["exact_length_match"]}
        key = treatment + "_minus_" + control
        all_result, _ = paired_readout(rows_a, rows_b, (treatment, "raw_packets"), (control, "raw_packets"), family=1, replicates=replicates)
        matched_result, details = paired_readout(rows_a, rows_b, (treatment, "raw_packets"), (control, "raw_packets"), eligible_ids=matched, replicates=replicates)
        matched_result["crossing_interpretation"] = "descriptive; the length-control confirmatory family concerns signed accuracy only"
        padding_effect, _ = paired_readout(rows_a, rows_b, (control, "raw_packets"),
                                           (control.removesuffix("_length_control"), "raw_packets"), family=1, replicates=replicates)
        length_controls[key] = {"eligible_pairs": len(selected), "exact_length_pairs": len(matched),
                                "mismatch_pairs": len(selected) - len(matched), "exact_length": matched_result,
                                "all_pairs_descriptive": all_result, "padding_effect_descriptive": padding_effect}
        pair_rows.extend({**r, "contrast": key + "/exact_length"} for r in details)
    secondary = {}
    for condition in ("clean", "duplicate4", "partial1", "partial4"):
        for treatment, control in (("exact_packet_dedup", "raw_packets"), ("exact_fragment_dedup", "raw_packets"),
                                   ("ledger_counts", "exact_fragment_dedup")):
            result, _ = paired_readout(rows_a, rows_b, (condition, treatment), (condition, control), family=1, replicates=replicates)
            secondary[f"{condition}/{treatment}_minus_{control}"] = result
    return {"cells": cells, "raw_primary": effects, "length_controls": length_controls,
            "secondary_descriptive": secondary, "physical_cost": {"A": physical_cost(predictions_a), "B_fresh_repeat": physical_cost(predictions_b)}}, rows_a, rows_b, pair_rows


def evaluate(args):
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    manifest, artifacts = load_prepared(args.prepared)
    requests, references, padding, _, lengths = artifacts
    if args.realization_a.resolve() == args.realization_b.resolve():
        raise ValueError("Two distinct fresh realization files required")
    runs = [checked_run(p, args.prepared / "requests.jsonl", requests, lengths, manifest["model"])
            for p in (args.realization_a, args.realization_b)]
    (a, config_a, ma), (b, config_b, mb) = runs
    if (config_a != config_b or (ma.get("process_id"), ma.get("start_utc")) == (mb.get("process_id"), mb.get("start_utc"))
            or not all(m.get("process_id") and m.get("start_utc") for m in (ma, mb))):
        raise ValueError("Repeat must use an identical configuration in a fresh process")
    if any(a[key]["batch_size"] != b[key]["batch_size"] or a[key]["cached"] != b[key]["cached"] for key in a):
        raise ValueError("Whole-file replay changed effective batch/cache layout")
    report, rows_a, rows_b, paired = summarize(references, a, b, padding)
    report.update(version=VERSION, model_key=manifest["model"]["key"], native_records=manifest["native_records"],
                  exact_claim_groups=manifest["exact_claim_groups"], evidence_type="native_four_way_provided_evidence_measurement",
                  preparation_sha256=file_sha(args.prepared / "preparation.json"),
                  source=manifest["source"], source_binding=manifest["source_binding"],
                  generation_manifests={"A": ma, "B": mb},
                  prediction_files=[{"path": str(p.resolve()), "sha256": file_sha(p),
                                     "manifest_sha256": file_sha(p.with_suffix(p.suffix + ".manifest.json"))}
                                    for p in (args.realization_a, args.realization_b)],
                  families={"signed_accuracy": 4, "crossing_excess": 4, "length_matched_signed_accuracy": 4},
                  new_human_annotations=0, natural_lineage_labels=False, retrieval_executed=False,
                  interpretation=["All emitted invalid/long outputs count as incorrect.",
                                  "Crossing inference conditions on four valid outputs; report coverage.",
                                  "Padding is an operational control and is not assumed behaviorally inert.",
                                  "Identical-message equality within a realization is a global-cache contract.",
                                  "Native records sharing exact claim text remain distinct but resample together."])
    args.output.mkdir(parents=True)
    for name, rows in (("outcomes_A.jsonl", rows_a), ("outcomes_B.jsonl", rows_b), ("paired_records.jsonl", paired)):
        write_rows(args.output / name, rows)
    report["outcome_files"] = {p.name: file_sha(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "summary.json", report)
    print(json.dumps({"output": str(args.output), "records": manifest["native_records"], "requests_per_realization": len(requests)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("prepare")
    one.add_argument("--source", type=Path, default=HERE / "data" / "averitec_dev.json")
    one.add_argument("--model-key", choices=MODELS, required=True)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("evaluate")
    two.add_argument("--prepared", type=Path, required=True)
    two.add_argument("--realization-a", type=Path, required=True)
    two.add_argument("--realization-b", type=Path, required=True)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
