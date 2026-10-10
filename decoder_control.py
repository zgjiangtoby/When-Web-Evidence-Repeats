#!/usr/bin/env python3
"""Prospective decoder factorial on copied, frozen supplement requests; CPU only."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import allocation as sp
from generate import load_predictions, read_requests, sha
import generate_closed_labels as closed

fm, HERE, ROOT = sp.fm, sp.HERE, sp.ROOT
VERSION = "decoder_control_20261002_v1"
SEED, BOOTSTRAP = 20261002, 10000
DECODERS, REALIZATIONS = ("native", "closed"), ("A", "B")
POPULATION = {"averitec": (500, 491), "scifact": (209, 164)}
FROZEN_PYTHON = str(Path(sys.executable).resolve())


def bindings():
    result = sp.bindings()
    for name in ("generate_closed_labels.py", "decoder_control.py"):
        result[name] = fm.file_sha(ROOT / name)
    return result


def add_metadata(requests, references):
    refs = {row["id"]: row for row in references}
    if len(refs) != len(references) or set(refs) != {row["id"] for row in requests}:
        raise ValueError("Request/reference identity mismatch")
    result = []
    for request in requests:
        if request.get("metadata") != {}:
            raise ValueError("Frozen supplement requests must have empty metadata")
        ref = refs[request["id"]]
        corpus, condition = ref["corpus"], ref["condition"]
        if corpus not in sp.ARMS or condition not in sp.ARMS[corpus]:
            raise ValueError("Unknown frozen corpus or presentation condition")
        result.append({**deepcopy(request), "metadata": {"corpus": corpus, "condition": condition,
                                                         "allowed_labels": closed.ALPHABETS[corpus]}})
    closed.validate_requests(result)
    return result


def load_original(model_key):
    directory = HERE / "prepared" / ("supplement_" + model_key + "_v1")
    manifest = json.loads((directory / "preparation.json").read_text())
    if (manifest["version"] != sp.VERSION or manifest["status"] != "PREPARED_NOT_RUN"
            or manifest["binding"] != sp.bindings() or manifest["model"]["key"] != model_key):
        raise ValueError("Frozen supplement preparation or source binding changed")
    for name, digest in manifest["files"].items():
        if fm.file_sha(directory / name) != digest:
            raise ValueError("Frozen supplement artifact changed: " + name)
    observations, sources = sp.native_observations()
    if (observations != fm.ev.read_jsonl(directory / "observations.jsonl") or sources != manifest["sources"]
            or sp.model_binding(model_key) != manifest["model"]):
        raise ValueError("Native source, frozen target, label, or model changed")
    requests = read_requests(directory / "requests.jsonl")
    references = fm.ev.read_jsonl(directory / "references.jsonl")
    padding = fm.ev.read_jsonl(directory / "padding_audit.jsonl")
    lengths = fm.ev.read_jsonl(directory / "length_audit.jsonl")
    for corpus, (records, groups) in POPULATION.items():
        subset = [r for r in references if r["corpus"] == corpus]
        by_id = defaultdict(list)
        for row in subset:
            by_id[row["claim_id"]].append(row["condition"])
        if (len(by_id) != records or len({r["resampling_group"] for r in subset}) != groups
                or any(Counter(arms) != Counter(sp.ARMS[corpus]) for arms in by_id.values())):
            raise ValueError("Incomplete original native population, arms, or exact-text components")
    if len(requests) != 4672 or manifest["logical_requests"] != 4672:
        raise ValueError("The complete 4672-request population is required")
    return directory, manifest, requests, references, padding, lengths


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a new decoder-control preparation directory")
    original, old, requests, references, _, _ = load_original(args.model_key)
    augmented = add_metadata(requests, references)
    tokenizer = closed.load_local_tokenizer(Path(old["model"]["path"]))
    contract = closed.decision_contract(tokenizer, json.loads(
        (Path(old["model"]["path"]) / "generation_config.json").read_text()))
    args.output.mkdir(parents=True)
    fm.write_rows(args.output / "requests.jsonl", augmented)
    manifest = {"version": VERSION, "status": "PREPARED_NOT_RUN", "binding": bindings(), "model": old["model"],
                "original_prepared": str(original.resolve()), "original_preparation_sha256": fm.file_sha(original / "preparation.json"),
                "requests_sha256": fm.file_sha(args.output / "requests.jsonl"), "logical_requests": len(augmented),
                "unique_messages": len({sha(r["messages"]) for r in augmented}), "population": POPULATION,
                "decision_contract": contract, "metadata_fields": ["corpus", "condition", "allowed_labels"],
                "messages_equal_in_original_order": all(a["messages"] == b["messages"] for a, b in zip(requests, augmented)),
                "new_human_annotations": 0, "prospective_after_prior_results_seen": True}
    fm.write_json(args.output / "preparation.json", manifest)
    print(json.dumps({"status": manifest["status"], "requests": len(augmented), "decision_contract": contract}))


def load_prepared(directory):
    manifest = json.loads((directory / "preparation.json").read_text())
    if manifest["version"] != VERSION or manifest["status"] != "PREPARED_NOT_RUN" or manifest["binding"] != bindings():
        raise ValueError("Nonrunnable preparation or changed extension source binding")
    original, old, requests, refs, padding, lengths = load_original(manifest["model"]["key"])
    augmented = read_requests(directory / "requests.jsonl")
    if (str(original.resolve()) != manifest["original_prepared"] or old["model"] != manifest["model"]
            or fm.file_sha(original / "preparation.json") != manifest["original_preparation_sha256"]
            or fm.file_sha(directory / "requests.jsonl") != manifest["requests_sha256"]
            or augmented != add_metadata(requests, refs)):
        raise ValueError("Extension changed original request messages, IDs, metadata, or source references")
    tokenizer = closed.load_local_tokenizer(Path(old["model"]["path"]))
    contract = closed.decision_contract(tokenizer, json.loads(
        (Path(old["model"]["path"]) / "generation_config.json").read_text()))
    if contract != manifest["decision_contract"]:
        raise ValueError("Tokenizer or inherited generation contract changed")
    return manifest, augmented, refs, padding, lengths


def checked_closed_run(path, request_file, requests, lengths, preparation):
    predictions, config, manifests = load_predictions([path], request_file)
    manifest, model = manifests[0], preparation["model"]
    required = {"code_sha256": fm.file_sha(ROOT / "generate_closed_labels.py"),
                "native_runner_sha256": fm.RUNNER_SHA, "model_path": model["path"], "model_revision": model["revision"],
                "model_config_sha256": model["files"]["config.json"], "versions": model["versions"],
                "batch_size": 8, "max_input_tokens": 8192, "max_new_tokens": 16, "num_shards": 1, "seed": 17,
                "do_sample": False, "thinking": False, "tokenizer_use_fast": False, "dtype": "bfloat16", "attention": "sdpa",
                "decision_contract": preparation["decision_contract"]}
    if (any(config.get(k) != v for k, v in required.items()) or manifest["config"].get("shard_index") != 0
            or manifest.get("total_requests") != len(requests) or manifest.get("completed") != len(requests)
            or set(predictions) != {r["id"] for r in requests}):
        raise ValueError("Incomplete or nonfrozen closed-label run")
    expected = {r["id"]: r["input_tokens"] for r in lengths}
    contract, seen = preparation["decision_contract"], {}
    for request in requests:
        row, key = predictions[request["id"]], sha(request["messages"])
        if (row.get("status") not in ("ok", "input_too_long") or row.get("input_tokens") != expected[request["id"]]
                or (row["status"] == "input_too_long") != (row["input_tokens"] > 8192)
                or type(row.get("cached")) is not bool or row["cached"] != (key in seen)):
            raise ValueError("Closed output status, length, or global exact-message cache differs")
        if key in seen:
            origin = predictions[seen[key]]
            if row.get("cache_of") != seen[key] or any(row.get(k) != origin.get(k) for k in
                    ("status", "response", "input_tokens", "output_tokens", "generated_token_ids", "first_step")):
                raise ValueError("Cached completion differs from its same-run original")
        else:
            seen[key] = row["id"]
        for field in ("input_tokens", "output_tokens", "charged_input_tokens", "charged_output_tokens", "batch_size"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError("Invalid recorded closed-label cost")
        elapsed = row.get("elapsed_batch_seconds")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid batch time")
        if row["cached"] or row["status"] != "ok":
            if any(row[k] for k in ("charged_input_tokens", "charged_output_tokens", "batch_size", "elapsed_batch_seconds")):
                raise ValueError("Cached/rejected request improperly charged")
        elif not 1 <= row["batch_size"] <= 8 or any(row["charged_" + k] != row[k] for k in ("input_tokens", "output_tokens")):
            raise ValueError("Physical generation cost mismatch")
        if row["status"] == "ok":
            response = row["response"]
            if (response not in list(request["metadata"]["allowed_labels"]) or row["output_tokens"] != 2
                    or row.get("generated_token_ids") != [contract["label_token_ids"][response], contract["forced_eos_token_id"]]):
                raise ValueError("Closed completion violates letter/EOS contract")
            first = row["first_step"]
            scores = first["candidates"]
            if set(scores) != set(request["metadata"]["allowed_labels"]) or first["additional_forward_passes"] != 0:
                raise ValueError("Missing complete first-forward diagnostic")
            for label, values in scores.items():
                if (values["token_id"] != contract["label_token_ids"][label]
                        or type(values["token_occurs_in_prompt"]) is not bool
                        or any(not math.isfinite(values[k]) for k in ("raw_logit", "processed_score"))):
                    raise ValueError("Invalid paired raw/processed legal-label scores")
            choose = lambda field: min(scores, key=lambda label: (-scores[label][field], scores[label]["token_id"]))
            if (first["processed_argmax"] != response or choose("processed_score") != response
                    or first["raw_logit_shadow_argmax"] != choose("raw_logit")
                    or first["shadow_differs"] != (response != choose("raw_logit"))):
                raise ValueError("Inconsistent official/shadow first-forward decision")
        elif row["response"] is not None or row["output_tokens"] != 0 or row.get("generated_token_ids") != []:
            raise ValueError("Rejected input acquired a generated output")
    return predictions, config, manifest


def checked_model(prepared, paths):
    manifest, requests, refs, padding, lengths = load_prepared(prepared)
    runs = {}
    for decoder in DECODERS:
        for realization in REALIZATIONS:
            path = paths[(decoder, realization)]
            if decoder == "native":
                run = fm.checked_run(path, prepared / "requests.jsonl", requests, lengths, manifest["model"])
            else:
                run = checked_closed_run(path, prepared / "requests.jsonl", requests, lengths, manifest)
            if any(r["output_tokens"] > 16 for r in run[0].values()):
                raise ValueError("Output-token ceiling exceeded")
            runs[(decoder, realization)] = run
        a, b = [runs[(decoder, r)] for r in REALIZATIONS]
        ma, mb = a[2], b[2]
        if (a[1] != b[1] or not all(m.get("process_id") and m.get("start_utc") and m.get("end_utc") for m in (ma, mb))
                or (ma["process_id"], ma["start_utc"]) == (mb["process_id"], mb["start_utc"])
                or sp.utc(ma["end_utc"]) > sp.utc(mb["start_utc"])):
            raise ValueError("Each decoder needs identical settings and fresh complete chronological A then B")
    baseline = runs[("native", "A")][0]
    for predictions, _, _ in runs.values():
        if any(any(predictions[k][field] != baseline[k][field] for field in
                   ("input_tokens", "charged_input_tokens", "cached", "batch_size", "prompt_sha256")) for k in baseline):
            raise ValueError("Decoder/repeat changed input or effective batch/cache layout")
    return manifest, refs, padding, runs


def group_interval(pairs, family=1, replicates=BOOTSTRAP):
    """Same record-weighted component bootstrap as frozen measurement.py, with this plan's seed."""
    import numpy as np
    groups = defaultdict(list)
    for group, value in pairs:
        groups[group].append(value)
    if not groups:
        return {"records": 0, "groups": 0, "difference": None, "ci95": None, "ci_familywise": None}
    ordered = [groups[g] for g in sorted(groups)]
    totals, counts = np.array([sum(v) for v in ordered]), np.array([len(v) for v in ordered])
    rng, samples = np.random.default_rng(SEED), []
    for start in range(0, replicates, 128):
        indices = rng.integers(len(ordered), size=(min(128, replicates - start), len(ordered)))
        samples.extend(totals[indices].sum(axis=1) / counts[indices].sum(axis=1))
    return {"records": int(counts.sum()), "groups": len(ordered), "difference": float(totals.sum() / counts.sum()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "ci_familywise": np.quantile(samples, [.025 / family, 1 - .025 / family]).tolist(),
            "family_size": family, "replicates": replicates, "seed": SEED,
            "weighting": "native-record weighted; exact claim (AV) or exact claim/full-abstract components (SciFact)"}


def contrast_records(indexes, corpus, treatment, control):
    result = []
    ids = sorted({identity for identity, _ in indexes[("native", "A")][corpus]})
    for identity in ids:
        item = {"observation_id": identity}
        all_valid, rob_valid = True, True
        for decoder in DECODERS:
            values, valid, rob = [], True, True
            for realization in REALIZATIONS:
                index = indexes[(decoder, realization)][corpus]
                for side in ("front", "back"):
                    t = index[(identity, treatment + "-" + side if treatment in ("R", "S") else treatment)]
                    c = index[(identity, control + "-" + side if control == "O" else control)]
                    values.append(t["correct"] - c["correct"])
                    valid &= t["prediction"] is not None and c["prediction"] is not None
                    item["group"] = t["resampling_group"]
                if corpus == "scifact":
                    rob &= all(index[(identity, arm)]["prediction"] is not None
                               for arm in ("B", "R-front", "R-back", "O-front", "O-back"))
                else:
                    rob &= valid
            item[decoder] = {"difference": sum(values) / 4, "values_Afront_Aback_Bfront_Bback": values,
                             "valid": bool(valid), "ROB_valid": bool(rob)}
            all_valid &= valid
            rob_valid &= rob
        item.update(interaction=item["closed"]["difference"] - item["native"]["difference"],
                    common_valid=bool(all_valid), common_ROB_valid=bool(rob_valid))
        result.append(item)
    return result


def summarize_contrast(records, primary, replicates):
    family = 4 if primary else 1
    interaction = group_interval([(r["group"], r["interaction"]) for r in records], family, replicates)
    panels = {}
    for field in ("common_valid", "common_ROB_valid"):
        complete = [r for r in records if r[field]]
        panels[field] = {"records": len(complete), "coverage": len(complete) / len(records),
                         "coverage_pass_95pct": len(complete) / len(records) >= .95,
                         "interaction_descriptive": group_interval([(r["group"], r["interaction"]) for r in complete], 1, replicates)}
    decoder_reports = {}
    for decoder in DECODERS:
        signed = group_interval([(r["group"], r[decoder]["difference"]) for r in records], family, replicates)
        positions = {side: sum((r[decoder]["values_Afront_Aback_Bfront_Bback"][i]
                                + r[decoder]["values_Afront_Aback_Bfront_Bback"][i + 2]) / 2 for r in records) / len(records)
                     for i, side in enumerate(("front", "back"))}
        valid = [r for r in records if r[decoder]["valid"]]
        rob = [r for r in records if r[decoder]["ROB_valid"]]
        mean_valid = fm.fraction(sum(r[decoder]["difference"] for r in valid), len(valid))
        mean_rob = fm.fraction(sum(r[decoder]["difference"] for r in rob), len(rob))
        screen = (signed["difference"] >= .03 and signed["ci_familywise"][0] > 0 and min(positions.values()) > 0
                  and len(valid) / len(records) >= .95 and mean_valid is not None and mean_valid > 0
                  and len(rob) / len(records) >= .95 and mean_rob is not None and mean_rob > 0)
        decoder_reports[decoder] = {"signed_descriptive": signed, "positions": positions,
                                    "valid_records": len(valid), "valid_coverage": len(valid) / len(records),
                                    "valid_difference": mean_valid, "ROB_valid_records": len(rob),
                                    "ROB_valid_coverage": len(rob) / len(records), "ROB_valid_difference": mean_rob,
                                    "legacy_joint_screen_component_descriptive": screen if primary else None}
    ci = interaction["ci_familywise"]
    return {"interaction_closed_minus_native": interaction,
            "interaction_detected_two_sided": ci[0] > 0 or ci[1] < 0,
            "common_panels": panels, "decoder_contrasts": decoder_reports,
            "interpretation": "primary two-sided decoder interaction" if primary else "descriptive transfer only"}


def transitions(native, constrained):
    n = len(native)
    repairs = sum(b["correct"] > a["correct"] for a, b in zip(native, constrained))
    breakages = sum(b["correct"] < a["correct"] for a, b in zip(native, constrained))
    return {"records": n, "repairs": repairs, "breakages": breakages, "net_correctness": repairs - breakages,
            "accuracy_delta": (repairs - breakages) / n,
            "format_failure_delta": sum((b["output_status"] == "invalid_format") - (a["output_status"] == "invalid_format")
                                        for a, b in zip(native, constrained)) / n,
            "both_valid_records": sum(a["prediction"] is not None and b["prediction"] is not None for a, b in zip(native, constrained)),
            "changed_valid_labels": sum(a["prediction"] is not None and b["prediction"] is not None
                                        and a["prediction"] != b["prediction"] for a, b in zip(native, constrained)),
            "invalid_to_valid": sum(a["prediction"] is None and b["prediction"] is not None for a, b in zip(native, constrained)),
            "valid_to_invalid": sum(a["prediction"] is not None and b["prediction"] is None for a, b in zip(native, constrained))}


def summarize(refs, padding, runs, replicates=BOOTSTRAP):
    outcomes = {key: sp.construction_status(sp.native_outcomes(refs, run[0]), padding) for key, run in runs.items()}
    indexes = {key: {corpus: {(r["claim_id"], r["condition"]): r for r in rows if r["corpus"] == corpus}
                     for corpus in sp.ARMS} for key, rows in outcomes.items()}
    primary, details = {}, []
    for treatment in ("R", "B"):
        records = contrast_records(indexes, "scifact", treatment, "O")
        primary[treatment + "_minus_O"] = summarize_contrast(records, True, replicates)
        details.extend({**r, "contrast": treatment + "_minus_O", "corpus": "scifact"} for r in records)
    av = contrast_records(indexes, "averitec", "S", "B")
    transfer = summarize_contrast(av, False, replicates)
    details.extend({**r, "contrast": "S_minus_B", "corpus": "averitec"} for r in av)
    cells, ordering, shadow = {}, {}, {}
    for corpus, arms in sp.ARMS.items():
        for arm in arms:
            members = {key: [r for r in rows if r["corpus"] == corpus and r["condition"] == arm]
                       for key, rows in outcomes.items()}
            cell = {decoder: {realization: sp.cell_metrics(members[(decoder, realization)])
                               for realization in REALIZATIONS} for decoder in DECODERS}
            cell["decoder_changes"] = {r: transitions(members[("native", r)], members[("closed", r)]) for r in REALIZATIONS}
            cell["fresh_repeat_changes"] = {d: transitions(members[(d, "A")], members[(d, "B")]) for d in DECODERS}
            cells[corpus + "/" + arm] = cell
            for realization in REALIZATIONS:
                selected = [runs[("closed", realization)][0][r["id"]] for r in members[("closed", realization)]]
                valid = [r for r in selected if r["status"] == "ok"]
                official = members[("closed", realization)]
                labels = sp.pr.LABELS if corpus == "averitec" else sp.SCI_LABELS
                shadow_correct = [int(raw["status"] == "ok" and scored["output_status"] != "construction_length_mismatch"
                                      and labels["ABCD".index(raw["first_step"]["raw_logit_shadow_argmax"])] == scored["label"])
                                  for raw, scored in zip(selected, official)]
                repairs = sum(s > r["correct"] for s, r in zip(shadow_correct, official))
                breakages = sum(s < r["correct"] for s, r in zip(shadow_correct, official))
                shadow[f"{corpus}/{arm}/{realization}"] = {"logical_generated": len(valid),
                    "logical_shadow_disagreements": sum(r["first_step"]["shadow_differs"] for r in valid),
                    "physical_generated": sum(not r["cached"] for r in valid),
                    "physical_shadow_disagreements": sum(not r["cached"] and r["first_step"]["shadow_differs"] for r in valid),
                    "records_including_failures": len(official), "official_correct": sum(r["correct"] for r in official),
                    "raw_logit_shadow_correct": sum(shadow_correct), "shadow_repairs_vs_official": repairs,
                    "shadow_breakages_vs_official": breakages, "shadow_net_correctness_vs_official": repairs - breakages,
                    "shadow_accuracy_delta_vs_official": (repairs - breakages) / len(official),
                    "additional_forward_passes": 0, "used_for_official_predictions": False}
        ordering[corpus] = {}
        for decoder in DECODERS:
            accuracy = {arm: sum(cells[corpus + "/" + arm][decoder][r]["accuracy"] for r in REALIZATIONS) / 2 for arm in arms}
            for role in (("S",) if corpus == "averitec" else ("R", "O")):
                accuracy[role + "-fixed_average"] = (accuracy[role + "-front"] + accuracy[role + "-back"]) / 2
            ordering[corpus][decoder] = sorted(accuracy.items(), key=lambda item: (-item[1], item[0]))
    cost = {decoder: {r: fm.physical_cost(runs[(decoder, r)][0]) for r in REALIZATIONS} for decoder in DECODERS}
    return {"primary_interactions": primary, "averitec_S_minus_B_descriptive": transfer,
            "cells": cells, "fresh_policy_ordering": ordering, "physical_cost": cost,
            "first_forward_shadow_diagnostic": shadow, "input_batch_cache_identity_verified": True}, outcomes, details


def expected_argv(prepared, output, model, decoder):
    result = sp.expected_argv(prepared, output, model)
    if decoder == "closed":
        result[0] = str(ROOT / "generate_closed_labels.py")
    return result


def check_receipt(receipt, bundles, paths, prepared):
    entries = receipt.get("runs", [])
    required = {(m, d, r) for m in fm.MODELS for d in DECODERS for r in REALIZATIONS}
    keys = [(e["model_key"], e["decoder"], e["realization"]) for e in entries]
    if len(keys) != 8 or set(keys) != required:
        raise ValueError("Receipt requires eight distinct fixed model/decoder/realization cells")
    occupied, times = 0, {}
    for entry, key in zip(entries, keys):
        model, decoder, realization = key
        output = paths[model][(decoder, realization)]
        manifest = bundles[model][3][(decoder, realization)][2]
        begin, end = sp.utc(entry["launch_utc"]), sp.utc(entry["finish_utc"])
        if (entry.get("visible_gpu_count") != 1 or entry.get("output_existed_before_launch") is not False
                or entry.get("resume") is not False or entry.get("returncode") != 0
                or Path(entry["output_path"]).resolve() != output.resolve()
                or entry.get("output_sha256") != fm.file_sha(output)
                or entry.get("manifest_sha256") != fm.file_sha(output.with_suffix(output.suffix + ".manifest.json"))
                or entry.get("argv") != [FROZEN_PYTHON] + expected_argv(prepared[model], output, bundles[model][0]["model"], decoder)
                or end < begin or begin > sp.utc(manifest["start_utc"]) or end < sp.utc(manifest["end_utc"])):
            raise ValueError("Missing or inconsistent fresh execution, frozen argv, output, or timing receipt")
        occupied += (end - begin).total_seconds()
        times[key] = (begin, end)
    if any(times[(m, d, "A")][1] > times[(m, d, "B")][0] for m in fm.MODELS for d in DECODERS):
        raise ValueError("A must finish before B starts for each model and decoder")
    attempts = receipt.get("additional_attempts")
    if not isinstance(attempts, list):
        raise ValueError("Record additional_attempts explicitly, including an empty list when none")
    for attempt in attempts:
        if any(field not in attempt for field in ("argv", "launch_utc", "finish_utc", "returncode", "log_path", "reason")):
            raise ValueError("Failed/additional attempts need full execution records")
        begin, end = sp.utc(attempt["launch_utc"]), sp.utc(attempt["finish_utc"])
        if end < begin:
            raise ValueError("Additional attempt finish precedes launch")
        occupied += (end - begin).total_seconds()
    return occupied


def evaluate(args):
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    prepared = {model: getattr(args, model + "_prepared") for model in fm.MODELS}
    paths = {model: {(d, r): getattr(args, f"{model}_{d}_{r.lower()}") for d in DECODERS for r in REALIZATIONS}
             for model in fm.MODELS}
    if len({p.resolve() for group in paths.values() for p in group.values()}) != 8:
        raise ValueError("Eight distinct fresh output paths required")
    # No native scoring or model comparison until all eight complete runs pass the gate.
    bundles = {m: checked_model(prepared[m], paths[m]) for m in fm.MODELS}
    if any(bundle[0]["model"]["key"] != model for model, bundle in bundles.items()):
        raise ValueError("Prepared model keys are swapped")
    receipt = json.loads(args.execution_receipt.read_text())
    occupied = check_receipt(receipt, bundles, paths, prepared)
    historical = HERE / "results/supplement_v1_evaluation/summary.json"
    report = {"version": VERSION, "status": "COMPLETED_DECODER_SYSTEM_MEASUREMENT", "models": {},
              "binding": bindings(), "primary_family_size": 4, "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAP,
              "execution_receipt_sha256": fm.file_sha(args.execution_receipt), "occupied_gpu_seconds": occupied,
              "additional_attempts": receipt["additional_attempts"], "prospective_after_prior_results_seen": True,
              "historical_original_results": {"path": str(historical.resolve()), "sha256": fm.file_sha(historical),
                                               "role": "unchanged historical two-model result, not the fresh native comparator"},
              "interpretation": ["New decoder/system setting; no post-hoc parsing, semantic-recovery, or pure-mechanism claim.",
                                 "Primary interactions are two-sided with no preferred direction.",
                                 "All native records remain; invalid and overlength outputs score zero.",
                                 "The inherited four-component positive screen is a separate descriptive migration diagnostic.",
                                 "Common-valid semantic panel claims require at least 95% coverage.",
                                 "Shadow decisions reuse first-forward raw logits and never replace official outputs."]}
    tables = {}
    for model, (manifest, refs, padding, runs) in bundles.items():
        summary, outcomes, details = summarize(refs, padding, runs)
        summary["generation_manifests"] = {f"{d}_{r}": runs[(d, r)][2] for d in DECODERS for r in REALIZATIONS}
        summary["preparation_sha256"] = fm.file_sha(prepared[model] / "preparation.json")
        report["models"][model] = summary
        tables[model + "_contrast_records.jsonl"] = details
        for (decoder, realization), rows in outcomes.items():
            tables[f"{model}_{decoder}_{realization}_outcomes.jsonl"] = rows
    report["closed_legacy_joint_screen_descriptive"] = all(
        contrast["decoder_contrasts"]["closed"]["legacy_joint_screen_component_descriptive"]
        for model in report["models"].values() for contrast in model["primary_interactions"].values())
    report["joint_screen_interpretation"] = ("descriptive transfer screen passed; not joint inference with primary interaction family"
        if report["closed_legacy_joint_screen_descriptive"] else "bounded negative: closed decoding did not satisfy the inherited joint screen")
    costs = [c for m in report["models"].values() for d in m["physical_cost"].values() for c in d.values()]
    if (sum(c["logical_requests"] for c in costs) != 37376
            or sum(c["charged_input_tokens"] for c in costs) > 306184192
            or sum(c["charged_output_tokens"] for c in costs) > 598016):
        raise ValueError("Fixed logical-request/token budget exceeded")
    args.output.mkdir(parents=True)
    for name, rows in tables.items():
        fm.write_rows(args.output / name, rows)
    report["outcome_files"] = {name: fm.file_sha(args.output / name) for name in tables}
    fm.write_json(args.output / "summary.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "closed_legacy_joint_screen_descriptive": report["closed_legacy_joint_screen_descriptive"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model-key", choices=fm.MODELS, required=True)
    prep.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--prepared", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    for model in fm.MODELS:
        evaluation.add_argument("--" + model + "-prepared", type=Path, required=True)
        for decoder in DECODERS:
            for realization in REALIZATIONS:
                evaluation.add_argument(f"--{model}-{decoder}-{realization.lower()}", type=Path, required=True)
    evaluation.add_argument("--execution-receipt", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        manifest, *_ = load_prepared(args.prepared)
        print(json.dumps({"status": "VERIFIED_PREPARED_NOT_RUN", "logical_requests": manifest["logical_requests"]}))
    else:
        {"prepare": prepare, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
