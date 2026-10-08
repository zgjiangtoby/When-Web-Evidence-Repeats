#!/usr/bin/env python3
"""CPU preparation/evaluation for the frozen operational repetition supplement."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formal as fm
import prepare as pr
import regex
from generate import sha, load_local_tokenizer
from replay import extract_units

HERE, ROOT = fm.HERE, fm.ROOT
VERSION = "repetition_allocation_supplement_v1"
CANDIDATE_SHA = "9caa2d01cd4746f0b5c8738a54a2af194d403cc8e46f739162eef1fdbedd0217"
SCI_FILES = {
    "data.tar.gz": "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
    "corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
    "claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    "claims_train.jsonl": "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0",
}
FROZEN_FORMAL = {
    "paper2/formal.py": "eb88f14e7b562e055ef6aafd3aa6b9d15415f13b6424880d6e007ce799600b46",
    "paper2/test_formal.py": "6abd10defa6c9a159625a780d86c299f6ec309411f7312aafa1579c2f9a670eb",
    "paper2/FORMAL_PROTOCOL_V1.md": "de7c414c746c30cf220ce22ca5b7a5693a5ca7115a33a57c66e6fb751e17a0dd",
}
SCI_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
SCI_SYSTEM = ("Decide whether the scientific claim is supported, contradicted, or not established by the provided abstract. "
              "Repetition does not add factual information. Document and sentence numbers identify the same supplied source, "
              "not independent sources. Return exactly one letter, with no explanation: A = SUPPORT; B = CONTRADICT; C = NOINFO.")
ARMS = {"averitec": ("C", "P", "B", "W", "S-front", "S-back"),
        "scifact": ("C", "P", "B", "W", "R-front", "R-back", "O-front", "O-back")}
PATTERNS = {
    "GPT2Tokenizer": r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    "Qwen2Tokenizer": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
ARTIFACTS = ("requests.jsonl", "references.jsonl", "padding_audit.jsonl", "copy_audit.jsonl", "length_audit.jsonl")
WEIGHTS = {
    "qwen": ("a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
             "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
             "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
             "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd"),
    "granite": ("b309f7b9e116fafc27c1100629bfcccbcebdab18a4bf7df7accfa728260dc57e",
                "f2e40f157f0139e4b67bbaa4b10efa351162c062771cb543b143c6916a687266",
                "5cf5203a027b7055c995a9d4bdb3a4c9dd44849ae6c9eefa6a16852a816bea76",
                "76ac0c0dbd790a1e003d2fc098a297bfe82a422c129046e9d4e67e6ed3d820f0"),
}


def bindings():
    result = fm.bindings()
    for name, expected in FROZEN_FORMAL.items():
        if result[name] != expected:
            raise ValueError("Immutable formal dependency changed: " + name)
    for name in ("supplement.py", "test_supplement.py", "SUPPLEMENT_PROTOCOL_V1.md", "SUPPLEMENT_DESIGN_CANDIDATE_20260930.md"):
        result["paper2/" + name] = fm.file_sha(HERE / name)
    if result["paper2/SUPPLEMENT_DESIGN_CANDIDATE_20260930.md"] != CANDIDATE_SHA:
        raise ValueError("Adopted candidate changed")
    return result


def model_binding(key):
    result = fm.model_binding(key)
    directory = Path(result["path"])
    index = directory / "model.safetensors.index.json"
    names = [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    if set(json.loads(index.read_text())["weight_map"].values()) != set(names):
        raise ValueError("Frozen checkpoint index/shard set changed")
    weights = {}
    for name, expected in zip(names, WEIGHTS[key]):
        with (directory / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Checkpoint bytes differ from official frozen LFS hash: " + name)
        weights[name] = {"sha256": actual, "bytes": (directory / name).stat().st_size}
    result.update(weights=weights, weight_index_sha256=fm.file_sha(index))
    return result


def canonical_av(record):
    original = extract_units(record)
    if not original:
        raise ValueError("No native QA units")
    classes = {}
    for unit in original:
        entry = classes.setdefault(unit["text"], {"text": unit["text"], "ids": [], "urls": set()})
        entry["ids"].append(unit["unit_id"])
        entry["urls"].update(unit["source_urls"])
    units = []
    for entry in classes.values():
        entry["urls"] = sorted(entry["urls"])
        entry["rendered"] = "Evidence unit " + entry["ids"][0] + ":\n" + entry["text"]
        if entry["urls"]:
            entry["rendered"] += "\nSources: " + ", ".join(entry["urls"])
        units.append(entry)
    return original, units


def component_groups(observations):
    """Only exact claim/full-abstract equality; no inferred relationship inputs."""
    parent = list(range(len(observations)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, obs in enumerate(observations):
        keys = [("claim", obs["exact_claim_group"])]
        if obs["corpus"] == "scifact":
            keys.append(("abstract", obs["abstract_group"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i, obs in enumerate(observations):
        members[find(i)].append(obs["observation_id"])
    for i, obs in enumerate(observations):
        obs["resampling_group"] = sha(sorted(members[find(i)]))


def observations(av_records, sci_claims, documents):
    result = []
    for i, record in enumerate(av_records):
        native_id = f"dev:{i}"
        original, units = canonical_av(record)
        anchor = original[pr.anchor_index(native_id, len(original))]["unit_id"]
        target = next(j for j, unit in enumerate(units) if anchor in unit["ids"])
        result.append({"observation_id": "averitec:" + native_id, "corpus": "averitec",
                       "claim": record["claim"], "exact_claim_group": pr.digest(record["claim"]),
                       "label": pr.canonical_label(record["label"]), "original_label": record["label"],
                       "system": pr.SYSTEM, "header": "Claim:\n" + record["claim"] + "\n\nProvided evidence:\n",
                       "units": units, "targets": {"S": target}, "native_anchor_id": anchor,
                       "original_unit_count": len(original), "unit_count": len(units)})
    docs = {str(d["doc_id"]): d for d in documents}
    if len(docs) != len(documents):
        raise ValueError("Duplicate native document ID")
    for claim in sorted(sci_claims, key=lambda c: c["id"]):
        for doc_id, rationales in sorted(claim["evidence"].items(), key=lambda item: int(item[0])):
            doc = docs[str(doc_id)]
            abstract = doc["abstract"]
            labels = {r["label"] for r in rationales}
            if len(labels) != 1 or not labels <= set(SCI_LABELS[:2]) or not abstract:
                raise ValueError("Missing/mixed/unknown native evidence label or empty abstract")
            sets = [r["sentences"] for r in rationales]
            if (any(not s or len(set(s)) != len(s) for s in sets)
                    or any(type(i) is not int or not 0 <= i < len(abstract) for s in sets for i in s)
                    or len({tuple(sorted(s)) for s in sets}) != len(sets)):
                raise ValueError("Invalid native rationale indices/sets")
            marked = {i for s in sets for i in s}
            aliases = defaultdict(list)
            for i, sentence in enumerate(abstract):
                if not isinstance(sentence, str) or not sentence.strip():
                    raise ValueError("Empty native sentence")
                aliases[sentence].append(i)
            pools = {"R": [], "O": []}
            for indices in aliases.values():
                pools["R" if marked.intersection(indices) else "O"].append(indices[0])
            if not all(pools.values()):
                raise ValueError("Native pair lacks one of the two fixed target pools")
            targets = {}
            for role, namespace in (("R", "marked"), ("O", "unmarked")):
                key = f"paper2-supplement-v1:{namespace}:{claim['id']}:{doc_id}"
                targets[role] = pools[role][int(pr.digest(key), 16) % len(pools[role])]
            result.append({"observation_id": f"scifact:{claim['id']}:{doc_id}", "corpus": "scifact",
                           "claim": claim["claim"], "exact_claim_group": pr.digest(claim["claim"]),
                           "abstract_group": sha(abstract), "doc_id": str(doc_id), "label": next(iter(labels)),
                           "original_label": next(iter(labels)), "system": SCI_SYSTEM,
                           "header": "Claim:\n" + claim["claim"] + "\n\nDocument title:\n" + doc["title"] + "\n\nProvided abstract:\n",
                           "units": [{"text": s, "ids": [i], "urls": [], "rendered": f"Sentence {i}:\n{s}"} for i, s in enumerate(abstract)],
                           "unit_count": len(abstract), "targets": targets, "rationale_sets": sets,
                           "marked_union": sorted(marked), "pools": pools,
                           "target_aliases": {role: aliases[abstract[index]] for role, index in targets.items()}})
    for corpus in ARMS:
        component_groups([r for r in result if r["corpus"] == corpus])
    return result


def native_observations():
    av, av_source = fm.native_source(HERE / "data" / "averitec_dev.json")
    directory = HERE / "data" / "scifact_official_20260930"
    actual = {name: fm.file_sha(directory / name) for name in SCI_FILES}
    if actual != SCI_FILES:
        raise ValueError("Official SciFact byte binding changed")
    obs = observations(av, fm.ev.read_jsonl(directory / "claims_dev.jsonl"), fm.ev.read_jsonl(directory / "corpus.jsonl"))
    av_obs, sci = [[o for o in obs if o["corpus"] == c] for c in ARMS]
    facts = (len(av_obs), len(sci), len({o["exact_claim_group"] for o in av_obs}),
             len({o["exact_claim_group"] for o in sci}), len({o["abstract_group"] for o in sci}),
             len({o["resampling_group"] for o in sci}), sum(o["unit_count"] == 1 for o in av_obs))
    if facts != (500, 209, 491, 188, 182, 164, 109):
        raise ValueError("Native population/group/degeneracy counts changed")
    if Counter(o["label"] for o in sci) != {"SUPPORT": 138, "CONTRADICT": 71}:
        raise ValueError("Native SciFact label support changed")
    return obs, {"averitec": av_source, "scifact": {"directory": str(directory.resolve()), "files": actual}}


def certified_runs(tokenizer, text):
    """A lower bound on tokens for an independently newline-separated copy."""
    name = type(tokenizer).__name__
    if name not in PATTERNS or tokenizer.pat.pattern != PATTERNS[name] or not hasattr(tokenizer, "byte_decoder"):
        raise ValueError("Unsupported tokenizer: copy-bound proof requires inspected byte-BPE regex")
    # Qwen2 applies NFC in prepare_for_tokenization; raw combining marks could
    # otherwise overcount letter runs and invalidate the upper-bound proof.
    text, _ = tokenizer.prepare_for_tokenization(text)
    added = list(tokenizer.get_added_vocab())
    if any("\n" in token or "\r" in token or not token for token in added):
        raise ValueError("Added token can span a copy boundary; numerical design review required")
    # All added spans are removed before counting; lstrip/rstrip only consume whitespace.
    for token in sorted(added, key=len, reverse=True):
        text = text.replace(token, " ")
    runs = len(regex.findall(r"\p{L}+", text))
    if not runs:
        raise ValueError("No certified per-copy letter runs")
    return runs


def raw_request(obs, arm, evidence):
    return {"id": sha([VERSION, obs["observation_id"], arm]), "metadata": {},
            "messages": [{"role": "system", "content": obs["system"]},
                         {"role": "user", "content": obs["header"] + evidence}]}


def maximal_copies(make, budget, count, letter_runs):
    if type(letter_runs) is not int or letter_runs < 1:
        raise ValueError("Positive integer certified letter-run bound required")
    upper = budget // letter_runs
    if not 1 <= upper <= 4096:
        raise ValueError("Certified copy bound outside 1..4096; no truncated search permitted")
    measured, selected = [], None
    for k in range(1, upper + 1):
        lengths = [count(fm.wrapped(make(k, side))["messages"]) for side in ("front", "back")]
        measured.append([k, *lengths])
        if max(lengths) <= budget:
            selected = k
    if selected is None:
        raise ValueError("No whole copy fits both placements; no replacement target permitted")
    return selected, {"letter_runs_per_copy": letter_runs, "certified_upper_bound": upper,
                      "evaluated_k_front_back_tokens": measured, "selected_k": selected, "budget_tokens": budget}


def build_requests(obs_list, tokenizer, bound_function=certified_runs):
    raw_count = fm.token_counter(tokenizer)
    def count(messages):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 thinking=False, enable_thinking=False)
        if messages[1]["content"] not in rendered:
            raise ValueError("Chat template transforms/removes the literal user payload")
        return raw_count(messages)
    requests, references, padding, copies = [], [], [], []
    for obs in obs_list:
        units = [unit["rendered"] for unit in obs["units"]]
        evidence = "\n\n".join(units)
        bases = {"C": raw_request(obs, "C", evidence), "P": raw_request(obs, "P", evidence),
                 "B": raw_request(obs, "B", evidence + "\n\n" + evidence)}
        bases["W"] = raw_request(obs, "W", evidence + "\n\n" + obs["header"] + evidence)
        budget = count(fm.wrapped(bases["B"])["messages"])
        target_tokens = max(budget, count(fm.wrapped(bases["W"])["messages"]))
        for role, index in obs["targets"].items():
            unit = units[index]
            def make(k, side):
                extra = "\n\n".join([unit] * k)
                content = extra + "\n\n" + evidence if side == "front" else evidence + "\n\n" + extra
                return raw_request(obs, role + "-" + side, content)
            k, audit = maximal_copies(make, budget, count, bound_function(tokenizer, unit))
            copies.append({**audit, "observation_id": obs["observation_id"], "role": role,
                           "target_native_index": index, "target_characters": len(unit), "target_sha256": pr.digest(unit)})
            for side in ("front", "back"):
                bases[role + "-" + side] = make(k, side)
        emitted = {}
        for arm in ARMS[obs["corpus"]]:
            base = bases[arm]
            if arm == "C":
                request = fm.wrapped(base)
            else:
                request, audit = fm.padding_control(base, target_tokens, count)
                padding.append({**audit, "id": request["id"], "observation_id": obs["observation_id"], "arm": arm})
            # Full input reconstruction is structural, never a substring-only permission to omit facts.
            if request["messages"][1]["content"].split(fm.PAD_START)[0] != base["messages"][1]["content"]:
                raise ValueError("Padding altered complete evidence")
            emitted[arm] = request
            requests.append(request)
            references.append({"id": request["id"], "claim_id": obs["observation_id"], "corpus": obs["corpus"],
                               "exact_claim_group": obs["exact_claim_group"], "resampling_group": obs["resampling_group"],
                               "label": obs["label"], "condition": arm, "method": "fixed_policy",
                               "unit_count": obs["unit_count"]})
        if obs["corpus"] == "averitec" and obs["unit_count"] == 1:
            if not (emitted["B"]["messages"] == emitted["S-front"]["messages"] == emitted["S-back"]["messages"]):
                raise ValueError("Single-content B/selective identity violated")
            if copies[-1]["selected_k"] != 1:
                raise ValueError("Single-content copy count is not one")
    lengths = [{"id": r["id"], "prompt_sha256": sha(r["messages"]), "input_tokens": count(r["messages"]),
                "over_input_ceiling": count(r["messages"]) > 8192} for r in requests]
    return requests, references, padding, copies, lengths


def coverage(references, padding, lengths):
    pads, lens = {r["id"]: r for r in padding}, {r["id"]: r for r in lengths}
    groups = defaultdict(list)
    for ref in references:
        groups[(ref["corpus"], ref["claim_id"])].append(ref)
    result = {}
    for corpus in ARMS:
        records = [rows for (name, _), rows in groups.items() if name == corpus]
        good = [rows[0]["claim_id"] for rows in records if all(
            not lens[r["id"]]["over_input_ceiling"] and (r["condition"] == "C" or pads[r["id"]]["exact_length_match"]) for r in rows)]
        result[corpus] = {"records": len(records), "matched_nonlong_records": len(good),
                          "coverage": fm.fraction(len(good), len(records)), "eligible_ids": good,
                          "guard_pass": bool(records) and len(good) / len(records) >= .95}
    return result


def prepare(args):
    if args.output.exists():
        raise ValueError("Fresh supplement preparation directory required")
    source_binding = bindings()
    obs, sources = native_observations()
    model = model_binding(args.model_key)
    tokenizer = load_local_tokenizer(Path(model["path"]))
    args.output.mkdir(parents=True)
    fm.write_rows(args.output / "observations.jsonl", obs)
    try:
        artifacts = build_requests(obs, tokenizer)
    except (ValueError, RuntimeError) as error:
        fm.write_json(args.output / "preparation.json", {"version": VERSION, "status": "STOP_NUMERICAL_OR_INTEGRITY",
                      "error": str(error), "observations": len(obs), "sources": sources, "model": model,
                      "binding": source_binding, "observations_sha256": fm.file_sha(args.output / "observations.jsonl")})
        raise
    for name, rows in zip(ARTIFACTS, artifacts):
        fm.write_rows(args.output / name, rows)
    requests, refs, pads, _, lengths = artifacts
    guard = coverage(refs, pads, lengths)
    manifest = {"version": VERSION, "status": "PREPARED_NOT_RUN" if all(x["guard_pass"] for x in guard.values()) else "STOP_PREPARATION_GUARD",
                "sources": sources, "model": model, "binding": source_binding, "coverage": guard,
                "files": {n: fm.file_sha(args.output / n) for n in (*ARTIFACTS, "observations.jsonl")},
                "observations": len(obs), "logical_requests": len(requests),
                "unique_messages": len({sha(r["messages"]) for r in requests}),
                "logical_tokens_per_realization": sum(x["input_tokens"] for x in lengths),
                "tokenizer_class": type(tokenizer).__name__, "tokenizer_pattern": tokenizer.pat.pattern}
    if len(requests) != 4672:
        raise ValueError("Incomplete frozen request population")
    fm.write_json(args.output / "preparation.json", manifest)
    print(json.dumps({**{k: manifest[k] for k in ("status", "observations", "logical_requests", "unique_messages", "logical_tokens_per_realization")},
                      "coverage": {c: {k: v for k, v in a.items() if k != "eligible_ids"} for c, a in guard.items()}}))
    if manifest["status"] != "PREPARED_NOT_RUN":
        raise ValueError("Frozen pre-inference population guard failed")


def load_prepared(path):
    manifest = json.loads((path / "preparation.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("status") != "PREPARED_NOT_RUN" or manifest["binding"] != bindings():
        raise ValueError("Nonrunnable preparation or changed source/protocol/test binding")
    obs, sources = native_observations()
    model = model_binding(manifest["model"]["key"])
    if sources != manifest["sources"] or model != manifest["model"]:
        raise ValueError("Source/model binding changed")
    for name, digest in manifest["files"].items():
        if fm.file_sha(path / name) != digest:
            raise ValueError("Prepared artifact hash changed: " + name)
    if fm.ev.read_jsonl(path / "observations.jsonl") != obs:
        raise ValueError("Native observations do not reconstruct")
    artifacts = build_requests(obs, load_local_tokenizer(Path(model["path"])))
    for name, rows in zip(ARTIFACTS, artifacts):
        if fm.ev.read_jsonl(path / name) != rows:
            raise ValueError("Prepared messages/reference/audit do not reconstruct: " + name)
    if manifest["coverage"] != coverage(artifacts[1], artifacts[2], artifacts[4]):
        raise ValueError("Population coverage audit does not reconstruct")
    return manifest, artifacts


def native_outcomes(references, predictions):
    av = fm.ev.outcomes([r for r in references if r["corpus"] == "averitec"], predictions)
    sci = []
    for ref in references:
        if ref["corpus"] != "scifact":
            continue
        out = predictions[ref["id"]]
        response = out.get("response")
        response = response.strip() if isinstance(response, str) else None
        valid = out["status"] == "ok" and response in tuple("ABC")
        label = SCI_LABELS["ABC".index(response)] if valid else None
        sci.append({**ref, "prediction": label, "correct": int(label == ref["label"]),
                    "logical_input_tokens": out["input_tokens"], "logical_output_tokens": out["output_tokens"],
                    "output_status": "valid" if valid else out["status"] if out["status"] != "ok" else "invalid_format"})
    return av + sci


def construction_status(rows, padding):
    """Keep raw predictions, but a failed matched-policy construction scores zero."""
    failed = {p["id"] for p in padding if not p["exact_length_match"]}
    for row in rows:
        if row["id"] in failed:
            row["raw_prediction_before_construction_check"] = row["prediction"]
            row["prediction"], row["correct"], row["output_status"] = None, 0, "construction_length_mismatch"
    return rows


def cell_metrics(rows):
    labels = pr.LABELS if rows[0]["corpus"] == "averitec" else SCI_LABELS
    f1, support = {}, Counter(r["label"] for r in rows)
    for label in labels:
        tp = sum(r["label"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["label"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["label"] == label and r["prediction"] != label for r in rows)
        f1[label] = fm.fraction(2 * tp, 2 * tp + fp + fn) or 0.0
    return {"records": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "macro_f1": sum(f1.values()) / len(labels), "per_class_f1": f1,
            "per_class_support": {label: support[label] for label in labels},
            "confusion": {label: dict(Counter(r["prediction"] or "INVALID" for r in rows if r["label"] == label)) for label in labels},
            "status_counts": dict(Counter(r["output_status"] for r in rows)),
            "valid_output_fraction": sum(r["prediction"] is not None for r in rows) / len(rows),
            "logical_input_tokens": sum(r["logical_input_tokens"] for r in rows),
            "logical_output_tokens": sum(r["logical_output_tokens"] for r in rows)}


def interval(values, family=1, replicates=fm.BOOTSTRAP):
    result = fm.group_interval(values, family=family, replicates=replicates)
    result["weighting"] = "native-record weighted; exact claim groups (AV) or exact claim/full-abstract components (SciFact)"
    return result


def primary_contrast(rows_a, rows_b, treatment, replicates=fm.BOOTSTRAP):
    indexes = [{(r["claim_id"], r["condition"]): r for r in rows if r["corpus"] == "scifact"} for rows in (rows_a, rows_b)]
    records = []
    ids = sorted({key[0] for key in indexes[0]})
    for identity in ids:
        differences, valid = [], True
        for index in indexes:
            for side in ("front", "back"):
                t, c = index[(identity, "B" if treatment == "B" else "R-" + side)], index[(identity, "O-" + side)]
                differences.append(t["correct"] - c["correct"])
                valid &= t["prediction"] is not None and c["prediction"] is not None
        common_rob = all(index[(identity, arm)]["prediction"] is not None for index in indexes
                         for arm in ("B", "R-front", "R-back", "O-front", "O-back"))
        records.append({"observation_id": identity, "group": t["resampling_group"], "values_Afront_Aback_Bfront_Bback": differences,
                        "difference": sum(differences) / 4, "common_valid": bool(valid), "common_ROB_valid": common_rob})
    complete = [r for r in records if r["common_valid"]]
    result = interval([(r["group"], r["difference"]) for r in records], family=4, replicates=replicates)
    positions = {side: sum((r["values_Afront_Aback_Bfront_Bback"][i] + r["values_Afront_Aback_Bfront_Bback"][i + 2]) / 2 for r in records) / len(records)
                 for i, side in enumerate(("front", "back"))}
    common_mean = fm.fraction(sum(r["difference"] for r in complete), len(complete))
    rob = [r for r in records if r["common_ROB_valid"]]
    rob_mean = fm.fraction(sum(r["difference"] for r in rob), len(rob))
    passed = (result["difference"] >= .03 and result["ci_familywise"][0] > 0 and min(positions.values()) > 0
              and len(complete) / len(records) >= .95 and common_mean is not None and common_mean > 0
              and len(rob) / len(records) >= .95 and rob_mean is not None and rob_mean > 0)
    return {"contrast": treatment + "_minus_O", "signed": result, "positions": positions,
            "common_valid_records": len(complete), "common_valid_coverage": len(complete) / len(records),
            "common_valid_difference": common_mean, "common_ROB_valid_records": len(rob),
            "common_ROB_coverage": len(rob) / len(records), "common_ROB_difference": rob_mean,
            "mechanical_screen_pass": passed}, records


def summarize(references, a, b, padding=(), replicates=fm.BOOTSTRAP):
    rows_a, rows_b = [construction_status(native_outcomes(references, p), padding) for p in (a, b)]
    cells, diagnostics, paired = {}, {}, []
    for corpus, arms in ARMS.items():
        for arm in arms:
            members = [[r for r in rows if r["corpus"] == corpus and r["condition"] == arm] for rows in (rows_a, rows_b)]
            metrics = [cell_metrics(rows) for rows in members]
            cells[corpus + "/" + arm] = {"A": metrics[0], "B": metrics[1], "mean_accuracy": sum(x["accuracy"] for x in metrics) / 2,
                                        "mean_macro_f1": sum(x["macro_f1"] for x in metrics) / 2}
        selective = ("S",) if corpus == "averitec" else ("R", "O")
        for role in selective:
            cells[corpus + "/" + role + "-fixed_average"] = {
                "mean_accuracy": sum(cells[corpus + "/" + role + "-" + side]["mean_accuracy"] for side in ("front", "back")) / 2,
                "mean_macro_f1": sum(cells[corpus + "/" + role + "-" + side]["mean_macro_f1"] for side in ("front", "back")) / 2,
                "definition": "equal average of the two fixed placements and two fresh realizations; not a selected prediction"}
        comparisons = [("P", "C"), ("B", "C"), ("W", "C"), ("B", "P"), ("W", "B")]
        comparisons.extend((role + "-" + side, "B") for role in selective for side in ("front", "back"))
        strata = {"all": None}
        if corpus == "averitec":
            strata.update({name: {r["claim_id"] for r in references if r["corpus"] == corpus and (r["unit_count"] == 1) == single}
                           for name, single in (("single_content", True), ("multiple_content", False))})
        # Adapter for the helper's legacy group field; original outcome/reference fields stay intact.
        grouped = [[{**r, "exact_claim_group": r["resampling_group"]} for r in rows if r["corpus"] == corpus] for rows in (rows_a, rows_b)]
        for stratum, eligible in strata.items():
            for treatment, control in comparisons:
                key = f"{corpus}/{stratum}/{treatment}_minus_{control}"
                report, details = fm.paired_readout(*grouped, (treatment, "fixed_policy"), (control, "fixed_policy"),
                                                    family=1, eligible_ids=eligible, replicates=replicates)
                report["interpretation"] = "descriptive only; group IDs denote declared exact-text resampling components"
                for value in report.values():
                    if isinstance(value, dict) and "weighting" in value:
                        value["weighting"] = "native-record weighted; declared exact-text resampling groups"
                for name, rows in (("A", rows_a), ("B", rows_b)):
                    index = {(r["claim_id"], r["condition"]): r for r in rows if r["corpus"] == corpus}
                    chosen = [d["claim_id"] for d in details]
                    report["by_realization"][name]["changed_wrong_labels"] = sum(
                        index[(i, treatment)]["prediction"] is not None and index[(i, control)]["prediction"] is not None
                        and not index[(i, treatment)]["correct"] and not index[(i, control)]["correct"]
                        and index[(i, treatment)]["prediction"] != index[(i, control)]["prediction"] for i in chosen)
                diagnostics[key] = report
                for d in details:
                    d["resampling_group"] = d.pop("exact_claim_group")
                    paired.append({**d, "contrast": key})
            for role in selective:
                placement_rows = [[r for r in paired if r["contrast"] == f"{corpus}/{stratum}/{role}-{side}_minus_B"] for side in ("front", "back")]
                indexes = [{r["claim_id"]: r for r in rows} for rows in placement_rows]
                averages = []
                for identity, front in indexes[0].items():
                    back = indexes[1][identity]
                    common = front["complete_four_valid"] and back["complete_four_valid"]
                    averages.append({"claim_id": identity, "resampling_group": front["resampling_group"],
                                     "average_signed_correctness_change": (front["average_signed_correctness_change"] + back["average_signed_correctness_change"]) / 2,
                                     "complete_both_placements_valid": common,
                                     "correctness_crossing_excess": sum((r["intervention"]["correctness_crossings"] - r["repeat"]["correctness_crossings"]) / 4 for r in (front, back)) if common else None})
                key = f"{corpus}/{stratum}/{role}-fixed_average_minus_B"
                complete = [r for r in averages if r["complete_both_placements_valid"]]
                diagnostics[key] = {"paired_records": len(averages), "complete_records": len(complete),
                    "signed": interval([(r["resampling_group"], r["average_signed_correctness_change"]) for r in averages], replicates=replicates),
                    "crossing_excess": interval([(r["resampling_group"], r["correctness_crossing_excess"]) for r in complete], replicates=replicates),
                    "interpretation": "descriptive fixed placement average; crossing uses the common-valid placement panel"}
                paired.extend({**r, "contrast": key} for r in averages)
    primary, primary_records = {}, []
    for treatment in ("R", "B"):
        report, details = primary_contrast(rows_a, rows_b, treatment, replicates)
        primary[report["contrast"]] = report
        primary_records.extend({**r, "contrast": report["contrast"]} for r in details)
    return {"cells": cells, "primary": primary, "descriptive": diagnostics,
            "physical_cost": {"A": fm.physical_cost(a), "B": fm.physical_cost(b)}}, rows_a, rows_b, paired, primary_records


def checked_repeats(prepared, paths):
    manifest, artifacts = load_prepared(prepared)
    req, refs, _, _, lengths = artifacts
    runs = [fm.checked_run(path, prepared / "requests.jsonl", req, lengths, manifest["model"]) for path in paths]
    (a, ca, ma), (b, cb, mb) = runs
    if (ca != cb or not all(m.get("process_id") and m.get("start_utc") and m.get("end_utc") for m in (ma, mb))
            or (ma["process_id"], ma["start_utc"]) == (mb["process_id"], mb["start_utc"])
            or ma["end_utc"] > mb["start_utc"]):
        raise ValueError("Require identical configuration and fresh chronological whole-file A then B")
    if any(a[k]["batch_size"] != b[k]["batch_size"] or a[k]["cached"] != b[k]["cached"] for k in a):
        raise ValueError("Fresh-repeat batch/cache layout differs")
    if any(r["output_tokens"] > 16 for p in (a, b) for r in p.values()):
        raise ValueError("Output-token ceiling exceeded")
    return manifest, refs, runs, artifacts[2]


def expected_argv(prepared, output, model):
    return [str(ROOT / "shared" / "generate.py"), "--input", str((prepared / "requests.jsonl").resolve()),
            "--output", str(output.resolve()), "--model", model["path"], "--model-revision", model["revision"],
            "--batch-size", "8", "--max-input-tokens", "8192", "--max-new-tokens", "16",
            "--num-shards", "1", "--shard-index", "0", "--seed", "17"]


def utc(value):
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if time.tzinfo is None:
        raise ValueError("Execution times require an explicit UTC offset")
    return time


def check_receipt(receipt, bundles, paths, prepared):
    entries = receipt.get("runs", [])
    if len(entries) != 4 or len({(r["model_key"], r["realization"]) for r in entries}) != 4:
        raise ValueError("Execution receipt must contain exactly four distinct fixed runs")
    occupied = 0
    by_key = {}
    for entry in entries:
        model, realization = entry["model_key"], entry["realization"]
        if model not in fm.MODELS or realization not in ("A", "B"):
            raise ValueError("Unknown execution receipt cell")
        i = ("A", "B").index(realization)
        path, bundle = paths[model][i], bundles[model]
        manifest = bundle[2][i][2]
        begin, end = utc(entry["launch_utc"]), utc(entry["finish_utc"])
        if (entry.get("visible_gpu_count") != 1 or entry.get("output_existed_before_launch") is not False
                or entry.get("resume") is not False or Path(entry["output_path"]).resolve() != path.resolve()
                or entry.get("output_sha256") != fm.file_sha(path)
                or entry.get("manifest_sha256") != fm.file_sha(path.with_suffix(path.suffix + ".manifest.json"))
                or entry.get("argv", [])[1:] != expected_argv(prepared[model], path, bundle[0]["model"])
                or end < begin or begin > utc(manifest["start_utc"]) or end < utc(manifest["end_utc"])):
            raise ValueError("Missing/inconsistent fresh launch, bytes, arguments, or occupied-GPU time receipt")
        occupied += (end - begin).total_seconds()
        by_key[(model, realization)] = (begin, end)
    if any(by_key[(model, "A")][1] > by_key[(model, "B")][0] for model in fm.MODELS):
        raise ValueError("A must finish before the model's B process starts")
    if occupied > 14400 or receipt.get("additional_attempts", []) != []:
        raise ValueError("GPU budget exceeded or unreviewed additional attempt exists")
    return occupied


def evaluate(args):
    if args.output.exists():
        raise ValueError("Fresh supplement evaluation directory required")
    prepared = {model: getattr(args, model + "_prepared") for model in fm.MODELS}
    paths = {model: [getattr(args, model + "_" + r.lower()) for r in ("A", "B")] for model in fm.MODELS}
    if len({p.resolve() for pair in paths.values() for p in pair}) != 4:
        raise ValueError("Four distinct output paths required")
    # All four must be complete and bound before any statistical calculation.
    bundles = {model: checked_repeats(prepared[model], paths[model]) for model in fm.MODELS}
    if any(bundle[0]["model"]["key"] != model for model, bundle in bundles.items()):
        raise ValueError("Prepared model keys are swapped")
    receipt = json.loads(args.execution_receipt.read_text())
    occupied = check_receipt(receipt, bundles, paths, prepared)
    report = {"version": VERSION, "status": "COMPLETED_MEASUREMENT_NOT_PUBLICATION_APPROVAL", "models": {},
              "binding": bindings(), "execution_receipt_sha256": fm.file_sha(args.execution_receipt), "occupied_gpu_seconds": occupied}
    tables = {}
    for model, (manifest, refs, runs, padding) in bundles.items():
        summary, ra, rb, paired, primary = summarize(refs, runs[0][0], runs[1][0], padding)
        summary.update(preparation_sha256=fm.file_sha(prepared[model] / "preparation.json"), coverage=manifest["coverage"],
                       generation_manifests=[r[2] for r in runs],
                       output_sha256=[fm.file_sha(p) for p in paths[model]])
        report["models"][model] = summary
        for name, rows in (("outcomes_A", ra), ("outcomes_B", rb), ("paired", paired), ("primary_records", primary)):
            tables[model + "_" + name + ".jsonl"] = rows
    report["mechanical_screen_pass"] = all(p["mechanical_screen_pass"] for m in report["models"].values() for p in m["primary"].values())
    costs = [c for m in report["models"].values() for c in m["physical_cost"].values()]
    if (sum(c["logical_requests"] for c in costs) != 18688 or sum(c["charged_input_tokens"] for c in costs) > 153092096
            or sum(c["charged_output_tokens"] for c in costs) > 299008):
        raise ValueError("Fixed request/token budget violated")
    args.output.mkdir(parents=True)
    for name, rows in tables.items():
        fm.write_rows(args.output / name, rows)
    report["outcome_files"] = {name: fm.file_sha(args.output / name) for name in tables}
    fm.write_json(args.output / "summary.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "mechanical_screen_pass": report["mechanical_screen_pass"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--model-key", choices=fm.MODELS, required=True)
    prep.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--prepared", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    for model in fm.MODELS:
        for suffix in ("prepared", "a", "b"):
            evaluation.add_argument("--" + model + "-" + suffix, type=Path, required=True)
    evaluation.add_argument("--execution-receipt", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        manifest, _ = load_prepared(args.prepared)
        print(json.dumps({"status": "VERIFIED_PREPARED_NOT_RUN", "logical_requests": manifest["logical_requests"]}))
    else:
        {"prepare": prepare, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
