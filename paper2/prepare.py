"""Build gold-free local-model requests and separate native-label references."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from replay import (anchor_index, deduplicate_packets, extract_units, ledger,
                    make_replay, render_ledger, render_packets)

LABELS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking")
SYSTEM = (
    "Decide the claim's veracity using the provided question-answer evidence. "
    "Repetition does not add factual information. Packet and unit numbers are presentation "
    "identifiers, not judgments of independent sources. Return exactly one letter, with no "
    "explanation: A = Supported; B = Refuted; C = Not Enough Evidence; "
    "D = Conflicting Evidence/Cherrypicking."
)
METHODS = ("raw_packets", "exact_packet_dedup", "exact_fragment_dedup", "ledger_counts")


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def canonical_label(label):
    if label == "Conflicting Evidence/Cherry-picking":
        label = LABELS[3]
    if label not in LABELS:
        raise ValueError("unrecognized native label")
    return label


def build_claim(record, claim_id, split, copies=4):
    claim = record.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must contain nonempty text")
    units = extract_units(record)
    if not units:
        raise ValueError("claim has no provided QA evidence units")
    label = canonical_label(record.get("label"))
    anchor = anchor_index(claim_id, len(units))
    conditions = [("clean", "clean", 1), (f"duplicate{copies}", "duplicate", copies)]
    if len(units) >= 2:
        conditions.extend([("partial1", "partial_copy", 1),
                           (f"partial{copies}", "partial_copy", copies)])
    requests, references = [], []
    group = digest(claim)  # Exact original claim text, without inferred relationships.
    for condition, mode, count in conditions:
        packets = make_replay(units, mode, count, anchor)
        entries = ledger(packets)
        full_ids = {unit["unit_id"] for unit in units}
        for method in METHODS:
            visible = deduplicate_packets(packets) if method == "exact_packet_dedup" else packets
            if method in ("exact_fragment_dedup", "ledger_counts"):
                content = render_ledger(entries, show_counts=method == "ledger_counts")
                retained = {uid for entry in entries for uid in entry["unit_ids"]}
            else:
                content = render_packets(visible)
                retained = {unit["unit_id"] for packet in visible for unit in packet}
            request_id = f"{claim_id}/{condition}/{method}"
            requests.append({"id": request_id,
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": "Claim:\n" + claim
                                           + "\n\nProvided evidence:\n" + content}],
                             "metadata": {"claim_id": claim_id, "exact_claim_group": group,
                                          "split": split, "condition": condition, "method": method,
                                          "evidence_setting": "provided_gold_qa",
                                          "original_units": len(units),
                                          "occurrences": sum(map(len, packets)),
                                          "retained_unit_fraction": len(retained & full_ids) / len(full_ids)}})
            references.append({"id": request_id, "claim_id": claim_id,
                               "exact_claim_group": group, "label": label,
                               "original_label": record["label"],
                               "condition": condition, "method": method})
    return requests, references


def load_records(path):
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("dataset must be a JSON list of objects")
    return records, sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--holdout", type=Path, help="Required for train pilot: official dev JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-size", type=int, default=32)
    parser.add_argument("--copies", type=int, default=4)
    args = parser.parse_args()
    if args.copies < 2 or args.pilot_size < 1:
        parser.error("copies must be at least two and pilot-size must be positive")
    if args.split == "train" and args.holdout is None:
        parser.error("train pilot requires --holdout to remove exact dev-claim overlaps")
    paths = [args.output_dir / name for name in ("requests.jsonl", "references.jsonl", "preparation.json")]
    if any(path.exists() for path in paths):
        parser.error("output files already exist; choose a new frozen output directory")
    records, source_sha = load_records(args.input)
    holdout_claims, holdout_sha = set(), None
    if args.holdout is not None:
        heldout, holdout_sha = load_records(args.holdout)
        holdout_claims = {row["claim"] for row in heldout if isinstance(row.get("claim"), str)}
    eligible, exclusions, seen = [], [], set()
    for index, row in enumerate(records):
        claim = row.get("claim")
        claim_id = f"{args.split}:{index}"
        try:
            if args.split == "train" and claim in holdout_claims:
                raise ValueError("exact_claim_overlaps_locked_dev")
            build_claim(row, claim_id, args.split, args.copies)
            if args.split == "train" and claim in seen:
                raise ValueError("duplicate_exact_train_claim_in_pilot")
            seen.add(claim)
            eligible.append((index, row))
        except (ValueError, TypeError, KeyError) as error:
            exclusions.append({"claim_id": claim_id, "reason": str(error)})
    if args.split == "train":
        eligible.sort(key=lambda pair: digest("paper2-pilot-v1:" + pair[1]["claim"]))
        selected = eligible[:args.pilot_size]
    else:
        selected = eligible  # Official dev is evaluation; no favorable subset selection.
    if not selected:
        parser.error("no eligible records; see source/schema rather than inventing replacements")
    requests, references = [], []
    for index, record in selected:
        x, y = build_claim(record, f"{args.split}:{index}", args.split, args.copies)
        requests.extend(x)
        references.extend(y)
    unique_messages = {json.dumps(row["messages"], ensure_ascii=False, sort_keys=True) for row in requests}
    manifest = {"task": "provided_evidence_native_verification_controlled_replication",
                "source_path": str(args.input.resolve()), "source_sha256": source_sha,
                "holdout_sha256": holdout_sha, "split": args.split,
                "selection": "hash_order_train_pilot" if args.split == "train" else "all_eligible_official_dev",
                "source_records": len(records), "eligible_records": len(eligible),
                "selected_claim_ids": [f"{args.split}:{i}" for i, _ in selected],
                "selected_exact_claim_groups": len({digest(r["claim"]) for _, r in selected}),
                "native_label_counts": dict(Counter(canonical_label(r["label"]) for _, r in selected)),
                "partial_not_applicable": [f"{args.split}:{i}" for i, r in selected if len(extract_units(r)) < 2],
                "exclusions": exclusions, "copies": args.copies, "methods": METHODS,
                "request_count": len(requests), "unique_message_arrays": len(unique_messages),
                "new_human_annotations": 0, "inference_executed": False,
                "natural_lineage_labels": False, "retrieval_executed": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths[:2], (requests, references)):
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["output_sha256"] = {path.name: sha256(path.read_bytes()).hexdigest() for path in paths[:2]}
    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"selected_claims": len(selected), "requests": len(requests),
                      "unique_message_arrays": len(unique_messages), "manifest": str(paths[2])}))


if __name__ == "__main__":
    main()
