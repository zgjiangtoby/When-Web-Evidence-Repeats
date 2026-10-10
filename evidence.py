"""Build gold-free local-model requests and separate native-label references."""

from hashlib import sha256
import json

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
