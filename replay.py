"""Controlled repackaging of existing AVeriTeC QA evidence, not Web propagation."""

from hashlib import sha256
from math import isfinite


def extract_units(record):
    """Whitelist evidence fields; never include label or reference justification.

    Unit IDs identify existing QA records, not independently acquired evidence.
    Empty answer lists use the official evaluator's existing rendering rule.
    """
    if not isinstance(record, dict) or not isinstance(record.get("questions"), list):
        raise ValueError("record must contain a questions list")
    units = []
    for qindex, question in enumerate(record["questions"]):
        if not isinstance(question, dict) or not isinstance(question.get("question"), str):
            raise ValueError("question must contain text")
        answers = question.get("answers")
        if isinstance(answers, dict):
            answers = [answers]
        if not isinstance(answers, list):
            raise ValueError("answers must be a list or the documented single object")
        if not answers:
            answers = [{"answer": "No answer could be found.", "source_url": ""}]
        for aindex, answer in enumerate(answers):
            if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
                raise ValueError("answer must contain text")
            text = "Question: " + question["question"] + "\nAnswer: " + answer["answer"]
            if str(answer.get("answer_type", "")).lower() == "boolean":
                explanation = answer.get("boolean_explanation", "")
                if not isinstance(explanation, str):
                    raise ValueError("boolean_explanation must be text")
                if explanation:
                    text += "\nExplanation: " + explanation
            url = answer.get("source_url", "")
            if url is None:
                url = ""
            if not isinstance(url, str):
                raise ValueError("source_url must be text or null")
            units.append({"unit_id": f"q{qindex}:a{aindex}", "text": text,
                          "source_urls": [url] if url else []})
    return units


def anchor_index(claim_id, n):
    """Choose a replication target without consulting labels or model outputs."""
    if not isinstance(claim_id, str) or type(n) is not int or n < 1:
        raise ValueError("a string claim ID and positive unit count are required")
    return int(sha256(("paper2-anchor-v1:" + claim_id).encode()).hexdigest(), 16) % n


def make_replay(units, mode="clean", copies=1, anchor=0):
    """Return packets of unchanged units; every original unit remains present.

    'duplicate' appends copies-1 occurrences of one unit to singleton packets.
    'partial_copy' pairs each other unit with copies occurrences of the anchor.
    The latter contains shared text inside different packets, with no new facts.
    """
    if not units or type(copies) is not int or copies < 1:
        raise ValueError("nonempty units and a positive integer copy count are required")
    if type(anchor) is not int or not 0 <= anchor < len(units):
        raise ValueError("invalid anchor")
    if len({u["unit_id"] for u in units}) != len(units):
        raise ValueError("input unit IDs must be unique")
    if mode == "clean":
        if copies != 1:
            raise ValueError("clean uses copies=1")
        return [[u] for u in units]
    if mode == "duplicate":
        return [[u] for u in units] + [[units[anchor]] for _ in range(copies - 1)]
    if mode == "partial_copy":
        if len(units) < 2:
            raise ValueError("partial_copy requires at least two original QA units")
        return [[units[anchor]] * copies + [u]
                for i, u in enumerate(units) if i != anchor]
    raise ValueError("unknown replay mode")


def ledger(packets):
    """Canonical exact-fragment baseline, retaining every observed source URL.

    Only byte-identical rendered QA text is merged. Sorting makes clean/replay
    evidence text identical regardless of packing. IDs and occurrence counts
    are diagnostic metadata, never evidence of natural source independence.
    """
    grouped = {}
    for packet in packets:
        for unit in packet:
            text = unit["text"]
            if not isinstance(text, str) or not text:
                raise ValueError("unit text must be nonempty")
            entry = grouped.setdefault(text, {"text": text, "unit_ids": set(),
                                             "source_urls": set(), "occurrences": 0})
            entry["unit_ids"].add(unit["unit_id"])
            entry["source_urls"].update(unit["source_urls"])
            entry["occurrences"] += 1
    return [{**entry, "unit_ids": sorted(entry["unit_ids"]),
             "source_urls": sorted(entry["source_urls"])}
            for _, entry in sorted(grouped.items())]


def deduplicate_packets(packets, threshold=None):
    """Exact whole-packet dedup, or a specified lexical Jaccard control.

    This baseline can lose unique units when a mixed packet resembles an earlier
    one. It is not the required semantic/neural deduplication comparator.
    """
    if threshold is not None and (isinstance(threshold, bool)
                                  or not isinstance(threshold, (int, float))
                                  or not isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must lie in [0, 1]")
    kept, signatures, token_sets = [], set(), []
    for packet in packets:
        text = "\n\n".join(unit["text"] for unit in packet)
        signature = tuple((unit["text"], tuple(unit["source_urls"])) for unit in packet)
        if signature in signatures:
            continue
        words = set(text.split())
        if threshold is not None and any(
                len(words & other) / max(1, len(words | other)) >= threshold
                for other in token_sets):
            continue
        kept.append(packet)
        signatures.add(signature)
        token_sets.append(words)
    return kept


def render_packets(packets):
    """Public inference text: occurrence numbers, unchanged QA, and actual URLs."""
    return "\n\n".join(
        f"Evidence packet {i + 1}:\n" + "\n\n".join(
            u["text"] + "\nSources: " + ("; ".join(u["source_urls"]) or "unspecified")
            for u in packet)
        for i, packet in enumerate(packets))


def render_ledger(entries, show_counts=False):
    """Never render original-unit IDs; exact text dedup does not require them."""
    return "\n\n".join(
        f"Evidence unit {i + 1}:\n" + entry["text"]
        + "\nSources: " + ("; ".join(entry["source_urls"]) or "unspecified")
        + (f"\nOccurrences of this identical text: {entry['occurrences']}" if show_counts else "")
        for i, entry in enumerate(entries))
