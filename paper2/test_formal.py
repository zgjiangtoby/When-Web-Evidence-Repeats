"""Synthetic formal measurement checks; no native corpus or model inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import formal as fm
import prepare as pr
from replay import extract_units, ledger, make_replay, render_ledger
from generate import sha


class ConstructedTokenizer:
    """Length oracle with one token per character, exercising the full chat wrapper."""
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] and kwargs["thinking"] is False
        return "<chat>" + "".join(m["role"] + ":" + m["content"] for m in messages) + "<assistant>"

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return list(range(len(text)))


def source_record(label="Supported", text="Constructed claim", units=2):
    return {"claim": text, "label": label, "justification": "HIDDEN JUSTIFICATION SENTINEL",
            "questions": [{"question": f"Constructed question {i}?", "answers": [
                {"answer": f"Constructed answer {i}.", "source_url": f"https://example.org/{i}",
                 "answer_type": "Boolean", "boolean_explanation": f"Existing explanation {i}."}]} for i in range(units)]}


def predictions(requests, lengths, references, realization=0):
    gold = {r["id"]: "ABCD"[pr.LABELS.index(r["label"])] for r in references}
    counts = {r["id"]: r["input_tokens"] for r in lengths}
    result, seen = {}, {}
    for req in requests:
        key = sha(req["messages"])
        if key in seen:
            row = {**result[seen[key]], "id": req["id"], "cached": True, "cache_of": seen[key],
                   "charged_input_tokens": 0, "charged_output_tokens": 0, "elapsed_batch_seconds": 0, "batch_size": 0}
        else:
            long = counts[req["id"]] > 8192
            row = {"id": req["id"], "status": "input_too_long" if long else "ok", "cached": False,
                   "response": None if long else gold[req["id"]], "input_tokens": counts[req["id"]],
                   "output_tokens": 0 if long else 1, "charged_input_tokens": 0 if long else counts[req["id"]],
                   "charged_output_tokens": 0 if long else 1, "batch_size": 0 if long else 1,
                   "elapsed_batch_seconds": 0 if long else .1}
            seen[key] = req["id"]
        row.update(metadata=req["metadata"], request_sha256=sha(req), prompt_sha256=key)
        result[req["id"]] = row
    return result


class FormalTests(unittest.TestCase):
    def test_duplicate_native_unit_ids_preserve_content_not_multiplicity(self):
        record = source_record(units=3)
        record["questions"][0]["answers"].append(deepcopy(record["questions"][0]["answers"][0]))
        raw, expected_refs = pr.build_claim(record, "dev:0", "dev", copies=4)
        requests, references, _, audit, _ = fm.build_requests([record], ConstructedTokenizer())
        by_id = {r["id"]: r for r in requests}
        self.assertEqual((len(requests), len(references), audit[0]["qa_units"]), (18, 18, 4))
        self.assertEqual(references[:len(expected_refs)], expected_refs)
        self.assertEqual([by_id[r["id"]] for r in raw], [fm.wrapped(r) for r in raw])
        self.assertEqual(by_id["dev:0/clean/exact_packet_dedup"]["metadata"]["retained_unit_fraction"], .75)
        for unit in extract_units(record):
            for req in requests:
                self.assertIn(unit["text"], req["messages"][1]["content"])
                self.assertTrue(all(url in req["messages"][1]["content"] for url in unit["source_urls"]))
        fragment = [r["messages"] for r in requests if r["metadata"]["method"] == "exact_fragment_dedup"]
        self.assertTrue(all(messages == fragment[0] for messages in fragment))
        # Same text at a distinct URL remains separate in packet dedup and is unioned in fragments.
        record["questions"][0]["answers"][-1]["source_url"] = "https://example.org/alternate"
        requests = fm.build_requests([record], ConstructedTokenizer())[0]
        self.assertTrue(all("https://example.org/alternate" in r["messages"][1]["content"] for r in requests))
        self.assertEqual(next(r for r in requests if r["id"] == "dev:0/clean/exact_packet_dedup")
                         ["metadata"]["retained_unit_fraction"], 1)
        exact_dedup = pr.deduplicate_packets
        with patch.object(pr, "deduplicate_packets", side_effect=lambda packets: exact_dedup(packets)[:1]):
            with self.assertRaisesRegex(ValueError, "original QA text or source URLs"):
                fm.build_requests([record], ConstructedTokenizer())

    def test_hidden_labels_facts_padding_and_complete_cache_bound_runs(self):
        records = [source_record(), source_record("Refuted", "Second claim", units=1)]
        artifacts = fm.build_requests(records, ConstructedTokenizer())
        requests, references, padding, audit, lengths = artifacts
        changed = deepcopy(records)
        changed[0]["label"] = "Not Enough Evidence"
        changed[0]["justification"] = "Different hidden text"
        self.assertEqual(requests, fm.build_requests(changed, ConstructedTokenizer())[0])
        self.assertNotIn("HIDDEN JUSTIFICATION SENTINEL", json.dumps(requests))
        with self.assertRaisesRegex(ValueError, "unrecognized native label"):
            fm.build_requests([source_record("invented")], ConstructedTokenizer())
        self.assertEqual(len(requests), 27)
        self.assertEqual([r["partial_applicable"] for r in audit], [True, False])
        by_id = {r["id"]: r for r in requests}
        for item in padding:
            target = by_id[item["target_id"]]
            control = by_id[item["id"]]
            self.assertIn(fm.PAD_START, target["messages"][1]["content"])
            self.assertIn(fm.PAD_START, control["messages"][1]["content"])
            self.assertEqual(item["token_mismatch"], item["actual_tokens"] - item["target_tokens"])
            self.assertLessEqual(item["search_candidates"], 50)
            original = by_id[item["baseline_id"]]["messages"][1]["content"].split(fm.PAD_START)[0]
            self.assertEqual(control["messages"][1]["content"].split(fm.PAD_START)[0], original)
        units = extract_units(records[0])
        reference = render_ledger(ledger(make_replay(units)))
        self.assertEqual(reference, render_ledger(ledger(make_replay(units, "partial_copy", 4))))
        fragments = [r for r in requests if r["metadata"]["claim_id"] == "dev:0" and
                     r["metadata"]["method"] == "exact_fragment_dedup"]
        self.assertTrue(all(r["messages"] == fragments[0]["messages"] for r in fragments))
        for unit in units:
            self.assertIn(unit["text"], fragments[0]["messages"][1]["content"])
            self.assertTrue(all(url in fragments[0]["messages"][1]["content"] for url in unit["source_urls"]))
        # Deterministic nonexact oracle: every count is even; odd target cannot be matched.
        raw, _ = pr.build_claim(records[0], "dev:0", "dev")
        _, mismatch = fm.padding_control(raw[0], 101, lambda messages: 2 * len(messages[1]["content"]))
        self.assertFalse(mismatch["exact_length_match"])
        self.assertNotEqual(mismatch["token_mismatch"], 0)
        a = predictions(requests, lengths, references)
        report, _, _, _ = fm.summarize(references, a, a, padding, replicates=20)
        self.assertEqual(report["raw_primary"]["duplicate4_minus_clean"]["signed_accuracy_change"]["difference"], 0)
        self.assertGreater(report["physical_cost"]["A"]["cached_requests"], 0)
        self.assertEqual({a[r["id"]]["response"] for r in fragments}, {"A"})
        with tempfile.TemporaryDirectory(prefix="paper2_formal_constructed_") as directory:
            root = Path(directory)
            request_file = root / "requests.jsonl"
            fm.write_rows(request_file, requests)
            model = {"path": "constructed-model", "revision": "constructed", "files": {"config.json": "constructed"}, "versions": {}}
            config = {"input_sha256": fm.file_sha(request_file), "code_sha256": fm.RUNNER_SHA,
                      "model_path": model["path"], "model_revision": model["revision"], "model_config_sha256": "constructed",
                      "versions": {}, "num_shards": 1, "shard_index": 0, "batch_size": 8, "seed": 17,
                      "max_input_tokens": 8192, "max_new_tokens": 16, "do_sample": False, "thinking": False,
                      "dtype": "bfloat16", "attention": "sdpa", "tokenizer_use_fast": False}
            path = root / "A.jsonl"
            output = [{**r, "run_signature": sha(config)} for r in a.values()]
            fm.write_rows(path, output)
            fm.write_json(path.with_suffix(".jsonl.manifest.json"), {"config": config, "run_signature": sha(config),
                "status": "COMPLETED", "completed": len(requests), "total_requests": len(requests), "process_id": 1, "start_utc": "fixture"})
            self.assertEqual(len(fm.checked_run(path, request_file, requests, lengths, model)[0]), len(requests))
            corrupt = next(r for r in output if r["cached"])
            corrupt["response"] = "D"
            path.write_text("".join(json.dumps(r) + "\n" for r in output))
            with self.assertRaisesRegex(ValueError, "Identical messages"):
                fm.checked_run(path, request_file, requests, lengths, model)

    def test_matched_four_outcomes_failures_and_record_weighted_groups(self):
        # One repair and one break: zero net, nonzero crossing; identical fresh repeat.
        refs = [{"id": f"dev:{i}/{c}/raw_packets", "claim_id": f"dev:{i}", "exact_claim_group": "same-original",
                 "label": "Supported", "condition": c, "method": "raw_packets"}
                for i in range(2) for c in ("clean", "duplicate4")]
        outputs = {r["id"]: {"status": "ok", "response": "A" if (r["claim_id"] == "dev:0") == (r["condition"] == "duplicate4") else "B"} for r in refs}
        rows = fm.ev.outcomes(refs, outputs)
        result, details = fm.paired_readout(rows, rows, ("duplicate4", "raw_packets"), ("clean", "raw_packets"), replicates=20)
        self.assertEqual(result["signed_accuracy_change"]["difference"], 0)
        self.assertEqual(result["signed_accuracy_change"]["groups"], 1)
        self.assertEqual((result["by_realization"]["A"]["repairs"], result["by_realization"]["A"]["breakages"]), (1, 1))
        self.assertEqual(result["intervention"]["possible_pairs"], result["repeat"]["possible_pairs"])
        self.assertEqual(result["correctness_crossings_excess_over_repeat"]["difference"], 1)
        outputs["dev:0/duplicate4/raw_packets"] = {"status": "input_too_long", "response": None}
        failed = fm.ev.outcomes(refs, outputs)
        result, _ = fm.paired_readout(failed, rows, ("duplicate4", "raw_packets"), ("clean", "raw_packets"), replicates=20)
        self.assertEqual(result["paired_records"], 2)
        self.assertEqual(result["complete_four_valid_records"], 1)
        self.assertEqual(result["signed_accuracy_change"]["difference"], -.25)
        weighted = fm.group_interval([("g1", 1), ("g1", 1), ("g2", -1)], replicates=20)
        self.assertAlmostEqual(weighted["difference"], 1 / 3)
        self.assertEqual((weighted["records"], weighted["groups"]), (3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
