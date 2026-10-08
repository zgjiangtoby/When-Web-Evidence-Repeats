"""Compact CPU regressions; synthetic fixtures, no model generation or outcomes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unicodedata
import unittest
from unittest.mock import patch

import supplement as sp
from test_formal import source_record
from generate import read_requests, sha


class SyntheticTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "thinking": False, "enable_thinking": False}
        return "<chat>" + "".join(m["role"] + ":" + m["content"] for m in messages) + "<assistant>"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        # Sole filler ' .' adds exactly one synthetic token, so all targets are reachable.
        return list(range(len(text.replace(" ", ""))))


def fixture():
    av = [source_record(), source_record("Conflicting Evidence/Cherrypicking", "Single-unit claim", units=1)]
    docs = [{"doc_id": 11, "title": "Constructed title", "abstract": ["Marked sentence.", "Other sentence.", "Third sentence."]}]
    sci = [{"id": 3, "claim": "Constructed scientific claim", "evidence": {"11": [{"label": "SUPPORT", "sentences": [0]}]},
            "cited_doc_ids": [999999], "hidden_justification": "NEVER_VISIBLE_SCI_SENTINEL"}]
    return av, sci, docs


def artifacts():
    return sp.build_requests(sp.observations(*fixture()), SyntheticTokenizer(),
                             bound_function=lambda _, text: len(sp.regex.findall(r"\p{L}+", text)))


def predictions(requests, refs, lengths):
    gold = {r["id"]: "ABCD"[(sp.pr.LABELS if r["corpus"] == "averitec" else sp.SCI_LABELS).index(r["label"])] for r in refs}
    lens = {r["id"]: r["input_tokens"] for r in lengths}
    result, seen = {}, {}
    for request in requests:
        key, identity = sha(request["messages"]), request["id"]
        if key in seen:
            row = {**result[seen[key]], "id": identity, "cached": True, "cache_of": seen[key], "batch_size": 0,
                   "charged_input_tokens": 0, "charged_output_tokens": 0, "elapsed_batch_seconds": 0}
        else:
            long = lens[identity] > 8192
            row = {"id": identity, "status": "input_too_long" if long else "ok", "response": None if long else gold[identity],
                   "cached": False, "input_tokens": lens[identity], "output_tokens": 0 if long else 1,
                   "charged_input_tokens": 0 if long else lens[identity], "charged_output_tokens": 0 if long else 1,
                   "elapsed_batch_seconds": 0 if long else .01, "batch_size": 0 if long else 1}
            seen[key] = identity
        result[identity] = {**row, "metadata": {}, "prompt_sha256": key, "request_sha256": sha(request)}
    return result


def primary_rows():
    rows = []
    for i in range(4):
        for arm in sp.ARMS["scifact"]:
            correct = int(not arm.startswith("O-"))
            rows.append({"claim_id": str(i), "corpus": "scifact", "condition": arm, "resampling_group": "g" + str(i),
                         "correct": correct, "prediction": "SUPPORT" if correct else "CONTRADICT"})
    return rows


class SupplementTests(unittest.TestCase):
    def test_maximum_enumerates_nonmonotone_lengths_and_rejects_capped_search(self):
        def make(k, side):
            return {"id": "synthetic", "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": f"{k}:{side}"}]}
        def count(messages):
            return {1: 2, 2: 8, 3: 4, 4: 9, 5: 9}[int(messages[1]["content"].split(":")[0])]
        k, audit = sp.maximal_copies(make, 5, count, 1)
        self.assertEqual(k, 3)
        self.assertEqual([r[0] for r in audit["evaluated_k_front_back_tokens"]], [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "no truncated search"):
            sp.maximal_copies(make, 4097, count, 1)
        with self.assertRaisesRegex(ValueError, "No whole copy"):
            sp.maximal_copies(make, 1, lambda _: 2, 1)

    def test_bound_normalization_added_tokens_and_unsupported_regex(self):
        tokenizer_type = type("Qwen2Tokenizer", (), {})
        tokenizer = tokenizer_type()
        tokenizer.pat = sp.regex.compile(sp.PATTERNS["Qwen2Tokenizer"])
        tokenizer.byte_decoder = {}
        tokenizer.get_added_vocab = lambda: {"<|special_token|>": 100}
        tokenizer.prepare_for_tokenization = lambda text: (unicodedata.normalize("NFC", text), {})
        self.assertEqual(sp.certified_runs(tokenizer, "e\u0301x <|special_token|>"), 1)
        tokenizer.get_added_vocab = lambda: {"bad\nboundary": 1}
        with self.assertRaisesRegex(ValueError, "span a copy boundary"):
            sp.certified_runs(tokenizer, "Words here")
        tokenizer.pat = sp.regex.compile(".*")
        with self.assertRaisesRegex(ValueError, "Unsupported tokenizer"):
            sp.certified_runs(tokenizer, "Words here")

    def test_native_ids_duplicate_urls_aliases_and_gold_separation(self):
        av, sci, docs = fixture()
        duplicate = deepcopy(av[0]["questions"][0]["answers"][0])
        duplicate["source_url"] = "https://example.org/alternate"
        av[0]["questions"][0]["answers"].append(duplicate)
        docs[0]["abstract"] = ["Alias marked.", "Alias marked.", "Other sentence."]
        sci[0]["evidence"]["11"][0]["sentences"] = [1]
        obs = sp.observations(av, sci, docs)
        self.assertEqual(len(obs[0]["units"][0]["ids"]), 2)
        self.assertEqual(obs[0]["units"][0]["urls"], ["https://example.org/0", "https://example.org/alternate"])
        self.assertEqual(obs[-1]["targets"], {"R": 0, "O": 2})
        self.assertEqual(obs[-1]["target_aliases"]["R"], [0, 1])
        built = sp.build_requests(obs, SyntheticTokenizer(), lambda _, t: len(sp.regex.findall(r"\p{L}+", t)))
        requests = built[0]
        hidden = ("HIDDEN JUSTIFICATION SENTINEL", "NEVER_VISIBLE_SCI_SENTINEL", "999999", "marked_union", "rationale_sets")
        self.assertTrue(all(marker not in json.dumps(requests) for marker in hidden))
        self.assertTrue(all(r["metadata"] == {} and len(r["id"]) == 64 for r in requests))
        altered = deepcopy(sci)
        altered[0]["evidence"]["11"][0]["label"] = "CONTRADICT"
        av[0]["label"] = "Not Enough Evidence"
        rebuilt = sp.build_requests(sp.observations(av, altered, docs), SyntheticTokenizer(), lambda _, t: len(sp.regex.findall(r"\p{L}+", t)))
        self.assertEqual(requests, rebuilt[0])
        altered[0]["evidence"]["11"].append({"label": "CONTRADICT", "sentences": [999]})
        with self.assertRaisesRegex(ValueError, "indices"):
            sp.observations(av, altered, docs)

    def test_end_to_end_gold_free_file_and_single_content_identity(self):
        requests, refs, pads, copies, lengths = artifacts()
        self.assertEqual(len(requests), 20)
        by_id = {r["id"]: r for r in requests}
        one = {r["condition"]: by_id[r["id"]]["messages"] for r in refs if r["claim_id"] == "averitec:dev:1"}
        self.assertEqual(one["B"], one["S-front"])
        self.assertEqual(one["B"], one["S-back"])
        self.assertEqual(next(r["selected_k"] for r in copies if r["observation_id"] == "averitec:dev:1"), 1)
        self.assertTrue(all(p["exact_length_match"] and p["search_candidates"] <= 50 for p in pads))
        self.assertTrue(all(v["guard_pass"] for v in sp.coverage(refs, pads, lengths).values()))
        with tempfile.TemporaryDirectory(prefix="paper2_supplement_test_") as directory:
            path = Path(directory) / "synthetic_requests.jsonl"
            sp.fm.write_rows(path, requests)
            self.assertEqual(read_requests(path), requests)
        p = predictions(requests, refs, lengths)
        summary, ra, rb, paired, primary = sp.summarize(refs, p, deepcopy(p), pads, replicates=20)
        self.assertFalse(any(v["mechanical_screen_pass"] for v in summary["primary"].values()))
        self.assertEqual(summary["cells"]["scifact/C"]["A"]["per_class_support"]["NOINFO"], 0)
        self.assertEqual(len(primary), 2)
        self.assertTrue(paired)

    def test_noinfo_valid_fourth_scifact_letter_invalid_and_failures_retained(self):
        requests, refs, pads, _, lengths = artifacts()
        p = predictions(requests, refs, lengths)
        sci = [r for r in refs if r["corpus"] == "scifact"]
        p[sci[0]["id"]]["response"] = " C "
        p[sci[1]["id"]]["response"] = "D"
        rows = sp.native_outcomes(refs, p)
        index = {r["id"]: r for r in rows}
        self.assertEqual((index[sci[0]["id"]]["prediction"], index[sci[0]["id"]]["correct"]), ("NOINFO", 0))
        self.assertIsNone(index[sci[1]["id"]]["prediction"])
        selected = next(x for x in pads if x["id"] == sci[2]["id"])
        failed = [{**selected, "exact_length_match": False}]
        rows = sp.construction_status(rows, failed)
        changed = next(r for r in rows if r["id"] == selected["id"])
        self.assertEqual((changed["correct"], changed["prediction"], changed["output_status"]), (0, None, "construction_length_mismatch"))
        self.assertEqual(len(rows), len(refs))
        long = deepcopy(lengths)
        long[0]["input_tokens"], long[0]["over_input_ceiling"] = 9000, True
        guard = sp.coverage(refs, pads, long)
        self.assertEqual(guard["averitec"]["matched_nonlong_records"], 1)
        self.assertFalse(guard["averitec"]["guard_pass"])

    def test_exact_text_components_and_record_weighted_bootstrap(self):
        rows = [{"observation_id": "a", "corpus": "scifact", "exact_claim_group": "c1", "abstract_group": "d1"},
                {"observation_id": "b", "corpus": "scifact", "exact_claim_group": "c1", "abstract_group": "d2"},
                {"observation_id": "c", "corpus": "scifact", "exact_claim_group": "c2", "abstract_group": "d2"},
                {"observation_id": "d", "corpus": "scifact", "exact_claim_group": "c3", "abstract_group": "d3"}]
        sp.component_groups(rows)
        self.assertEqual(len({r["resampling_group"] for r in rows}), 2)
        weighted = sp.interval([("large", 1), ("large", 1), ("small", 0)], family=4, replicates=20)
        self.assertAlmostEqual(weighted["difference"], 2 / 3)
        self.assertEqual(weighted["family_size"], 4)

    def test_complete_role_panel_and_fixed_placement_screen(self):
        rows = primary_rows()
        good, _ = sp.primary_contrast(rows, deepcopy(rows), "R", replicates=20)
        self.assertTrue(good["mechanical_screen_pass"])
        bad = deepcopy(rows)
        next(r for r in bad if r["claim_id"] == "0" and r["condition"] == "B")["prediction"] = None
        checked, _ = sp.primary_contrast(rows, bad, "R", replicates=20)
        self.assertEqual(checked["common_valid_records"], 4)
        self.assertEqual(checked["common_ROB_valid_records"], 3)
        self.assertFalse(checked["mechanical_screen_pass"])
        for row in bad:
            if row["condition"] == "R-back":
                row["correct"] = 0
        for row in rows:
            if row["condition"] == "R-back":
                row["correct"] = 0
        checked, _ = sp.primary_contrast(rows, bad, "R", replicates=20)
        self.assertEqual(checked["positions"]["back"], 0)
        self.assertFalse(checked["mechanical_screen_pass"])

    def test_all_four_completion_gate_precedes_statistics(self):
        with tempfile.TemporaryDirectory(prefix="paper2_supplement_gate_") as directory:
            root = Path(directory)
            args = argparse.Namespace(output=root / "new", execution_receipt=root / "receipt.json")
            for model in sp.fm.MODELS:
                for suffix in ("prepared", "a", "b"):
                    setattr(args, model + "_" + suffix, root / (model + suffix))
            with patch.object(sp, "checked_repeats", side_effect=[("synthetic",), ValueError("incomplete second model")]), patch.object(sp, "summarize") as summary:
                with self.assertRaisesRegex(ValueError, "incomplete second model"):
                    sp.evaluate(args)
                summary.assert_not_called()
            args.granite_b = args.qwen_a
            with self.assertRaisesRegex(ValueError, "Four distinct"):
                sp.evaluate(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
