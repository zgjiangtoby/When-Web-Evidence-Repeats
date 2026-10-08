"""Constructed CPU checks; no source-population scoring, checkpoint load, or CUDA."""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import decoder_control as dc


class TinyTokenizer:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [" ABCD".index(c) for c in text]

    def decode(self, tokens, skip_special_tokens=False, **kwargs):
        return "".join(" ABCD"[t] for t in tokens if not (skip_special_tokens and t == 0))


def fixture():
    refs, requests = [], []
    for corpus, arms in dc.sp.ARMS.items():
        for arm in arms:
            identity = corpus + ":synthetic/" + arm
            refs.append({"id": identity, "claim_id": corpus + ":synthetic", "corpus": corpus, "condition": arm,
                         "method": "fixed_policy", "label": dc.sp.pr.LABELS[0] if corpus == "averitec" else "SUPPORT",
                         "exact_claim_group": corpus + ":exact", "resampling_group": corpus + ":component", "unit_count": 2})
            requests.append({"id": identity, "messages": [{"role": "user", "content": "Constructed " + identity}], "metadata": {}})
    return requests, refs


def fake_runs(refs):
    runs = {}
    for decoder in dc.DECODERS:
        for realization in dc.REALIZATIONS:
            rows = {}
            for ref in refs:
                response = "malformed" if decoder == "native" and ref["condition"].startswith("O-") else "A"
                rows[ref["id"]] = {"id": ref["id"], "response": response, "status": "ok", "input_tokens": 10,
                    "output_tokens": 2, "charged_input_tokens": 10, "charged_output_tokens": 2,
                    "batch_size": 1, "cached": False, "elapsed_batch_seconds": .1,
                    "first_step": {"shadow_differs": False, "raw_logit_shadow_argmax": "A"}}
            runs[(decoder, realization)] = (rows, {}, {})
    return runs


class DecoderControlTests(unittest.TestCase):
    def test_only_metadata_changes_and_labels_do_not_enter_requests(self):
        requests, refs = fixture()
        original = deepcopy(requests)
        prepared = dc.add_metadata(requests, refs)
        changed_refs = deepcopy(refs)
        for ref in changed_refs:
            ref["label"] = "HIDDEN_GOLD_SENTINEL"
            ref["rationale_sets"] = [[999]]
        self.assertEqual(prepared, dc.add_metadata(requests, changed_refs))
        self.assertEqual(requests, original)
        self.assertEqual([r["messages"] for r in prepared], [r["messages"] for r in original])
        self.assertEqual(prepared[-1]["metadata"]["allowed_labels"], "ABC")
        self.assertEqual(prepared[0]["metadata"]["allowed_labels"], "ABCD")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            dc.add_metadata(requests, refs[:-1])
        prepared[0]["metadata"]["label"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "Only corpus"):
            dc.closed.validate_requests(prepared)

    def test_single_token_roundtrip_eos_and_legal_completions(self):
        tokenizer = TinyTokenizer()
        contract = dc.closed.decision_contract(tokenizer, {"eos_token_id": [0], "repetition_penalty": 1.05})
        row = dc.add_metadata(*fixture())[-1]
        self.assertEqual(dc.closed.validate_completion([3, 0], row, contract, tokenizer), "C")
        with self.assertRaisesRegex(ValueError, "decision contract"):
            dc.closed.validate_completion([4, 0], row, contract, tokenizer)
        with self.assertRaisesRegex(ValueError, "decision contract"):
            dc.closed.validate_completion([1], row, contract, tokenizer)
        with patch.object(tokenizer, "encode", return_value=[1, 2]):
            with self.assertRaisesRegex(ValueError, "one round-tripping"):
                dc.closed.decision_contract(tokenizer, {"eos_token_id": 0})
        with patch.object(tokenizer, "decode", return_value=" A"):
            with self.assertRaisesRegex(ValueError, "one round-tripping"):
                dc.closed.decision_contract(tokenizer, {"eos_token_id": 0})

    def test_official_cpu_processor_mixed_alphabets_and_shadow_pairing(self):
        import torch
        from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
        prepared = dc.add_metadata(*fixture())
        rows = [prepared[0], prepared[-1]]
        contract = dc.closed.decision_contract(TinyTokenizer(), {"eos_token_id": 0})
        prefix = torch.tensor([[0, 0, 1, 3, 4], [0, 1, 2, 3, 4]], device="cpu")
        raw = torch.tensor([[100., 2.04, 2., 1., .5], [100., 1., 2., 3., 99.]], device="cpu")
        after_penalty = raw.clone()
        after_penalty[0, 1] /= 1.05
        processor = PrefixConstrainedLogitsProcessor(dc.closed.allowed_prefix(rows, 5, contract), num_beams=1)
        processed = processor(prefix, after_penalty)
        selected = processed.argmax(dim=-1)
        self.assertEqual(selected.tolist(), [2, 3])
        second_prefix = torch.cat([prefix, selected[:, None]], dim=1)
        self.assertEqual(processor(second_prefix, raw).argmax(dim=-1).tolist(), [0, 0])
        diagnostic = dc.closed.first_step_diagnostics(raw[0], processed[0], prefix[0].tolist(), rows[0], contract, 2)
        self.assertTrue(diagnostic["shadow_differs"])
        self.assertEqual(diagnostic["raw_logit_shadow_argmax"], "A")
        self.assertEqual(diagnostic["processed_argmax"], "B")
        self.assertFalse(diagnostic["candidates"]["B"]["token_occurs_in_prompt"])
        self.assertEqual(diagnostic["additional_forward_passes"], 0)

    def test_constructed_cpu_generation_returns_paired_raw_and_processed_steps(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, RepetitionPenaltyLogitsProcessor
        torch.manual_seed(17)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=16, n_embd=8, n_layer=1, n_head=1,
                                         bos_token_id=0, eos_token_id=0, pad_token_id=0)).eval()
        prepared = dc.add_metadata(*fixture())
        rows = [prepared[0], prepared[-1]]
        contract = dc.closed.decision_contract(TinyTokenizer(), {"eos_token_id": 0, "repetition_penalty": 1.05})
        prefix = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device="cpu")
        allowed = dc.closed.allowed_prefix(rows, 4, contract)
        with torch.inference_mode():
            output = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix), do_sample=False,
                                    max_new_tokens=16, repetition_penalty=1.05, prefix_allowed_tokens_fn=allowed,
                                    return_dict_in_generate=True, output_logits=True, output_scores=True)
        expected = PrefixConstrainedLogitsProcessor(allowed, 1)(prefix,
            RepetitionPenaltyLogitsProcessor(1.05)(prefix, output.logits[0].clone()))
        self.assertTrue(torch.equal(expected, output.scores[0]))
        self.assertEqual(len(output.logits), 2)
        self.assertEqual(len(output.scores), 2)
        for i, row in enumerate(rows):
            tokens = output.sequences[i, 4:].tolist()
            dc.closed.validate_completion(tokens, row, contract, TinyTokenizer())
            dc.closed.first_step_diagnostics(output.logits[0][i], output.scores[0][i], prefix[i].tolist(), row, contract, tokens[0])

    def test_full_population_interaction_invalid_zero_and_noinfo_support(self):
        _, refs = fixture()
        runs = fake_runs(refs)
        runs[("closed", "A")][0]["scifact:synthetic/O-front"]["first_step"].update(
            shadow_differs=True, raw_logit_shadow_argmax="B")
        report, outcomes, details = dc.summarize(refs, [], runs, replicates=40)
        primary = report["primary_interactions"]["R_minus_O"]
        self.assertEqual(primary["interaction_closed_minus_native"]["difference"], -1)
        self.assertEqual(primary["interaction_closed_minus_native"]["ci_familywise"], [-1, -1])
        self.assertTrue(primary["interaction_detected_two_sided"])
        self.assertEqual(primary["interaction_closed_minus_native"]["family_size"], 4)
        self.assertFalse(primary["common_panels"]["common_valid"]["coverage_pass_95pct"])
        self.assertFalse(primary["decoder_contrasts"]["closed"]["legacy_joint_screen_component_descriptive"])
        cell = report["cells"]["scifact/O-front"]
        self.assertEqual(cell["native"]["A"]["accuracy"], 0)
        self.assertEqual(cell["closed"]["A"]["per_class_support"]["NOINFO"], 0)
        self.assertEqual(cell["decoder_changes"]["A"]["repairs"], 1)
        self.assertEqual(cell["decoder_changes"]["A"]["format_failure_delta"], -1)
        shadow = report["first_forward_shadow_diagnostic"]["scifact/O-front/A"]
        self.assertEqual(shadow["raw_logit_shadow_correct"], 0)
        self.assertEqual(shadow["shadow_breakages_vs_official"], 1)
        self.assertEqual(shadow["shadow_net_correctness_vs_official"], -1)
        self.assertEqual(len(details), 3)
        self.assertEqual(len(outcomes[("native", "A")]), 14)

    def test_native_weighted_bootstrap_seed_and_overlength_retention(self):
        weighted = dc.group_interval([("shared", 1), ("shared", 1), ("singleton", 0)], family=4, replicates=40)
        self.assertAlmostEqual(weighted["difference"], 2 / 3)
        self.assertEqual(weighted["seed"], 20261002)
        self.assertEqual(weighted["groups"], 2)
        _, refs = fixture()
        runs = fake_runs(refs)
        for predictions, _, _ in runs.values():
            row = predictions[refs[0]["id"]]
            row.update(status="input_too_long", response=None, input_tokens=9000, output_tokens=0,
                       charged_input_tokens=0, charged_output_tokens=0, batch_size=0, elapsed_batch_seconds=0)
        report, _, _ = dc.summarize(refs, [], runs, replicates=20)
        self.assertEqual(report["cells"]["averitec/C"]["closed"]["A"]["status_counts"], {"input_too_long": 1})
        self.assertEqual(report["cells"]["averitec/C"]["closed"]["A"]["accuracy"], 0)

    def test_receipt_pins_interpreter_and_rejects_reversed_additional_attempt(self):
        prepared, paths, bundles, entries = {}, {}, {}, []
        for model, (directory, revision) in dc.fm.MODELS.items():
            prepared[model] = Path("/synthetic/prepared") / model
            definition = {"path": "/synthetic/models/" + directory, "revision": revision}
            paths[model], runs = {}, {}
            for decoder in dc.DECODERS:
                for realization, hour in zip(dc.REALIZATIONS, (10, 11)):
                    key = decoder, realization
                    output = Path("/synthetic/runs") / model / f"{decoder}_{realization}.jsonl"
                    begin, end = f"2026-10-02T{hour}:00:00Z", f"2026-10-02T{hour}:01:00Z"
                    paths[model][key] = output
                    runs[key] = ({}, {}, {"start_utc": begin, "end_utc": end})
                    entries.append({"model_key": model, "decoder": decoder, "realization": realization,
                        "visible_gpu_count": 1, "output_existed_before_launch": False, "resume": False,
                        "returncode": 0, "output_path": str(output), "output_sha256": "fixture",
                        "manifest_sha256": "fixture", "launch_utc": begin, "finish_utc": end,
                        "argv": [dc.FROZEN_PYTHON] + dc.expected_argv(prepared[model], output, definition, decoder)})
            bundles[model] = ({"model": definition}, None, None, runs)
        receipt = {"runs": entries, "additional_attempts": []}
        with patch.object(dc.fm, "file_sha", return_value="fixture"):
            self.assertEqual(dc.check_receipt(receipt, bundles, paths, prepared), 480)
            for executable in ("", "/usr/bin/python"):
                invalid = deepcopy(receipt)
                invalid["runs"][0]["argv"][0] = executable
                with self.assertRaisesRegex(ValueError, "frozen argv"):
                    dc.check_receipt(invalid, bundles, paths, prepared)
            invalid = deepcopy(receipt)
            invalid["additional_attempts"] = [{"argv": entries[0]["argv"], "launch_utc": "2026-10-02T09:01:00Z",
                "finish_utc": "2026-10-02T09:00:00Z", "returncode": 1, "log_path": "/synthetic/failed.log", "reason": "fixture"}]
            with self.assertRaisesRegex(ValueError, "finish precedes launch"):
                dc.check_receipt(invalid, bundles, paths, prepared)

    def test_completion_gate_precedes_any_statistics(self):
        with tempfile.TemporaryDirectory(prefix="decoder_control_cpu_") as temporary:
            root = Path(temporary)
            args = argparse.Namespace(output=root / "new", execution_receipt=root / "receipt.json")
            for model in dc.fm.MODELS:
                setattr(args, model + "_prepared", root / (model + "_prepared"))
                for decoder in dc.DECODERS:
                    for realization in dc.REALIZATIONS:
                        setattr(args, f"{model}_{decoder}_{realization.lower()}", root / f"{model}_{decoder}_{realization}")
            with patch.object(dc, "checked_model", side_effect=[("fixture",), ValueError("incomplete model")]), \
                    patch.object(dc, "summarize") as scorer:
                with self.assertRaisesRegex(ValueError, "incomplete model"):
                    dc.evaluate(args)
                scorer.assert_not_called()
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
