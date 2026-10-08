"""Synthetic wrapper tests: no data, GPU, model, or real inference subprocess."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "run_with_receipt.py"
SPEC = importlib.util.spec_from_file_location("release_receipt", SOURCE)
receipt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt)


class ReceiptTests(unittest.TestCase):
    def args(self, root, **changes):
        values = dict(kind="decoder", model_key="qwen", decoder="native", realization="A",
                      prepared=root / "prepared", output=root / "result.jsonl",
                      receipt=root / "cell.json", log=root / "child.log", gpu="0",
                      expandable_allocator=False, execute=True)
        values.update(changes)
        return argparse.Namespace(**values)

    def synthetic_cells(self, root, kind):
        """Minimal receipt fixtures, not dataset records or model outputs."""
        sys.path[:0] = [str(SOURCE.parents[1] / "paper2"), str(SOURCE.parents[1] / "shared")]
        import formal
        import supplement
        import decoder_control
        decoders = ("native",) if kind == "supplement" else ("native", "closed")
        receipts, prepared, paths, bundles = [], {}, {}, {}
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for model_key in ("qwen", "granite"):
            prepared[model_key] = root / model_key
            prepared[model_key].mkdir()
            directory, revision = formal.MODELS[model_key]
            model = {"key": model_key, "revision": revision,
                     "path": str((SOURCE.parents[1] / "models" / directory).resolve())}
            preparation = {"model": model, "status": "PREPARED_NOT_RUN",
                           "version": supplement.VERSION if kind == "supplement" else decoder_control.VERSION}
            (prepared[model_key] / "preparation.json").write_text(json.dumps(preparation))
            (prepared[model_key] / "requests.jsonl").write_text('{"synthetic": true}\n')
            paths[model_key] = [] if kind == "supplement" else {}
            generated = [] if kind == "supplement" else {}
            for decoder in decoders:
                for realization in ("A", "B"):
                    name = model_key + "_" + decoder + "_" + realization
                    output = root / (name + ".jsonl")
                    output.write_text('{"synthetic": true}\n')
                    begin = origin + timedelta(seconds=10 * len(receipts))
                    end = begin + timedelta(seconds=1)
                    manifest = {"start_utc": begin.isoformat(timespec="seconds"),
                                "end_utc": end.isoformat(timespec="seconds")}
                    output.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
                    args = self.args(root, kind=kind, model_key=model_key, decoder=decoder,
                                     realization=realization, prepared=prepared[model_key], output=output)
                    entry = {"model_key": model_key, "realization": realization,
                             "argv": receipt.command(args), "output_path": str(output),
                             "output_existed_before_launch": False, "resume": False,
                             "visible_gpu_count": 1, "returncode": 0,
                             "launch_utc": manifest["start_utc"], "finish_utc": manifest["end_utc"],
                             "output_sha256": receipt.digest(output),
                             "manifest_sha256": receipt.digest(output.with_suffix(".jsonl.manifest.json"))}
                    if kind == "decoder":
                        entry["decoder"] = decoder
                        paths[model_key][decoder, realization] = output
                        generated[decoder, realization] = (None, None, manifest)
                    else:
                        paths[model_key].append(output)
                        generated.append((None, None, manifest))
                    part = root / (name + "_receipt.json")
                    part.write_text(json.dumps({"kind": kind, "runs": [entry], "additional_attempts": []}))
                    receipts.append(part)
            bundles[model_key] = ((preparation, None, generated) if kind == "supplement"
                                 else (preparation, None, None, generated))
        return receipts, prepared, paths, bundles

    def test_dry_run_writes_nothing_and_launches_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root, execute=False)
            with mock.patch.object(receipt, "command", return_value=["synthetic-child"]), mock.patch.object(receipt.subprocess, "run") as run:
                self.assertEqual(receipt.capture(args), 0)
            run.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_capture_success_binds_actual_generated_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            def synthetic_child(argv, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                args.output.write_text('{"synthetic": true}\n')
                args.output.with_suffix(".jsonl.manifest.json").write_text('{"synthetic": true}\n')
                return subprocess.CompletedProcess(argv, 0)
            with mock.patch.object(receipt, "command", return_value=["synthetic-child"]), mock.patch.object(receipt.subprocess, "run", side_effect=synthetic_child), mock.patch.dict(receipt.os.environ, {}, clear=True):
                self.assertEqual(receipt.capture(args), 0)
            report = json.loads(args.receipt.read_text())
            self.assertEqual(report["additional_attempts"], [])
            self.assertEqual(report["runs"][0]["output_sha256"], receipt.digest(args.output))
            self.assertEqual(report["runs"][0]["returncode"], 0)

    def test_failure_is_preserved_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            with mock.patch.object(receipt, "command", return_value=["synthetic-child"]), mock.patch.object(receipt.subprocess, "run", return_value=subprocess.CompletedProcess(["synthetic-child"], 9)) as run, mock.patch.dict(receipt.os.environ, {}, clear=True):
                self.assertEqual(receipt.capture(args), 9)
            run.assert_called_once()
            report = json.loads(args.receipt.read_text())
            self.assertEqual(report["runs"], [])
            self.assertEqual(report["additional_attempts"][0]["returncode"], 9)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.output.write_text("synthetic existing output")
            with mock.patch.object(receipt, "command", return_value=["synthetic-child"]), mock.patch.object(receipt.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    receipt.capture(args)
            run.assert_not_called()
            self.assertEqual(args.output.read_text(), "synthetic existing output")

    def test_multiple_device_selection_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                receipt.capture(self.args(Path(temp), gpu="0,1"))

    def test_allocator_not_applied_to_other_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                receipt.capture(self.args(Path(temp), expandable_allocator=True))

    def test_timestamps_match_frozen_manifest_precision(self):
        self.assertEqual(datetime.fromisoformat(receipt.utc_now()).microsecond, 0)

    def test_unpassed_preparation_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, prepared, _, _ = self.synthetic_cells(root, "supplement")
            manifest_path = prepared["qwen"] / "preparation.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "STOP_PREPARATION_GUARD"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                receipt.command(self.args(root, kind="supplement", prepared=prepared["qwen"]))

    def test_symlink_venv_cannot_silently_lose_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, prepared, _, _ = self.synthetic_cells(root, "supplement")
            with mock.patch.object(receipt.sys, "prefix", str(root / "synthetic_venv")), mock.patch.object(receipt.sys, "base_prefix", str(root / "synthetic_base")):
                with self.assertRaisesRegex(ValueError, "symlink-based venv"):
                    receipt.command(self.args(root, kind="supplement", prepared=prepared["qwen"]))

    def test_merge_requires_complete_cell_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            part = root / "synthetic.json"
            part.write_text(json.dumps({"kind": "decoder", "runs": [], "additional_attempts": []}))
            with self.assertRaises(ValueError):
                receipt.merge(argparse.Namespace(kind="decoder", receipts=[part], output=root / "merged.json"))
            self.assertFalse((root / "merged.json").exists())

    def test_supplement_merged_receipt_passes_unchanged_frozen_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts, prepared, paths, bundles = self.synthetic_cells(root, "supplement")
            output = root / "merged.json"
            self.assertEqual(receipt.merge(argparse.Namespace(kind="supplement", receipts=receipts, output=output)), 0)
            import supplement
            self.assertEqual(supplement.check_receipt(json.loads(output.read_text()), bundles, paths, prepared), 4)

    def test_decoder_merged_receipt_passes_unchanged_frozen_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts, prepared, paths, bundles = self.synthetic_cells(root, "decoder")
            output = root / "merged.json"
            self.assertEqual(receipt.merge(argparse.Namespace(kind="decoder", receipts=receipts, output=output)), 0)
            import decoder_control
            self.assertEqual(decoder_control.check_receipt(json.loads(output.read_text()), bundles, paths, prepared), 8)

    def test_merge_rejects_tampered_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts, _, paths, _ = self.synthetic_cells(root, "supplement")
            paths["qwen"][0].write_text("changed synthetic fixture")
            with self.assertRaises(ValueError):
                receipt.merge(argparse.Namespace(kind="supplement", receipts=receipts, output=root / "merged.json"))

    def test_merge_rejects_b_starting_before_a_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts, _, _, _ = self.synthetic_cells(root, "decoder")
            value = json.loads(receipts[1].read_text())
            value["runs"][0]["launch_utc"] = "2026-01-01T00:00:00+00:00"
            receipts[1].write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                receipt.merge(argparse.Namespace(kind="decoder", receipts=receipts, output=root / "merged.json"))

    def test_supplement_does_not_hide_additional_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts, _, _, _ = self.synthetic_cells(root, "supplement")
            value = json.loads(receipts[0].read_text())
            attempt = dict(value["runs"][0], returncode=9, log_path="synthetic.log", reason="synthetic failure")
            value["additional_attempts"] = [attempt]
            receipts[0].write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                receipt.merge(argparse.Namespace(kind="supplement", receipts=receipts, output=root / "merged.json"))


if __name__ == "__main__":
    unittest.main()
