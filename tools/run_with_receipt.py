#!/usr/bin/env python3
"""Explicitly launch one frozen run and capture its actual execution receipt.

Capture is a dry run unless --execute is supplied. No inference occurs on import.
One receipt per child permits parallel runs without a shared mutable receipt.
The frozen evaluators remain authoritative for data and scientific validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    # Frozen generator manifests use whole seconds. Mixing microsecond launch
    # times with their floored start time can reject a valid same-second launch.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def exists(path):
    return path.exists() or path.is_symlink()


def command(args):
    # Import only local task definitions; model/data loading is deferred to the
    # explicitly executed, unchanged generator and subsequent frozen evaluator.
    sys.path[:0] = [str(ROOT / "paper2"), str(ROOT / "shared")]
    import formal
    import supplement
    import decoder_control
    interpreter = Path(sys.executable).resolve()
    if sys.prefix != sys.base_prefix and not interpreter.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Resolved Python leaves this symlink-based venv; use a Conda environment or venv --copies for the frozen receipt contract")
    if args.kind == "supplement" and args.decoder != "native":
        raise ValueError("The supplement uses only the native decoder")
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / "preparation.json").read_text())
    model = manifest["model"]
    directory, revision = formal.MODELS[args.model_key]
    if (model["key"] != args.model_key or model["revision"] != revision
            or Path(model["path"]).resolve() != (ROOT / "models" / directory).resolve()):
        raise ValueError("Prepared model identity differs from the frozen model")
    expected_version = supplement.VERSION if args.kind == "supplement" else decoder_control.VERSION
    if manifest["version"] != expected_version:
        raise ValueError("Preparation belongs to a different measurement")
    if manifest.get("status") != "PREPARED_NOT_RUN":
        raise ValueError("Preparation did not pass its frozen guards")
    if not (prepared / "requests.jsonl").is_file():
        raise ValueError("Prepared requests are missing")
    suffix = supplement.expected_argv(prepared, args.output.resolve(), model)
    if args.decoder == "closed":
        suffix[0] = str(ROOT / "shared" / "generate_closed_labels.py")
    return [str(interpreter)] + suffix


def capture(args):
    if not args.gpu.strip() or "," in args.gpu or args.gpu.strip() == "-1":
        raise ValueError("Select exactly one available CUDA device or GPU UUID")
    if args.expandable_allocator and not (args.model_key == "granite" and args.decoder == "closed"):
        raise ValueError("Expandable allocator is restricted to Granite closed runs")
    argv = command(args)
    if not args.execute:
        print(json.dumps({"execute": False, "argv": argv,
                          "CUDA_VISIBLE_DEVICES": args.gpu,
                          "note": "No child launched; add --execute only after checking GPU availability."}, indent=2))
        return 0
    output, receipt, log = (p.resolve() for p in (args.output, args.receipt, args.log))
    output_manifest = output.with_suffix(output.suffix + ".manifest.json")
    destinations = [output, output_manifest, receipt, log]
    if len(set(destinations)) != len(destinations) or any(exists(p) for p in destinations):
        raise FileExistsError("Output, manifest, receipt, and log must be distinct unused paths")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.expandable_allocator:
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    elif environment.get("PYTORCH_CUDA_ALLOC_CONF") or environment.get("PYTORCH_ALLOC_CONF"):
        raise ValueError("Unset allocator overrides for the default-allocator frozen runs")
    if environment.get("PYTORCH_ALLOC_CONF"):
        raise ValueError("Unset PYTORCH_ALLOC_CONF; use only the documented allocator option")
    # Exclusive creation prevents silently overwriting an earlier receipt/log.
    with receipt.open("x", encoding="utf-8") as receipt_stream, log.open("x", encoding="utf-8") as log_stream:
        begin = utc_now()
        launch_error = None
        try:
            completed = subprocess.run(argv, cwd=ROOT, env=environment,
                                       stdout=log_stream, stderr=subprocess.STDOUT, check=False)
            returncode = completed.returncode
        except OSError as error:
            returncode, launch_error = 127, str(error)
        except KeyboardInterrupt:
            returncode, launch_error = 130, "Operator interrupted the child process"
        end = utc_now()
        entry = {"model_key": args.model_key, "realization": args.realization,
                 "argv": argv, "output_path": str(output),
                 "output_existed_before_launch": False, "resume": False,
                 "visible_gpu_count": 1, "launch_utc": begin, "finish_utc": end,
                 "returncode": returncode, "log_path": str(log),
                 "cuda_visible_devices": args.gpu,
                 "allocator_override": environment.get("PYTORCH_CUDA_ALLOC_CONF")}
        if args.kind == "decoder":
            entry["decoder"] = args.decoder
        complete = returncode == 0 and output.is_file() and output_manifest.is_file()
        if complete:
            entry.update(output_sha256=digest(output), manifest_sha256=digest(output_manifest))
            result = {"kind": args.kind, "runs": [entry], "additional_attempts": []}
        else:
            entry["reason"] = launch_error or "Child failed or did not produce both output and generation manifest; inspect the log"
            result = {"kind": args.kind, "runs": [], "additional_attempts": [entry]}
        json.dump(result, receipt_stream, indent=2, sort_keys=True)
        receipt_stream.write("\n")
    print(json.dumps({"receipt": str(receipt), "child_returncode": returncode,
                      "output_and_manifest_present": complete}))
    return 0 if complete else (returncode if 0 < returncode < 256 else 1)


def merge(args):
    if exists(args.output):
        raise FileExistsError("Use a new merged receipt path")
    if len({p.resolve() for p in args.receipts}) != len(args.receipts):
        raise ValueError("Do not supply a cell receipt more than once")
    runs, attempts, inputs = [], [], []
    for path in args.receipts:
        value = json.loads(path.read_text())
        if value.get("kind") != args.kind or not isinstance(value.get("runs"), list) or not isinstance(value.get("additional_attempts"), list):
            raise ValueError("Receipt kind or lists are missing/inconsistent")
        runs.extend(value["runs"])
        attempts.extend(value["additional_attempts"])
        inputs.append({"path": str(path.resolve()), "sha256": digest(path)})
    keys = [(r["model_key"], r.get("decoder", "native"), r["realization"]) for r in runs]
    decoders = ("native",) if args.kind == "supplement" else ("native", "closed")
    required = {(m, d, r) for m in ("qwen", "granite") for d in decoders for r in ("A", "B")}
    if len(keys) != len(required) or set(keys) != required:
        raise ValueError("Require exactly the full distinct model/decoder/A-B cell set")
    by_key = dict(zip(keys, runs))
    for attempt in attempts:
        if any(field not in attempt for field in ("argv", "launch_utc", "finish_utc", "returncode", "log_path", "reason")):
            raise ValueError("Additional attempts need complete actual execution records")
    for entry in runs:
        path = Path(entry["output_path"])
        if (entry["returncode"] != 0 or entry["output_existed_before_launch"] is not False
                or entry["resume"] is not False or entry["visible_gpu_count"] != 1
                or digest(path) != entry["output_sha256"]
                or digest(path.with_suffix(path.suffix + ".manifest.json")) != entry["manifest_sha256"]):
            raise ValueError("Successful receipt fields or output bindings changed")
    for model in ("qwen", "granite"):
        for decoder in decoders:
            a, b = by_key[model, decoder, "A"], by_key[model, decoder, "B"]
            if datetime.fromisoformat(a["finish_utc"]) > datetime.fromisoformat(b["launch_utc"]):
                raise ValueError("A must finish before the corresponding B starts")
    for entry in runs + attempts:
        begin, end = (datetime.fromisoformat(entry[k]) for k in ("launch_utc", "finish_utc"))
        if begin.tzinfo is None or end.tzinfo is None or end < begin:
            raise ValueError("Invalid execution interval")
    if args.kind == "supplement":
        occupied = sum((datetime.fromisoformat(r["finish_utc"]) - datetime.fromisoformat(r["launch_utc"])).total_seconds() for r in runs)
        if attempts or occupied > 14400:
            raise ValueError("Frozen supplement rejects additional attempts or over-budget execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"kind": args.kind, "runs": runs, "additional_attempts": attempts,
                   "cell_receipt_inputs": inputs}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"receipt": str(args.output), "runs": len(runs), "additional_attempts": len(attempts)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("capture", help="Show or explicitly execute one frozen generator invocation")
    one.add_argument("--kind", choices=("supplement", "decoder"), required=True)
    one.add_argument("--model-key", choices=("qwen", "granite"), required=True)
    one.add_argument("--decoder", choices=("native", "closed"), default="native")
    one.add_argument("--realization", choices=("A", "B"), required=True)
    for name in ("prepared", "output", "receipt", "log"):
        one.add_argument("--" + name, type=Path, required=True)
    one.add_argument("--gpu", required=True, help="Exactly one operator-selected available CUDA device/UUID")
    one.add_argument("--expandable-allocator", action="store_true")
    one.add_argument("--execute", action="store_true", help="Explicitly authorize this local model-inference invocation")
    two = sub.add_parser("merge", help="Merge fresh cell receipts; does not execute a model")
    two.add_argument("--kind", choices=("supplement", "decoder"), required=True)
    two.add_argument("--receipts", type=Path, nargs="+", required=True)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return capture(args) if args.action == "capture" else merge(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, "error: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
