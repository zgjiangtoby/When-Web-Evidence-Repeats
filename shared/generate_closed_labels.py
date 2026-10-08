#!/usr/bin/env python3
"""Fresh offline greedy runs constrained to a bare label token, then EOS."""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from generate import atomic_json, load_local_tokenizer, read_requests, sha

ALPHABETS = {"averitec": "ABCD", "scifact": "ABC"}
NATIVE_RUNNER_SHA = "f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c"


def validate_requests(rows):
    seen = {}
    for row in rows:
        metadata = row.get("metadata", {})
        if (set(metadata) != {"corpus", "condition", "allowed_labels"}
                or metadata.get("corpus") not in ALPHABETS
                or metadata["allowed_labels"] != ALPHABETS[metadata["corpus"]]
                or not isinstance(metadata["condition"], str)):
            raise ValueError("Only corpus, condition and the fixed corpus alphabet may enter decoder metadata")
        key, alphabet = sha(row["messages"]), metadata["allowed_labels"]
        if key in seen and seen[key] != alphabet:
            raise ValueError("Identical messages cannot have different decoder alphabets")
        seen[key] = alphabet


def decision_contract(tokenizer, generation_config):
    tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if (len(ids) != 1 or ids[0] in tokenizer.all_special_ids
                or tokenizer.decode(ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) != letter):
            raise ValueError(f"Bare {letter!r} is not one round-tripping nonspecial token")
        tokens[letter] = ids[0]
    eos = generation_config.get("eos_token_id")
    eos_ids = eos if isinstance(eos, list) else [eos]
    if (len(set(tokens.values())) != 4 or not eos_ids
            or any(type(i) is not int or i < 0 for i in eos_ids)
            or set(tokens.values()).intersection(eos_ids)):
        raise ValueError("Ambiguous label tokens or missing/disjoint EOS contract")
    if generation_config.get("num_beams", 1) != 1:
        raise ValueError("The frozen experiment requires greedy single-beam generation")
    return {"name": "bare_single_label_then_first_model_eos_v1", "label_token_ids": tokens,
            "alphabets": ALPHABETS, "forced_eos_token_id": eos_ids[0], "model_eos_token_ids": eos_ids,
            "selection": "greedy legal-label argmax with all inherited native generation processors",
            "first_step": "one bare corpus-legal letter token; no whitespace, explanation, or EOS",
            "second_step": "first EOS declared by the unchanged model generation config",
            "inherited_generation_config": generation_config,
            "max_new_tokens_is_ceiling": 16, "posthoc_repair": False,
            "shadow_decision": "argmax of the same first-forward raw logits over legal labels; diagnostic only"}


def allowed_prefix(rows, prompt_width, contract):
    labels = [[contract["label_token_ids"][letter] for letter in row["metadata"]["allowed_labels"]]
              for row in rows]

    def allowed(batch_id, input_ids):
        generated = len(input_ids) - prompt_width
        if generated < 0:
            raise ValueError("Generation prefix shorter than the left-padded input width")
        return labels[batch_id] if generated == 0 else [contract["forced_eos_token_id"]]
    return allowed


def validate_completion(tokens, row, contract, tokenizer):
    legal = {contract["label_token_ids"][letter] for letter in row["metadata"]["allowed_labels"]}
    if len(tokens) != 2 or tokens[0] not in legal or tokens[1] != contract["forced_eos_token_id"]:
        raise ValueError("Generation violated the first-letter/second-EOS decision contract")
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    if response not in row["metadata"]["allowed_labels"] or len(response) != 1:
        raise ValueError("Raw decoded completion is not exactly one corpus-legal letter")
    return response


def first_step_diagnostics(raw, processed, prompt_ids, row, contract, generated_token):
    candidates = {letter: contract["label_token_ids"][letter] for letter in row["metadata"]["allowed_labels"]}
    scores = {letter: {"token_id": token, "raw_logit": float(raw[token]),
                       "processed_score": float(processed[token]), "token_occurs_in_prompt": token in prompt_ids}
              for letter, token in candidates.items()}
    if any(not math.isfinite(value[k]) for value in scores.values() for k in ("raw_logit", "processed_score")):
        raise ValueError("Nonfinite legal-label score")
    # torch.argmax breaks ties by the lowest vocabulary ID, not alphabet order.
    choose = lambda field: min(scores, key=lambda label: (-scores[label][field], candidates[label]))
    official, shadow = choose("processed_score"), choose("raw_logit")
    if candidates[official] != generated_token:
        raise ValueError("Generated label differs from recorded processed-score argmax")
    return {"candidates": scores, "processed_argmax": official, "raw_logit_shadow_argmax": shadow,
            "shadow_differs": official != shadow, "additional_forward_passes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-revision")
    for name, default in (("batch-size", 8), ("max-input-tokens", 8192), ("max-new-tokens", 16),
                          ("shard-index", 0), ("num-shards", 1), ("seed", 17)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--check-input", action="store_true")
    args = parser.parse_args()
    rows = read_requests(args.input)
    validate_requests(rows)
    if (args.batch_size, args.max_input_tokens, args.max_new_tokens,
            args.shard_index, args.num_shards, args.seed) != (8, 8192, 16, 0, 1, 17):
        parser.error("Frozen decoder control requires batch8/input8192/output16/shard0of1/seed17")
    if args.check_input:
        print(json.dumps({"requests": len(rows), "unique_prompts": len({sha(r['messages']) for r in rows})}))
        return
    if not args.output or not args.model or not args.model.is_dir() or not args.model_revision:
        parser.error("Local --model, --model-revision and --output required")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists() or args.output.resolve() == args.input.resolve():
        parser.error("Use a fresh output and manifest; no resume or cross-run caching")
    native = Path(__file__).with_name("generate.py")
    if hashlib.sha256(native.read_bytes()).hexdigest() != NATIVE_RUNNER_SHA:
        raise ValueError("Frozen native generator changed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done, cache = {}, {}
    manifest = {"status": "RUNNING", "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "process_id": os.getpid(), "total_requests": len(rows), "completed": 0,
                "stage": "initialization", "attempted_batch_ids": []}
    atomic_json(manifest_path, manifest)

    def mark_interrupted():
        if manifest["status"] == "RUNNING":
            manifest.update(status="INCOMPLETE", completed=len(done),
                            end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            atomic_json(manifest_path, manifest)
    atexit.register(mark_interrupted)
    try:
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        if not torch.cuda.is_available():
            raise RuntimeError("A visible CUDA device is required; use approved device access")
        torch.set_num_threads(4)
        set_seed(args.seed)
        tokenizer = load_local_tokenizer(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generation_config = json.loads((args.model / "generation_config.json").read_text())
        contract = decision_contract(tokenizer, generation_config)
        config = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "native_runner_sha256": NATIVE_RUNNER_SHA, "model_path": str(args.model.resolve()),
                  "model_revision": args.model_revision,
                  "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                  "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers")},
                  "tokenizer_use_fast": False, "dtype": "bfloat16", "attention": "sdpa", "do_sample": False,
                  "thinking": False, "batch_size": 8, "max_new_tokens": 16, "max_input_tokens": 8192,
                  "seed": 17, "shard_index": 0, "num_shards": 1, "decision_contract": contract}
        signature = sha(config)
        manifest.update(config=config, run_signature=signature, device=torch.cuda.get_device_name(0), stage="model_load")
        atomic_json(manifest_path, manifest)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                   use_safetensors=True, dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to("cuda").eval()
        actual_eos = model.generation_config.eos_token_id
        if (actual_eos if isinstance(actual_eos, list) else [actual_eos]) != contract["model_eos_token_ids"]:
            raise ValueError("Loaded model EOS differs from recorded generation config")
        print(json.dumps({"event": "loaded", "signature": signature, "decision_contract": contract}), flush=True)
        with args.output.open("x", buffering=1, encoding="utf-8") as stream:
            def emit(row, result):
                record = {**result, "id": row["id"], "request_sha256": sha(row),
                          "prompt_sha256": sha(row["messages"]), "run_signature": signature,
                          "metadata": row["metadata"]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[row["id"]] = record
                cache[record["prompt_sha256"]] = record

            for start in range(0, len(rows), 8):
                group = rows[start:start + 8]
                unique, rendered, delayed, in_batch = [], [], [], set()
                manifest.update(stage="generation", attempted_batch_ids=[r["id"] for r in group])
                atomic_json(manifest_path, manifest)
                for row in group:
                    key = sha(row["messages"])
                    if key in cache or key in in_batch:
                        delayed.append(row)
                        continue
                    text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True,
                                                         thinking=False, enable_thinking=False)
                    count = len(tokenizer.encode(text, add_special_tokens=False))
                    if count > 8192:
                        emit(row, {"status": "input_too_long", "response": None, "input_tokens": count,
                                   "output_tokens": 0, "charged_input_tokens": 0, "charged_output_tokens": 0,
                                   "elapsed_batch_seconds": 0, "batch_size": 0, "cached": False,
                                   "generated_token_ids": []})
                        continue
                    in_batch.add(key)
                    unique.append(row)
                    rendered.append(text)
                if unique:
                    encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                        add_special_tokens=False, truncation=False).to("cuda")
                    width = encoded["input_ids"].shape[1]
                    torch.cuda.synchronize()
                    began = time.monotonic()
                    with torch.inference_mode():
                        output = model.generate(**encoded, do_sample=False, max_new_tokens=16,
                                                pad_token_id=tokenizer.pad_token_id,
                                                prefix_allowed_tokens_fn=allowed_prefix(unique, width, contract),
                                                return_dict_in_generate=True, output_logits=True, output_scores=True)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - began
                    for index, row in enumerate(unique):
                        tokens = output.sequences[index, width:].tolist()
                        response = validate_completion(tokens, row, contract, tokenizer)
                        diagnostics = first_step_diagnostics(output.logits[0][index], output.scores[0][index],
                                                             encoded["input_ids"][index].tolist(), row, contract, tokens[0])
                        length = int(encoded["attention_mask"][index].sum().item())
                        emit(row, {"status": "ok", "response": response, "input_tokens": length, "output_tokens": len(tokens),
                                   "charged_input_tokens": length, "charged_output_tokens": len(tokens),
                                   "elapsed_batch_seconds": elapsed, "batch_size": len(unique), "cached": False,
                                   "generated_token_ids": tokens, "first_step": diagnostics})
                for row in delayed:
                    source = cache[sha(row["messages"])]
                    emit(row, {**source, "cache_of": source.get("cache_of", source["id"]), "cached": True,
                               "charged_input_tokens": 0, "charged_output_tokens": 0,
                               "elapsed_batch_seconds": 0, "batch_size": 0})
                manifest["completed"] = len(done)
                atomic_json(manifest_path, manifest)
                print(json.dumps({"event": "progress", "completed": len(done), "total": len(rows)}), flush=True)
        manifest.update(status="COMPLETED", completed=len(done), stage="finished", attempted_batch_ids=[],
                        end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="FAILED", completed=len(done), error_type=type(error).__name__, error=str(error),
                        end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(manifest_path, manifest)
        raise
    finally:
        atexit.unregister(mark_interrupted)


if __name__ == "__main__":
    main()
