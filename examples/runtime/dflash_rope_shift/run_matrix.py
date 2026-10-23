"""Run real image/text DFlash RoPE ablations against a fresh local server per arm."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import mimetypes
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_KEYS = (
    "SGLANG_DFLASH_PROMPT_IMAGE_SHIFT",
    "SGLANG_DFLASH_PROMPT_TEXT_SHIFT",
    "SGLANG_DFLASH_PROMPT_TEXT_KEEP_RECENT",
)


def arms(
    image_shift,
    text_shift,
    target_only,
    image_shift_sweep=None,
    image_kv_diagnostic=False,
    modality_kv_diagnostic=False,
):
    if image_kv_diagnostic or modality_kv_diagnostic:
        if image_shift_sweep is not None:
            raise ValueError(
                "Choose image KV diagnostic or image shift sweep, not both"
            )
        result = [
            ("baseline", 0, 0),
            ("image_nearest", "nearest", 0),
            ("image_mask", 0, 0),
        ]
        if modality_kv_diagnostic:
            result += [
                ("text_nearest", 0, "nearest"),
                ("text_mask", 0, 0),
                ("both_nearest", "nearest", "nearest"),
                ("both_mask", 0, 0),
            ]
    elif image_shift_sweep is None:
        result = [
            ("baseline", 0, 0),
            ("image", image_shift, 0),
            ("text", 0, text_shift),
            ("both", image_shift, text_shift),
        ]
    else:
        result = [("baseline", 0, 0)]
        seen = {0}
        for value in image_shift_sweep:
            try:
                delta = "nearest" if value == "nearest" else int(value)
            except ValueError:
                raise ValueError(
                    "Image sweep values must be integers or 'nearest'"
                ) from None
            if delta not in seen:
                result.append((f"image_{delta}", delta, 0))
                seen.add(delta)
    return ([("target_only", 0, 0)] if target_only else []) + result


def shift_summary(log_path, rid):
    """Read measured-request diagnostics, summing changed counts across chunks."""
    chunks = []
    with log_path.open() as file:
        for line in file:
            if "DFLASH_PROMPT_ROPE " in line:
                entry = json.loads(line.split("DFLASH_PROMPT_ROPE ", 1)[1])
                if entry.get("rid") == rid:
                    chunks.append(entry)
    if not chunks:
        raise RuntimeError(f"No confirmed RoPE mapping for {rid}; inspect {log_path}")
    keys = (
        "effective_image_shift",
        "effective_text_shift",
        "eligible_text_tokens",
        "image_end_distance_before",
        "image_end_distance_after",
        "eligible_text_end_distance_before",
        "eligible_text_end_distance_after",
    )
    summary = {key: chunks[0].get(key) for key in keys}
    if any(any(entry.get(key) != summary[key] for key in keys) for entry in chunks):
        raise RuntimeError(f"Inconsistent full-prompt mapping across chunks for {rid}")
    summary["changed_tokens"] = sum(entry["changed_tokens"] for entry in chunks)
    return summary


def mask_summary(
    log_path,
    rid,
    expected_image_tokens,
    masked,
    expected_text_tokens=0,
    mask_text=False,
):
    """Require runtime confirmation, including every measured Draft round."""
    entries = []
    for line in log_path.read_text().splitlines():
        if "DFLASH_IMAGE_MASK " in line:
            entry = json.loads(line.split("DFLASH_IMAGE_MASK ", 1)[1])
            if entry.get("rid") == rid:
                entries.append(entry)
    expected = expected_image_tokens if masked else 0
    expected_text = expected_text_tokens if mask_text else 0
    if not entries or any(
        x["masked_image_tokens"] != expected
        or x.get("masked_text_tokens", 0) != expected_text
        or x.get("masked_prompt_tokens", x["masked_image_tokens"])
        != expected + expected_text
        for x in entries
    ):
        raise RuntimeError(f"Unconfirmed image KV mask for {rid}; inspect {log_path}")
    return {
        "masked_image_tokens": expected,
        "masked_text_tokens": expected_text,
        "masked_prompt_tokens": expected + expected_text,
        "mask_draft_rounds": len(entries),
    }


def baseline_changes(row, baseline):
    """Acceptance comparisons require the same sample, context, and image size."""
    if baseline is None or row.get("spec_accept_length") is None:
        return {
            "baseline_spec_accept_length": None,
            "spec_accept_length_change_pct": None,
            "spec_verify_ct_change": None,
        }
    value = baseline["spec_accept_length"]
    return {
        "baseline_spec_accept_length": value,
        "spec_accept_length_change_pct": 100 * (row["spec_accept_length"] / value - 1)
        if value
        else None,
        "spec_verify_ct_change": row["spec_verify_ct"] - baseline["spec_verify_ct"],
    }


def post_stream(url, payload, timeout):
    """Native /generate streams cumulative IDs; retain the final complete reply."""
    request = urllib.request.Request(
        url + "/generate",
        json.dumps({**payload, "stream": True}).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    last = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event)
            if first is None and (event.get("output_ids") or event.get("text")):
                first = time.perf_counter() - started
            last = event
    elapsed = time.perf_counter() - started
    if last is None or not last.get("meta_info", {}).get("finish_reason"):
        raise RuntimeError(f"Incomplete generation response: {last}")
    return last, elapsed, first


def log_tail(path, lines=1):
    if path is None or not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(0, 2)
        file.seek(max(0, file.tell() - 8192))
        return "\n".join(file.read().decode(errors="replace").splitlines()[-lines:])


def wait_ready(url, process, timeout, log_path=None):
    started = time.monotonic()
    deadline = started + timeout
    next_update = started
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited during startup (exit={process.returncode}). "
                f"Log: {log_path}\n{log_tail(log_path, 40)}"
            )
        if time.monotonic() >= next_update:
            print(
                f"[startup] waiting {time.monotonic() - started:.0f}s/{timeout}s "
                f"pid={process.pid} log={log_path}\n  {log_tail(log_path)}",
                flush=True,
            )
            next_update = time.monotonic() + 10
        try:
            with urllib.request.urlopen(url + "/health_generate", timeout=2) as reply:
                if reply.status == 200:
                    print(
                        f"[startup] ready after {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(
        f"Server startup timed out. Log: {log_path}\n{log_tail(log_path, 40)}"
    )


def stop_server(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def image_data(path, max_edge):
    data = path.read_bytes()
    if max_edge:
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        scale = max_edge / max(image.size)
        size = tuple(max(1, round(side * scale)) for side in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        mime = "image/png"
    else:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(data).decode(), hashlib.sha256(
        data
    ).hexdigest()


def filler_text(tokenizer, source, count):
    if count == 0:
        return "", 0
    tokens = tokenizer.encode(source, add_special_tokens=False)
    if not tokens:
        raise ValueError("Nonempty --filler-file is required for long-context runs")
    repeated = (tokens * ((count + len(tokens) - 1) // len(tokens)))[:count]
    text = tokenizer.decode(repeated)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def load_samples(path):
    samples = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        sample_id = str(sample["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen.add(sample_id)
        image = (path.parent / sample["image"]).resolve()
        question = sample.get("question") or sample.get("prompt")
        if (
            sample.get("question")
            and sample.get("prompt")
            and sample["question"] != sample["prompt"]
        ):
            raise ValueError(
                f"Sample {sample_id} has conflicting question/prompt fields"
            )
        if not image.is_file() or not isinstance(question, str) or not question.strip():
            raise ValueError(f"Sample {sample_id} needs a local image and question")
        samples.append(
            {**sample, "id": sample_id, "image": image, "question": question}
        )
    if not samples:
        raise ValueError("Samples file is empty")
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3.5-4B/9B VLM checkpoint")
    parser.add_argument("--draft", required=True, help="Matching DFlash checkpoint")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-shift", type=int, default=128)
    parser.add_argument(
        "--image-shift-sweep",
        nargs="+",
        metavar="SHIFT",
        help="Run baseline plus image-only translations, e.g. 0 128 1024 4096 nearest; ignores --image-shift/--text-shift",
    )
    parser.add_argument("--text-shift", type=int, default=128)
    parser.add_argument("--text-keep-recent", type=int, default=256)
    parser.add_argument("--context-tokens", nargs="+", type=int, default=[0])
    parser.add_argument("--filler-file", type=Path)
    parser.add_argument(
        "--image-max-edges",
        nargs="+",
        type=int,
        default=[0],
        help="0 keeps original image; other values resize longest edge",
    )
    parser.add_argument("--include-target-only", action="store_true")
    diagnostic_group = parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument(
        "--image-kv-diagnostic",
        action="store_true",
        help="Compare baseline/nearest/image_mask with matched eager FlashInfer custom masks; not a speed benchmark",
    )
    diagnostic_group.add_argument(
        "--modality-kv-diagnostic",
        action="store_true",
        help="Baseline plus image/text/both nearest and masking (7 Draft arms), matched eager FlashInfer",
    )
    parser.add_argument(
        "--round-log",
        action="store_true",
        help="Log acceptance and image distance per Draft round; D2H copies perturb timing",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Separate diagnostic run; synchronizes CUDA and logs per-round phase timings",
    )
    parser.add_argument(
        "--sequential-kv",
        action="store_true",
        help="Exercise the non-fused Context KV path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Continue to the output budget after EOS for a fixed-length acceptance probe; not natural VQA answer evaluation",
    )
    parser.add_argument(
        "--allow-output-mismatch",
        action="store_true",
        help="Keep output identity flags but do not fail the completed acceptance-only matrix on mismatches",
    )
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument(
        "--server-args-json",
        default="[]",
        help='Extra launch arguments as a JSON array, e.g. ["--tp","2"]',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files/print launch commands only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    diagnostic = args.image_kv_diagnostic or args.modality_kv_diagnostic
    samples = load_samples(args.samples.resolve())
    extra = json.loads(args.server_args_json)
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise ValueError("--server-args-json must be a JSON array of strings")
    if any(x < 0 for x in args.context_tokens + args.image_max_edges):
        raise ValueError("Context lengths and image sizes must be nonnegative")
    if args.text_keep_recent < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid recent-text window or output length")
    filler = args.filler_file.read_text() if args.filler_file else ""
    if any(args.context_tokens) and not filler.strip():
        raise ValueError("Long-context runs require --filler-file")
    # Reject collisions with the experiment's fixed launch settings.
    owned = {
        "--model-path",
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--port",
        "--host",
        "--max-running-requests",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--enable-mixed-chunk",
        "--language-model-only",
        "--skip-tokenizer-init",
    }
    if any(x.split("=", 1)[0] in owned for x in extra):
        raise ValueError(
            "Extra server arguments override a required experiment setting"
        )
    if diagnostic:
        controlled = {
            "--speculative-draft-attention-backend",
            "--speculative-draft-window-size",
            "--disable-cuda-graph",
            "--enable-dp-attention",
            "--attn-dp-size",
            "--dp-size",
            "--tp",
            "--tp-size",
        }
        if any(
            x.split("=", 1)[0] in controlled
            or x.startswith(
                (
                    "--cuda-graph-",
                    "--disable-prefill-cuda-graph",
                    "--disable-decode-cuda-graph",
                )
            )
            for x in extra
        ):
            raise ValueError(
                "Image diagnostic controls graph/backend and single-GPU settings"
            )
    plans = []
    for name, image_shift, text_shift in arms(
        args.image_shift,
        args.text_shift,
        args.include_target_only,
        args.image_shift_sweep,
        args.image_kv_diagnostic,
        args.modality_kv_diagnostic,
    ):
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            *extra,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--disable-radix-cache",
            "--disable-overlap-schedule",
            "--max-running-requests",
            "1",
        ]
        if name != "target_only":
            command += [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft,
            ]
            if diagnostic:
                command += ["--speculative-draft-attention-backend", "flashinfer"]
        if diagnostic:
            command += [
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ]
        plans.append(
            (
                name,
                command,
                dict(
                    zip(
                        ENV_KEYS,
                        map(
                            str,
                            (
                                0 if image_shift == "nearest" else image_shift,
                                0 if text_shift == "nearest" else text_shift,
                                args.text_keep_recent,
                            ),
                        ),
                    )
                ),
            )
        )
        # Explicitly reset this for every arm, including baseline, so an inherited
        # environment cannot silently enable nearest placement in control runs.
        plans[-1][2]["SGLANG_DFLASH_PROMPT_IMAGE_NEAREST"] = str(
            int(image_shift == "nearest")
        )
        plans[-1][2].update(
            {
                "SGLANG_DFLASH_ROPE_DIAGNOSTIC": str(
                    int(diagnostic and name != "target_only")
                ),
                "SGLANG_DFLASH_PROMPT_TEXT_NEAREST": str(int(text_shift == "nearest")),
                "SGLANG_DFLASH_IMAGE_KV_MASK": str(
                    int(name in ("image_mask", "both_mask"))
                ),
                "SGLANG_DFLASH_TEXT_KV_MASK": str(
                    int(name in ("text_mask", "both_mask"))
                ),
                "SGLANG_DFLASH_ROPE_ROUND_LOG": str(int(args.round_log or diagnostic)),
            }
        )
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return

    print("[setup] Loading Target config and multimodal processor...", flush=True)
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(args.model)
    if "Qwen3_5ForConditionalGeneration" not in (config.architectures or []):
        raise ValueError("Use the full Qwen3.5-4B/9B VLM, not a text-only checkpoint")
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[setup] Processor ready. Results: {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "args": vars(args),
        "plans": plans,
        "target_architectures": config.architectures,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("sglang", "transformers")
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "filler_sha256": hashlib.sha256(filler.encode()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2)
    )
    url = f"http://127.0.0.1:{args.port}"
    # Refuse to accidentally benchmark a pre-existing service on this port.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    baselines = {}
    baseline_metrics = {}
    targets = {}
    rows = []
    mismatch = False
    with (args.output_dir / "results.jsonl").open("w") as output:
        for arm_index, (name, command, overrides) in enumerate(plans, 1):
            log_path = args.output_dir / f"server_{name}.log"
            print(
                f"\n[arm {arm_index}/{len(plans)}] {name}: starting server; log={log_path}",
                flush=True,
            )
            env = os.environ.copy()
            env.update(overrides)
            env["SGLANG_DFLASH_ROPE_PROFILE"] = str(int(args.profile))
            env["SGLANG_DFLASH_ROPE_FORCE_SEQUENTIAL_KV"] = str(int(args.sequential_kv))
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_ready(url, process, args.startup_timeout, log_path)
                    case_index = 0
                    total_cases = (
                        len(samples)
                        * len(args.image_max_edges)
                        * len(args.context_tokens)
                    )
                    for sample in samples:
                        for edge in args.image_max_edges:
                            image, image_hash = image_data(sample["image"], edge)
                            for count in args.context_tokens:
                                case_index += 1
                                print(
                                    f"[{name} {case_index}/{total_cases}] id={sample['id']} "
                                    f"context={count} image_edge={edge}: preparing input",
                                    flush=True,
                                )
                                context, actual_count = filler_text(
                                    processor.tokenizer, filler, count
                                )
                                content = [
                                    {"type": "image", "image": str(sample["image"])}
                                ]
                                content.append(
                                    {
                                        "type": "text",
                                        "text": context + "\n" + sample["question"],
                                    }
                                )
                                prompt = processor.apply_chat_template(
                                    [{"role": "user", "content": content}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                )
                                rid = f"{name}-{sample['id']}-{count}-{edge}"
                                payload = {
                                    "rid": rid,
                                    "text": prompt,
                                    "image_data": [image],
                                    "sampling_params": {
                                        "temperature": 0,
                                        "max_new_tokens": args.max_new_tokens,
                                        "ignore_eos": args.ignore_eos,
                                    },
                                }
                                # Warm each shape separately; no prefix reuse. Excluded from results.
                                print(
                                    f"[{name} {case_index}/{total_cases}] warmup (8 tokens)...",
                                    flush=True,
                                )
                                post_stream(
                                    url,
                                    {
                                        **payload,
                                        "rid": rid + "-warmup",
                                        "sampling_params": {
                                            "temperature": 0,
                                            "max_new_tokens": 8,
                                            "ignore_eos": args.ignore_eos,
                                        },
                                    },
                                    args.request_timeout,
                                )
                                print(
                                    f"[{name} {case_index}/{total_cases}] measuring (max {args.max_new_tokens} tokens)...",
                                    flush=True,
                                )
                                response, elapsed, ttft = post_stream(
                                    url, payload, args.request_timeout
                                )
                                meta = response["meta_info"]
                                if meta.get("image_tokens", 0) <= 0:
                                    raise RuntimeError(
                                        "No visual tokens reported: refusing a text-only result"
                                    )
                                if (
                                    name != "target_only"
                                    and meta.get("spec_verify_ct", 0) <= 0
                                ):
                                    raise RuntimeError(
                                        "No speculative verification reported; inspect server log"
                                    )
                                ids = response.get("output_ids")
                                if ids is None:
                                    raise RuntimeError(
                                        "Output token IDs required for correctness comparison"
                                    )
                                key = (sample["id"], count, edge)
                                if name == "target_only":
                                    targets[key] = ids
                                if name == "baseline":
                                    baselines[key] = ids
                                same = (
                                    ids == baselines[key] if key in baselines else None
                                )
                                same_target = (
                                    ids == targets[key] if key in targets else None
                                )
                                mismatch |= same is False or same_target is False
                                tokens = meta.get("completion_tokens", len(ids))
                                row = {
                                    "arm": name,
                                    "id": sample["id"],
                                    "rid": rid,
                                    "profile": args.profile,
                                    "image_kv_diagnostic": args.image_kv_diagnostic,
                                    "modality_kv_diagnostic": args.modality_kv_diagnostic,
                                    "round_log": args.round_log or diagnostic,
                                    "timing_diagnostic_only": args.profile
                                    or args.round_log
                                    or diagnostic,
                                    "requested_image_shift": overrides[ENV_KEYS[0]],
                                    "requested_text_shift": overrides[ENV_KEYS[1]],
                                    "image_shift_mode": "nearest"
                                    if overrides["SGLANG_DFLASH_PROMPT_IMAGE_NEAREST"]
                                    == "1"
                                    else "fixed",
                                    "text_shift_mode": "nearest"
                                    if overrides["SGLANG_DFLASH_PROMPT_TEXT_NEAREST"]
                                    == "1"
                                    else "fixed",
                                    "text_keep_recent": args.text_keep_recent,
                                    "context_tokens_requested": count,
                                    "context_tokens_actual": actual_count,
                                    "image_max_edge": edge,
                                    "image_sha256": image_hash,
                                    "prompt_tokens": meta.get("prompt_tokens"),
                                    "image_tokens": meta["image_tokens"],
                                    "completion_tokens": tokens,
                                    "max_new_tokens": args.max_new_tokens,
                                    "ignore_eos": args.ignore_eos,
                                    "fixed_output_budget_met": tokens
                                    == args.max_new_tokens,
                                    "spec_accept_length": meta.get(
                                        "spec_accept_length"
                                    ),
                                    "spec_accept_rate": meta.get("spec_accept_rate"),
                                    "spec_verify_ct": meta.get("spec_verify_ct"),
                                    "elapsed_s": elapsed,
                                    "client_ttft_s": ttft,
                                    "total_tokens_per_s": tokens / elapsed,
                                    "decode_tokens_per_s": (tokens - 1)
                                    / (elapsed - ttft)
                                    if tokens > 1
                                    and ttft is not None
                                    and elapsed > ttft
                                    else None,
                                    "matches_baseline": same,
                                    "matches_target": same_target,
                                }
                                intervention = (
                                    int(overrides[ENV_KEYS[0]]) != 0
                                    or int(overrides[ENV_KEYS[1]]) != 0
                                    or overrides["SGLANG_DFLASH_PROMPT_IMAGE_NEAREST"]
                                    == "1"
                                    or overrides["SGLANG_DFLASH_PROMPT_TEXT_NEAREST"]
                                    == "1"
                                    or (diagnostic and name != "target_only")
                                )
                                row.update(
                                    mask_summary(
                                        log_path,
                                        rid,
                                        meta["image_tokens"],
                                        name in ("image_mask", "both_mask"),
                                        meta["prompt_tokens"] - meta["image_tokens"],
                                        name in ("text_mask", "both_mask"),
                                    )
                                    if diagnostic and name != "target_only"
                                    else {
                                        "masked_image_tokens": 0,
                                        "masked_text_tokens": 0,
                                        "masked_prompt_tokens": 0,
                                        "mask_draft_rounds": None,
                                    }
                                )
                                if (
                                    diagnostic
                                    and name != "target_only"
                                    and row["mask_draft_rounds"]
                                    != meta["spec_verify_ct"]
                                ):
                                    raise RuntimeError(
                                        f"Mask diagnostics do not cover every verification round for {rid}"
                                    )
                                row.update(
                                    shift_summary(log_path, rid)
                                    if intervention
                                    else {
                                        "effective_image_shift": 0,
                                        "effective_text_shift": 0,
                                        "eligible_text_tokens": None,
                                        "image_end_distance_before": None,
                                        "image_end_distance_after": None,
                                        "eligible_text_end_distance_before": None,
                                        "eligible_text_end_distance_after": None,
                                        "changed_tokens": 0,
                                    }
                                )
                                if name == "baseline":
                                    baseline_metrics[key] = row.copy()
                                row.update(
                                    baseline_changes(row, baseline_metrics.get(key))
                                )
                                rows.append(row)
                                output.write(
                                    json.dumps(
                                        {
                                            **row,
                                            "response": response,
                                            "reference_answer": sample.get("answer"),
                                        }
                                    )
                                    + "\n"
                                )
                                output.flush()
                                print(json.dumps(row), flush=True)
                finally:
                    print(f"[{name}] stopping server...", flush=True)
                    stop_server(process)
            measured_rids = {row["rid"] for row in rows if row["arm"] == name}
            for marker, stem in (
                ("DFLASH_PROMPT_ROPE ", "shifts"),
                ("DFLASH_ROPE_ROUND ", "rounds"),
                ("DFLASH_IMAGE_MASK ", "masks"),
            ):
                with (args.output_dir / f"{stem}_{name}.jsonl").open("w") as extracted:
                    for line in (
                        (args.output_dir / f"server_{name}.log")
                        .read_text()
                        .splitlines()
                    ):
                        if marker in line:
                            data = json.loads(line.split(marker, 1)[1])
                            if (
                                set(data.get("rids", [data.get("rid", "")]))
                                <= measured_rids
                            ):
                                extracted.write(json.dumps(data) + "\n")
    with (args.output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from summarize_results import summarize, write_summary

    write_summary(summarize(rows), args.output_dir / "summary.csv")
    print(
        f"[done] {len(rows)} results saved to {args.output_dir / 'results.csv'}",
        flush=True,
    )
    print(
        f"[done] Acceptance comparison: {args.output_dir / 'summary.csv'}", flush=True
    )
    if mismatch:
        message = "Output mismatch detected; identity flags are retained in results."
        if args.allow_output_mismatch:
            print(f"[acceptance-only] {message}", flush=True)
        else:
            raise SystemExit(message)


if __name__ == "__main__":
    main()
