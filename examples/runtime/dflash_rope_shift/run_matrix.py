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


def arms(image_shift, text_shift, target_only):
    result = [
        ("baseline", 0, 0),
        ("image", image_shift, 0),
        ("text", 0, text_shift),
        ("both", image_shift, text_shift),
    ]
    return ([("target_only", 0, 0)] if target_only else []) + result


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
                    print(f"[startup] ready after {time.monotonic() - started:.1f}s", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Server startup timed out. Log: {log_path}\n{log_tail(log_path, 40)}")


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
        if not image.is_file() or not sample.get("question"):
            raise ValueError(f"Sample {sample_id} needs a local image and question")
        samples.append({**sample, "id": sample_id, "image": image})
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
    plans = []
    for name, image_shift, text_shift in arms(
        args.image_shift, args.text_shift, args.include_target_only
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
        plans.append(
            (
                name,
                command,
                dict(
                    zip(
                        ENV_KEYS,
                        map(str, (image_shift, text_shift, args.text_keep_recent)),
                    )
                ),
            )
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
    targets = {}
    rows = []
    mismatch = False
    with (args.output_dir / "results.jsonl").open("w") as output:
        for arm_index, (name, command, overrides) in enumerate(plans, 1):
            log_path = args.output_dir / f"server_{name}.log"
            print(f"\n[arm {arm_index}/{len(plans)}] {name}: starting server; log={log_path}", flush=True)
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
                    total_cases = len(samples) * len(args.image_max_edges) * len(args.context_tokens)
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
                                    },
                                }
                                # Warm each shape separately; no prefix reuse. Excluded from results.
                                print(f"[{name} {case_index}/{total_cases}] warmup (8 tokens)...", flush=True)
                                post_stream(
                                    url,
                                    {
                                        **payload,
                                        "rid": rid + "-warmup",
                                        "sampling_params": {
                                            "temperature": 0,
                                            "max_new_tokens": 8,
                                        },
                                    },
                                    args.request_timeout,
                                )
                                print(f"[{name} {case_index}/{total_cases}] measuring (max {args.max_new_tokens} tokens)...", flush=True)
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
                                    "context_tokens_requested": count,
                                    "context_tokens_actual": actual_count,
                                    "image_max_edge": edge,
                                    "image_sha256": image_hash,
                                    "prompt_tokens": meta.get("prompt_tokens"),
                                    "image_tokens": meta["image_tokens"],
                                    "completion_tokens": tokens,
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
    print(f"[done] {len(rows)} results saved to {args.output_dir / 'results.csv'}", flush=True)
    if mismatch:
        raise SystemExit(
            "Output mismatch detected. Inspect results before interpreting speedups."
        )


if __name__ == "__main__":
    main()
