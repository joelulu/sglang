"""CPU-only checks: python examples/runtime/dflash_rope_shift/test_cpu.py.

Load the lightweight helper directly so these tests do not require CUDA, torch,
or importing the complete SGLang runtime.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "python/sglang/srt/speculative/dflash_rope_shift.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shift_module = load_module("dflash_rope_shift_cpu_test", HELPER)
PromptRopeShift = shift_module.PromptRopeShift
runner = load_module(
    "dflash_rope_matrix_cpu_test", Path(__file__).with_name("run_matrix.py")
)
summary_module = load_module(
    "dflash_rope_summary_cpu_test", Path(__file__).with_name("summarize_results.py")
)
sys.modules["summarize_results"] = summary_module
round_summary_module = load_module(
    "dflash_round_summary_cpu_test", Path(__file__).with_name("summarize_rounds.py")
)


class PromptTranslationTests(unittest.TestCase):
    def test_generated_windows_keep_only_recent_history_and_all_block_keys(self):
        for history in [0, 1, 3, 16, 33]:
            for window in [-1, 0, 1, 4, 16]:
                with self.subTest(history=history, window=window):
                    prefix = 10 + history
                    allow = shift_module.context_key_allow_mask(
                        10, prefix, 8, [(2, 6)], True, True, window
                    )
                    expected = history if window == -1 else min(history, window)
                    self.assertEqual(allow[:10], [False] * 10)
                    self.assertEqual(
                        allow[10:prefix],
                        [False] * (history - expected) + [True] * expected,
                    )
                    self.assertEqual(allow[prefix:], [True] * 8)
        with self.assertRaises(ValueError):
            shift_module.context_key_allow_mask(10, 15, 8, [], True, True, -2)

    def test_generated_window_can_be_varied_without_changing_prompt_visibility(self):
        full = shift_module.context_key_allow_mask(10, 40, 8, [(2, 6)], False, False)
        recent = shift_module.context_key_allow_mask(
            10, 40, 8, [(2, 6)], False, False, 4
        )
        self.assertTrue(all(full))
        self.assertTrue(all(recent[:10]))
        self.assertEqual(recent[10:36], [False] * 26)
        self.assertTrue(all(recent[36:]))

    def test_generated_mask_confirmation_checks_history_counts_and_block(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            entries = []
            for history in [0, 1, 8]:
                entries.append(
                    {
                        "rid": "case",
                        "query_position": 10 + history,
                        "masked_image_tokens": 6,
                        "masked_text_tokens": 4,
                        "masked_prompt_tokens": 10,
                        "generated_kv_keep_recent": 4,
                        "generated_history_tokens": history,
                        "visible_generated_tokens": min(history, 4),
                        "masked_generated_tokens": max(history - 4, 0),
                        "block_tokens": 8,
                        "visible_block_tokens": 8,
                    }
                )

            def write():
                path.write_text(
                    "\n".join("DFLASH_IMAGE_MASK " + json.dumps(e) for e in entries)
                )

            write()
            result = runner.mask_summary(path, "case", 6, True, 4, True, 4)
            self.assertEqual(result["visible_generated_tokens_max"], 4)
            self.assertEqual(result["visible_generated_tokens_min"], 0)
            for key in [
                "visible_generated_tokens",
                "masked_generated_tokens",
                "visible_block_tokens",
                "generated_kv_keep_recent",
            ]:
                saved = entries[-1][key]
                entries[-1][key] += 1
                write()
                with self.assertRaises(RuntimeError):
                    runner.mask_summary(path, "case", 6, True, 4, True, 4)
                entries[-1][key] = saved

    def test_round_summaries_group_by_position_and_ignore_warmup(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            entries = []
            for rid, positions in [("case", [0, 1, 32, 65, 128]), ("warmup", [0])]:
                for position in positions:
                    entries.append(
                        {
                            "rids": [rid],
                            "generated_tokens_before": [position],
                            "accepted_draft_tokens": [0 if position == 0 else 3],
                            "committed_tokens": [1 if position == 0 else 4],
                        }
                    )
            (directory / "rounds_both_mask_gen1.jsonl").write_text(
                "\n".join(json.dumps(e) for e in entries)
            )
            rows = [
                {
                    "arm": "both_mask_gen1",
                    "rid": "case",
                    "spec_verify_ct": 5,
                    "context_tokens_requested": 1024,
                    "image_max_edge": 0,
                }
            ]
            groups = {
                r["stage_by_generated_position"]: r
                for r in round_summary_module.summarize_rounds(rows, directory)
            }
            self.assertEqual(groups["first_round"]["accepted_draft_tokens_mean"], 0)
            self.assertEqual(groups["0-31"]["rounds"], 2)
            self.assertEqual(groups["0-31"]["accepted_draft_tokens_mean"], 1.5)
            self.assertEqual(groups["32-63"]["rounds"], 1)
            rows[0]["spec_verify_ct"] = 6
            with self.assertRaises(ValueError):
                round_summary_module.summarize_rounds(rows, directory)

    def test_text_and_both_masks_cover_prompt_only(self):
        spans = [(2, 4), (7, 8)]
        for image, text in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(image=image, text=text):
                allow = shift_module.context_key_allow_mask(
                    10, 13, 4, spans, image, text
                )
                self.assertTrue(all(allow[10:]))
                self.assertEqual([allow[i] for i in [2, 3, 4, 7, 8]], [not image] * 5)
                self.assertEqual([allow[i] for i in [0, 1, 5, 6, 9]], [not text] * 5)
                self.assertEqual(allow.count(False), 5 * image + 5 * text)

    def test_text_nearest_preserves_recent_suffix_and_internal_spacing(self):
        config = PromptRopeShift(text_nearest=True, text_keep_recent=8)
        spans = [(4, 23)]
        full, info = config.chunk_offsets(100, 0, 110, spans)
        self.assertTrue(config.enabled)
        self.assertEqual(info["effective_text_shift"], 8)
        self.assertEqual(info["eligible_text_end_distance_after"], 1)
        self.assertEqual(full[92:], [0] * 18)
        self.assertEqual(full[4:24], [0] * 20)
        shifted_text = [p + full[p] for p in range(100) if p < 92 and not 4 <= p <= 23]
        original_text = [p for p in range(92) if not 4 <= p <= 23]
        self.assertEqual(shifted_text, [p + 8 for p in original_text])
        chunks = []
        for start, length in [(0, 10), (10, 90), (100, 10)]:
            offsets, _ = config.chunk_offsets(100, start, length, spans)
            chunks += offsets
        self.assertEqual(full, chunks)
        _, both = PromptRopeShift(
            text_nearest=True, image_nearest=True, text_keep_recent=8
        ).chunk_offsets(100, 0, 100, spans)
        self.assertEqual(both["effective_text_shift"], 8)
        self.assertEqual(both["effective_image_shift"], 76)
        _, no_recent = PromptRopeShift(
            text_nearest=True, text_keep_recent=0
        ).chunk_offsets(100, 0, 100, spans)
        self.assertEqual(no_recent["effective_text_shift"], 0)
        _, no_text = PromptRopeShift(text_nearest=True).chunk_offsets(
            10, 0, 10, [(0, 9)]
        )
        self.assertEqual(no_text["effective_text_shift"], 0)
        self.assertIsNone(no_text["eligible_text_end_distance_after"])

    def test_image_mask_keeps_text_generated_history_and_block(self):
        visibility = shift_module.context_key_allow_mask
        spans = [(2, 4), (7, 8)]
        mask = visibility(10, 13, 4, spans, True)
        self.assertEqual(len(mask), 17)
        self.assertEqual(
            [i for i, allowed in enumerate(mask) if not allowed], [2, 3, 4, 7, 8]
        )
        self.assertTrue(all(mask[10:]))
        self.assertTrue(all(visibility(10, 13, 4, spans, False)))
        # Same fixed prompt spans after another round; generated history stays visible.
        later = visibility(10, 20, 4, spans, True)
        self.assertEqual(later[:10], mask[:10])
        self.assertTrue(all(later[10:]))
        for prefix in [0, 9]:
            with self.assertRaises(ValueError):
                visibility(10, prefix, 4, spans, True)
        with self.assertRaises(ValueError):
            visibility(10, 13, 4, [(2, 7), (7, 8)], True)

    def test_nearest_is_fixed_and_preserves_visual_span(self):
        config = PromptRopeShift(image_nearest=True)
        self.assertTrue(config.enabled)
        full, info = config.chunk_offsets(100, 0, 110, [(4, 23)])
        self.assertEqual(info["effective_image_shift"], 76)
        self.assertEqual(info["image_end_distance_after"], 1)
        self.assertEqual([p + full[p] for p in range(4, 24)], list(range(80, 100)))
        self.assertEqual(full[100:], [0] * 10)
        chunks = []
        for start, size in [(0, 13), (13, 87), (100, 10)]:
            chunk, _ = config.chunk_offsets(100, start, size, [(4, 23)])
            chunks.extend(chunk)
        self.assertEqual(chunks, full)

    def test_nearest_without_image_is_noop(self):
        offsets, info = PromptRopeShift(image_nearest=True).chunk_offsets(30, 0, 30, [])
        self.assertEqual(offsets, [0] * 30)
        self.assertIsNone(info["image_end_distance_after"])

    def test_zero_identity(self):
        config = PromptRopeShift()
        self.assertFalse(config.enabled)
        offsets, _ = config.chunk_offsets(20, 0, 23, [(3, 9)])
        self.assertEqual(offsets, [0] * 23)

    def test_factorial_arms_and_prompt_boundary(self):
        for image_delta, text_delta in [(0, 0), (4, 0), (0, 3), (4, 3)]:
            config = PromptRopeShift(image_delta, text_delta, 5)
            offsets, _ = config.chunk_offsets(30, 0, 35, [(5, 14)])
            self.assertEqual(offsets[5:15], [image_delta] * 10)
            self.assertEqual(offsets[15:25], [text_delta] * 10)
            self.assertEqual(offsets[25:], [0] * 10)

    def test_chunked_prefill_matches_full_prompt(self):
        config = PromptRopeShift(19, 8, 10)
        spans = [(3, 22), (37, 49)]
        full, _ = config.chunk_offsets(80, 0, 90, spans)
        chunked = []
        for lo, hi in [(0, 12), (12, 44), (44, 80), (80, 90)]:
            part, _ = config.chunk_offsets(80, lo, hi - lo, spans)
            chunked += part
        self.assertEqual(full, chunked)

    def test_bounding_preserves_internal_distances(self):
        config = PromptRopeShift(1000, 0, 0)
        offsets, info = config.chunk_offsets(20, 0, 20, [(4, 17)])
        self.assertEqual(info["requested_image_shift"], 1000)
        self.assertEqual(info["effective_image_shift"], 2)
        moved = [p + offsets[p] for p in range(4, 18)]
        self.assertEqual(moved, list(range(6, 20)))

    def test_negative_shift_and_absent_modality(self):
        offsets, info = PromptRopeShift(-20, -20, 0).chunk_offsets(20, 0, 20, [(4, 8)])
        self.assertEqual(offsets[4:9], [-4] * 5)
        self.assertEqual(
            info["effective_text_shift"], 0
        )  # BOS at zero bounds the group
        _, info = PromptRopeShift(99, 0).chunk_offsets(20, 0, 20, [])
        self.assertEqual(info["effective_image_shift"], 0)

    def test_reject_misaligned_metadata(self):
        for spans in [[(-1, 4)], [(3, 20)], [(5, 3)], [(2, 5), (5, 8)]]:
            with self.assertRaises(ValueError):
                PromptRopeShift(1).chunk_offsets(20, 0, 20, spans)
        with self.assertRaises(ValueError):
            PromptRopeShift(text_keep_recent=-1)

    def test_multiple_requests_keep_independent_coordinates(self):
        config = PromptRopeShift(8, 0, 2)
        first, a = config.chunk_offsets(20, 0, 20, [(2, 15)])
        second, b = config.chunk_offsets(40, 10, 20, [(10, 15)])
        self.assertEqual(a["effective_image_shift"], 4)
        self.assertEqual(b["effective_image_shift"], 8)
        self.assertEqual(first[2], 4)
        self.assertEqual(second[0], 8)


class StreamTests(unittest.TestCase):
    def test_prompt_alias_preserves_questions_and_blank_answers(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "image.jpg").touch()
            path = directory / "samples.jsonl"
            sample = {
                "id": "a",
                "image": "image.jpg",
                "prompt": "what brand of phone?",
                "answer": "",
            }
            path.write_text(json.dumps(sample))
            loaded = runner.load_samples(path)
            self.assertEqual(loaded[0]["question"], sample["prompt"])
            self.assertEqual(loaded[0]["answer"], "")
            sample["question"] = "different question"
            path.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                runner.load_samples(path)

    def test_summary_pools_rounds_and_reports_paired_wins(self):
        rows = []
        for arm, counts in [("baseline", [2, 8]), ("both_mask", [3, 5])]:
            for i, rounds in enumerate(counts):
                rows.append(
                    {
                        "id": str(i),
                        "context_tokens_requested": 4096,
                        "image_max_edge": 0,
                        "arm": arm,
                        "image_sha256": "shared-image",
                        "completion_tokens": 8,
                        "spec_verify_ct": rounds,
                        "matches_baseline": True,
                        "matches_target": None,
                        "fixed_output_budget_met": True,
                        "timing_diagnostic_only": True,
                    }
                )
        summaries = {r["arm"]: r for r in summary_module.summarize(rows)}
        row = summaries["both_mask"]
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["unique_image_hashes"], 1)
        self.assertEqual(row["pooled_accept_length"], 2)
        self.assertAlmostEqual(row["pooled_accept_length_change_pct"], 25)
        self.assertEqual(row["both_mask_pooled_accept_length"], 2)
        self.assertEqual(row["vs_both_mask_change_pct"], 0)
        self.assertAlmostEqual(summaries["baseline"]["vs_both_mask_change_pct"], -20)
        self.assertAlmostEqual(
            row["paired_accept_change_mean_pct"], 100 * ((2 / 3 - 1) + (8 / 5 - 1)) / 2
        )
        self.assertEqual((row["win_count"], row["loss_count"]), (1, 1))
        self.assertEqual(row["target_comparisons"], 0)
        self.assertEqual(row["fixed_output_budget_met_count"], 2)
        # CSV serialization must give the same summaries as in-memory rows.
        string_rows = [
            {k: str(v) if v is not None else "" for k, v in r.items()} for r in rows
        ]
        self.assertEqual(
            summary_module.summarize(string_rows), summary_module.summarize(rows)
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summary_module.summarize(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Missing paired baseline"):
            summary_module.summarize(rows[2:])

    def test_backend_validation_runs_after_deferred_backend_initialization(self):
        # Execute the real lifecycle methods with a worker whose ModelRunner has
        # no backend until init_attention_backends(). No GPU imports required.
        import ast
        from contextlib import nullcontext
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch

        source = ROOT / "python/sglang/srt/speculative/dflash_worker_v2.py"
        tree = ast.parse(source.read_text())
        worker_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(item, ast.FunctionDef)
                and item.name == "_validate_image_diagnostic_backend"
                for item in node.body
            )
        )
        methods = {
            node.name: node
            for node in worker_class.body
            if isinstance(node, ast.FunctionDef)
        }
        constructor_backend_reads = [
            node
            for node in ast.walk(methods["__init__"])
            if isinstance(node, ast.Attribute) and node.attr == "attn_backend"
        ]
        self.assertEqual(constructor_backend_reads, [])
        module = ast.Module(
            body=[
                methods["init_attention_backends"],
                methods["_validate_image_diagnostic_backend"],
            ],
            type_ignores=[],
        )
        namespace = {
            "draft_pp_context": nullcontext,
            "draft_tp_context": lambda _: nullcontext(),
            "mambaish_config": lambda _: None,
        }
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

        class Backend:
            pass

        flashinfer = ModuleType("sglang.srt.layers.attention.flashinfer_backend")
        flashinfer.FlashInferAttnBackend = Backend
        runner = SimpleNamespace()  # Reproduce the missing attribute at construction.
        worker = SimpleNamespace(
            _rope_diagnostic=True,
            draft_owns_attention=False,
            draft_model_runner=runner,
            model_runner=SimpleNamespace(model_config=None),
        )
        worker._draft_worker = SimpleNamespace(
            init_attention_backends=lambda: setattr(runner, "attn_backend", Backend())
        )
        worker._validate_image_diagnostic_backend = lambda: namespace[
            "_validate_image_diagnostic_backend"
        ](worker)
        with patch.dict(sys.modules, {flashinfer.__name__: flashinfer}):
            self.assertFalse(hasattr(runner, "attn_backend"))
            namespace["init_attention_backends"](worker)
            self.assertIsInstance(runner.attn_backend, Backend)
            runner.attn_backend = object()
            with self.assertRaisesRegex(ValueError, "FlashInfer"):
                worker._validate_image_diagnostic_backend()
            worker._rope_diagnostic = False
            del runner.attn_backend
            worker._validate_image_diagnostic_backend()

    def test_mask_confirmation_checks_all_rounds_and_excludes_warmup(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            entries = [
                {"rid": "case-warmup", "masked_image_tokens": 0},
                {"rid": "case", "masked_image_tokens": 5},
                {"rid": "case", "masked_image_tokens": 5},
            ]

            def write():
                path.write_text(
                    "\n".join("DFLASH_IMAGE_MASK " + json.dumps(x) for x in entries)
                )

            write()
            self.assertEqual(
                runner.mask_summary(path, "case", 5, True),
                {
                    "masked_image_tokens": 5,
                    "masked_text_tokens": 0,
                    "masked_prompt_tokens": 5,
                    "mask_draft_rounds": 2,
                },
            )
            with self.assertRaises(RuntimeError):
                runner.mask_summary(path, "missing", 5, True)
            entries[-1]["masked_image_tokens"] = 0
            write()
            with self.assertRaises(RuntimeError):
                runner.mask_summary(path, "case", 5, True)

    def test_mask_confirmation_detects_text_mask_errors(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            for image, text in [
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ]:
                entry = {
                    "rid": "case",
                    "masked_image_tokens": 5 if image else 0,
                    "masked_text_tokens": 7 if text else 0,
                    "masked_prompt_tokens": 5 * image + 7 * text,
                }
                path.write_text("DFLASH_IMAGE_MASK " + json.dumps(entry))
                info = runner.mask_summary(path, "case", 5, image, 7, text)
                self.assertEqual(info["masked_prompt_tokens"], 5 * image + 7 * text)
                entry["masked_text_tokens"] += 1
                path.write_text("DFLASH_IMAGE_MASK " + json.dumps(entry))
                with self.assertRaises(RuntimeError):
                    runner.mask_summary(path, "case", 5, image, 7, text)

    def test_baseline_comparison_uses_acceptance_and_verification_counts(self):
        baseline = {"spec_accept_length": 2.0, "spec_verify_ct": 128}
        row = {"spec_accept_length": 2.56, "spec_verify_ct": 100}
        changes = runner.baseline_changes(row, baseline)
        self.assertAlmostEqual(changes["spec_accept_length_change_pct"], 28)
        self.assertEqual(changes["spec_verify_ct_change"], -28)
        self.assertEqual(
            runner.baseline_changes(baseline, baseline)[
                "spec_accept_length_change_pct"
            ],
            0,
        )
        self.assertIsNone(
            runner.baseline_changes(row, None)["spec_accept_length_change_pct"]
        )
        self.assertIsNone(
            runner.baseline_changes({"spec_accept_length": None}, baseline)[
                "spec_accept_length_change_pct"
            ]
        )

    def test_image_sweep_has_one_control_and_no_text_intervention(self):
        plans = runner.arms(
            128, 128, False, ["0", "128", "0128", "1024", "4096", "nearest"]
        )
        self.assertEqual(
            [name for name, _, _ in plans],
            ["baseline", "image_128", "image_1024", "image_4096", "image_nearest"],
        )
        self.assertTrue(all(text == 0 for _, _, text in plans))
        with self.assertRaises(ValueError):
            runner.arms(128, 128, False, ["invalid"])

    def test_shift_summary_aggregates_chunks_and_excludes_warmup(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            entries = [
                {"rid": "case-warmup", "changed_tokens": 100},
                {"rid": "case", "changed_tokens": 4, "effective_image_shift": 128},
                {"rid": "case", "changed_tokens": 6, "effective_image_shift": 128},
            ]
            path.write_text(
                "\n".join("[INFO] DFLASH_PROMPT_ROPE " + json.dumps(x) for x in entries)
            )
            summary = runner.shift_summary(path, "case")
            self.assertEqual(summary["changed_tokens"], 10)
            self.assertEqual(summary["effective_image_shift"], 128)
            with self.assertRaises(RuntimeError):
                runner.shift_summary(path, "missing")
            entries[-1]["effective_image_shift"] = 127
            path.write_text(
                "\n".join("DFLASH_PROMPT_ROPE " + json.dumps(x) for x in entries)
            )
            with self.assertRaises(RuntimeError):
                runner.shift_summary(path, "case")

    def test_diffusion_registry_inventory_schema(self):
        # This group is imported even by text/VLM launch_server. Its registration
        # loop unpacks exactly four fields per entry; an extra description broke
        # server startup before any Target or Draft was loaded.
        import ast

        path = ROOT / "python/sglang/kernels/ops/diffusion/__init__.py"
        tree = ast.parse(path.read_text())
        inventory = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_SPECS"
        )
        self.assertGreater(len(inventory.elts), 0)
        for entry in inventory.elts:
            with self.subTest(line=entry.lineno):
                self.assertIsInstance(entry, ast.Tuple)
                self.assertEqual(len(entry.elts), 4)

    def test_dry_run_has_real_vlm_matrix_and_isolated_servers(self):
        import json
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "image.jpg").touch()
            samples = directory / "samples.jsonl"
            samples.write_text(
                json.dumps(
                    {"id": "a", "image": "image.jpg", "question": "What is shown?"}
                )
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_matrix.py")),
                "--model",
                "Qwen/Qwen3.5-9B",
                "--draft",
                "matching-dflash",
                "--samples",
                str(samples),
                "--output-dir",
                str(directory / "out"),
                "--include-target-only",
                "--dry-run",
            ]
            plans = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(
                [plan[0] for plan in plans],
                ["target_only", "baseline", "image", "text", "both"],
            )
            for name, argv, environment in plans:
                self.assertIn("--disable-radix-cache", argv)
                self.assertNotIn("--language-model-only", argv)
                self.assertEqual(
                    "--speculative-algorithm" in argv, name != "target_only"
                )
            self.assertFalse((directory / "out").exists())
            sweep = json.loads(
                subprocess.check_output(
                    command
                    + ["--image-shift-sweep", "0", "128", "1024", "4096", "nearest"],
                    text=True,
                )
            )
            self.assertEqual(len(sweep), 6)
            for name, _, environment in sweep:
                self.assertEqual(environment["SGLANG_DFLASH_PROMPT_TEXT_SHIFT"], "0")
                self.assertEqual(
                    environment["SGLANG_DFLASH_PROMPT_IMAGE_NEAREST"],
                    "1" if name == "image_nearest" else "0",
                )
                if name == "image_4096":
                    self.assertEqual(
                        environment["SGLANG_DFLASH_PROMPT_IMAGE_SHIFT"], "4096"
                    )
            diagnostic = json.loads(
                subprocess.check_output(command + ["--image-kv-diagnostic"], text=True)
            )
            self.assertEqual(
                [x[0] for x in diagnostic],
                ["target_only", "baseline", "image_nearest", "image_mask"],
            )
            for name, argv, environment in diagnostic:
                self.assertIn("--cuda-graph-backend-decode", argv)
                self.assertIn("--cuda-graph-backend-prefill", argv)
                self.assertEqual(
                    environment["SGLANG_DFLASH_IMAGE_KV_MASK"],
                    "1" if name == "image_mask" else "0",
                )
                self.assertEqual(
                    environment["SGLANG_DFLASH_ROPE_DIAGNOSTIC"],
                    "0" if name == "target_only" else "1",
                )
                self.assertEqual(environment["SGLANG_DFLASH_ROPE_ROUND_LOG"], "1")
                self.assertEqual(
                    "--speculative-draft-attention-backend" in argv,
                    name != "target_only",
                )
            full = json.loads(
                subprocess.check_output(
                    command + ["--modality-kv-diagnostic"], text=True
                )
            )
            self.assertEqual(
                [plan[0] for plan in full],
                [
                    "target_only",
                    "baseline",
                    "image_nearest",
                    "image_mask",
                    "text_nearest",
                    "text_mask",
                    "both_nearest",
                    "both_mask",
                ],
            )
            for name, argv, environment in full:
                self.assertEqual(
                    environment["SGLANG_DFLASH_IMAGE_KV_MASK"],
                    str(int(name in ("image_mask", "both_mask"))),
                )
                self.assertEqual(
                    environment["SGLANG_DFLASH_TEXT_KV_MASK"],
                    str(int(name in ("text_mask", "both_mask"))),
                )
                self.assertEqual(
                    environment["SGLANG_DFLASH_PROMPT_IMAGE_NEAREST"],
                    str(int(name in ("image_nearest", "both_nearest"))),
                )
                self.assertEqual(
                    environment["SGLANG_DFLASH_PROMPT_TEXT_NEAREST"],
                    str(int(name in ("text_nearest", "both_nearest"))),
                )
                self.assertEqual(environment["SGLANG_DFLASH_PROMPT_TEXT_SHIFT"], "0")
                self.assertEqual(environment["SGLANG_DFLASH_PROMPT_IMAGE_SHIFT"], "0")
                self.assertIn("--cuda-graph-backend-decode", argv)
                self.assertEqual(
                    "--speculative-draft-attention-backend" in argv,
                    name != "target_only",
                )
            with self.assertRaises(ValueError):
                runner.arms(128, 128, False, ["nearest"], True)
            windows = json.loads(
                subprocess.check_output(
                    command + ["--generated-kv-diagnostic"], text=True
                )
            )
            self.assertEqual(
                [p[0] for p in windows],
                [
                    "target_only",
                    "baseline",
                    "both_mask",
                    "both_mask_gen1",
                    "both_mask_gen4",
                    "both_mask_gen16",
                ],
            )
            for name, argv, environment in windows:
                self.assertEqual(
                    environment["SGLANG_DFLASH_GENERATED_KV_KEEP_RECENT"],
                    name.removeprefix("both_mask_gen")
                    if name.startswith("both_mask_gen")
                    else "-1",
                )
                for modality in ["IMAGE", "TEXT"]:
                    self.assertEqual(
                        environment[f"SGLANG_DFLASH_{modality}_KV_MASK"],
                        str(int(name.startswith("both_mask"))),
                    )
                    self.assertEqual(
                        environment[f"SGLANG_DFLASH_PROMPT_{modality}_NEAREST"], "0"
                    )
                self.assertIn("--cuda-graph-backend-decode", argv)
            self.assertEqual(
                [
                    p[0]
                    for p in runner.arms(0, 0, False, generated_kv_windows=[0, 1, 1])
                ],
                ["baseline", "both_mask", "both_mask_gen0", "both_mask_gen1"],
            )
            for kwargs in [
                dict(generated_kv_windows=[-1]),
                dict(generated_kv_windows=[]),
                dict(generated_kv_windows=[1], image_kv_diagnostic=True),
                dict(generated_kv_windows=[1], image_shift_sweep=["nearest"]),
            ]:
                with self.assertRaises(ValueError):
                    runner.arms(0, 0, False, **kwargs)

    def test_native_stream_keeps_final_cumulative_ids(self):
        import io
        from unittest.mock import patch

        stream = io.BytesIO(
            b'data: {"text":"A","output_ids":[1],"meta_info":{}}\n\n'
            b'data: {"text":"AB","output_ids":[1,2],"meta_info":{"finish_reason":{"type":"length"}}}\n\n'
            b"data: [DONE]\n\n"
        )
        with patch.object(runner.urllib.request, "urlopen", return_value=stream):
            result, elapsed, first = runner.post_stream("http://localhost", {}, 1)
        self.assertEqual(result["output_ids"], [1, 2])
        self.assertLessEqual(first, elapsed)

    def test_incomplete_stream_fails(self):
        import io
        from unittest.mock import patch

        with patch.object(
            runner.urllib.request, "urlopen", return_value=io.BytesIO(b"data: [DONE]\n")
        ):
            with self.assertRaises(RuntimeError):
                runner.post_stream("http://localhost", {}, 1)


if __name__ == "__main__":
    unittest.main()
