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


class PromptTranslationTests(unittest.TestCase):
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
