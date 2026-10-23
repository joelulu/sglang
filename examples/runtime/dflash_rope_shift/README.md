# Qwen3.5 multimodal DFlash: prompt RoPE translations

Experimental, training-free image/text ablation. Use the **full multimodal
Qwen3.5-4B or Qwen3.5-9B Target** and its matching DFlash checkpoint. A text-only
checkpoint or a Draft trained for a different Target is not a valid baseline.
No GPU/model execution has been validated by the author of this patch; the
CPU tests cover mapping and streaming result handling. First establish that
your exact Target + Draft pair supports real images on this SGLang version.

## Checkout and environment

```bash
git clone -b research/vlm-dflash-rope-shift https://github.com/joelulu/sglang.git sglang-rope
cd sglang-rope
```

This branch starts at `1c967f5043e29bd08ae1ad80d1bba2757a17168f` (2026-10-22),
**not** SGLang 0.5.18. Use a separate environment with this checkout's runtime
and kernel requirements; do not copy these files into an old installation.
The base `python/pyproject.toml` declares torch 2.14.1 and FlashInfer 0.7.2.
Follow the repository installation instructions for your CUDA/platform. Once
the dependencies are installed, install this checkout into that environment:

```bash
python -m pip install --no-build-isolation --no-deps -e ./python
python -c 'import sglang; print(sglang.__file__)'
python examples/runtime/dflash_rope_shift/test_cpu.py
```

The printed import path must point into this checkout. The runner uses that
same Python to launch SGLang. Pillow is needed for the optional resize sweep;
Transformers AutoProcessor is used to format the real image/text chat template.

## Inputs

Create a JSONL file, one example per line. Image paths are relative to the JSONL
file (absolute paths also work). Images must exist locally. IDs must be unique.

```json
{"id":"example1","image":"images/example1.jpg","question":"What number is shown on the sign?","answer":"42"}
```

The optional answer is saved for manual/scorer evaluation, not auto-scored by
substring matching. Supply your own images/questions; no synthetic benchmark
results are included. For a visual grounding check, include the same question
with two different images whose correct answers differ.

## First run: short image/text inputs

```bash
python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model /path/to/Qwen3.5-9B \
  --draft /path/to/matching-dflash \
  --samples /path/to/examples.jsonl \
  --output-dir /path/to/results/rope_short \
  --include-target-only \
  --image-shift 128 --text-shift 128 --text-keep-recent 256 \
  --context-tokens 0 --max-new-tokens 256 \
  --server-args-json '["--mem-fraction-static","0.7"]'
```

Use `--model /path/to/Qwen3.5-4B` with its matching Draft if that is your working
pair. Add platform/TP/context-length arguments via the JSON array as required
by your already-working baseline. The runner does not override the checkpoint's
block size. `--dry-run` validates input files and prints launch commands without
loading Transformers, starting servers, or allocating GPU memory.

The runner launches and shuts down one local server per arm: Target-only
(optional), baseline DFlash, image-only, text-only, both. It forces sequential
requests, disabled overlap scheduling, and disabled radix reuse for **all**
arms. It refuses a pre-existing service on the chosen port. Each input shape
gets an 8-token warmup excluded from recorded results.

All measured requests are real image/text `/generate` calls using the Target's
chat template. A result with no reported `image_tokens` is rejected. DFlash
arms must also report positive `spec_verify_ct`. Output token IDs are compared
with baseline and, if requested, Target-only. A mismatch makes the run exit
with an error after saving results; inspect it before interpreting acceleration.

## Long context and visual length

For the text-distance experiment, keep images and questions unchanged. Supply
a neutral filler file with no answer clues. The runner inserts its text after
the image and before the final question:

```bash
python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model /path/to/Qwen3.5-9B --draft /path/to/matching-dflash \
  --samples /path/to/examples.jsonl --output-dir /path/to/results/rope_text_long \
  --include-target-only --context-tokens 0 4096 16384 \
  --filler-file /path/to/neutral_context.txt \
  --image-shift 128 --text-shift 128 --text-keep-recent 256
```

Lengths are requested filler-token counts, not total prompt lengths. CSV records
re-encoded filler count and server-reported total/visual counts. Set sufficient
model context capacity and GPU memory; the script does not truncate silently.
Length changes also change interference, so compare arms on each identical
input first, then examine the trend across lengths.

For the separate visual-length experiment, use `--context-tokens 0
--image-max-edges 0 1024 2048`. Nonzero sizes resize the same image's longest
edge, including upsampling; upsampling creates no new detail. The model's image
processor can cap resolution, so use **reported visual-token counts** to check
whether inputs actually differ. Do not initially combine both length sweeps:
one changes text distance, the other also changes visual content/quality.

## What is modified

* Only original Prompt Context **K** RoPE coordinates change: `p' = p + delta_m`.
* Target hidden states, Target M-RoPE/KV, Context V, Block Q/K positions,
  Attention masks and physical KV slots retain their original behavior.
* Each request uses fixed image/text translations throughout generation. There
  is no `q_current` mapping and no repeated rotation of cached keys.
* Images are identified using inclusive `mm_items[].offsets`, not token IDs
  (which can become hash placeholders). Missing/invalid/overlapping image spans
  fail rather than silently classify them as text. Video/audio inputs are out
  of scope. Visual boundaries/special tokens outside those spans count as text.
* `text_keep_recent` excludes the final N **sequence positions of the Prompt**
  from text shifts. This is not the most recent N text tokens or a moving
  generation window. All generated text remains unshifted, including re-prefill.
* A requested delta is bounded **once for the whole modality group** so its
  positions remain within `[0, prompt_length)`. Internal position differences
  remain exact. Image/text ordering relative to each other can change by design.
  Logs report requested/effective deltas, eligible counts and changed counts.
  A short question may leave little room for a positive image shift; negative
  text shifts may become zero because the group contains position zero.
  Never report a nonzero requested delta as an effective intervention without
  checking `shifts_*.jsonl`. Select offsets on development examples, then freeze
  them for held-out evaluation. A short Prompt may have no eligible early text.
* Mapping uses the full Prompt, so chunked Prefill computes the same result as
  unchunked Prefill. Both fused/sequential KV paths receive the same positions.
* Both delta defaults are zero: the worker returns the original position tensor
  without building modality maps or changing the existing KV materialization.

The helper is `python/sglang/srt/speculative/dflash_rope_shift.py`; integration
is the Prefill call in `dflash_worker_v2.py`. Kernel/model files are unchanged.
Runtime switches (restart the server after changing them):

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SGLANG_DFLASH_PROMPT_IMAGE_SHIFT` | 0 | Requested image translation |
| `SGLANG_DFLASH_PROMPT_TEXT_SHIFT` | 0 | Requested early-text translation |
| `SGLANG_DFLASH_PROMPT_TEXT_KEEP_RECENT` | 256 | Fixed Prompt suffix excluded from text shift |
| `SGLANG_DFLASH_ROPE_PROFILE` | 0 | Synchronized CUDA phase timing/acceptance diagnostics |
| `SGLANG_DFLASH_ROPE_FORCE_SEQUENTIAL_KV` | 0 | Use sequential instead of fused KV materialization |

Nonzero translations require `--disable-radix-cache`. Reusing a transformed
prefix across requests is unsupported. This is an experimental serving mode,
not a production cache-compatible feature.

## Results and optional diagnostics

* `manifest.json`: commands, environment translations, checkout revision,
  package versions, input arguments and filler hash.
* `results.jsonl`: raw responses/IDs and per-example metrics; flushed after
  every request. `results.csv`: flat comparison after a successful matrix.
* `server_<arm>.log`: startup and runtime logs, including shape/delta diagnostics.
* `shifts_<arm>.jsonl`: measured-request mapping diagnostics (no warmup/health).
* `rounds_<arm>.jsonl`: per-round diagnostics with `--profile`, `--round-log`,
  or `--image-kv-diagnostic`.

Acceptance length is SGLang's `completion_tokens / spec_verify_ct`, including
bonus tokens and affected by final output truncation. Acceptance rate excludes
bonus tokens. Client TTFT is time to the first nonempty streaming event;
`decode_tokens_per_s` is an approximate client-side metric, not GPU kernel time.
Use the same output budget across arms and report actual output length. No
accuracy improvement is expected from changing a lossless Draft; output identity
is a correctness check and provided reference answers need a task-specific scorer.

Run the same command in a **new output directory** with `--profile` to collect
CUDA stream elapsed `draft_ms` (Draft forward + proposal head),
`verify_forward_ms` (Target verification forward only), committed tokens per
round, and block size. These exclude KV materialization, acceptance and other
bookkeeping; do not add them up as end-to-end time. Profile runs synchronize and
copy acceptance lengths to CPU, so **do not use their throughput as speedup
results**. Normal runs do not add timing events or per-round host synchronization.

Repeat in another directory with `--sequential-kv` to exercise the other KV
materialization path. Both should preserve output identity; GPU numeric/fused
equivalence and Qwen3.5 hybrid-state rollback still require real hardware checks.

Interpretation: image-only improvements support investigating visual Context
positioning; text-only improvements indicate a broader text-position effect;
both exceeding single-modality arms suggests complementarity. Improvements in
acceptance must also improve unprofiled end-to-end speed. These experiments
alone do not establish RoPE training-range mismatch or improved visual grounding.

## Image-only translation sweep (next diagnostic)

Keep the same image/question, use one fixed 16K filler context, and vary only
image translation. No training is involved. Text and generated-token positions
remain unshifted. The optional Target-only arm is a correctness reference.

```bash
export CUDA_VISIBLE_DEVICES=6
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model "$HOME/LMM/lmm_model/Qwen3.5-9B" \
  --draft "$HOME/LMM/lmm_model/Qwen3.5-9B-DFlash" \
  --samples smoke_rope/samples.jsonl \
  --output-dir "smoke_rope/image_sweep_$(date +%Y%m%d_%H%M%S)" \
  --port 31001 \
  --include-target-only \
  --context-tokens 16384 \
  --filler-file smoke_rope/neutral_context.txt \
  --image-shift-sweep 0 128 1024 4096 nearest \
  --max-new-tokens 256 \
  --server-args-json '["--mem-fraction-static","0.7"]'
```

Keep the previously configured CUDA 13 compatibility environment active. The
example selects physical GPU 6, a single GPU. Add `--dry-run` to inspect launch
settings without starting services. The samples and filler files must exist.

`--image-shift-sweep` replaces the four-arm factorial matrix; its values are
integers or `nearest`. It always includes exactly one zero-shift baseline and
deduplicates repeated values. It ignores `--image-shift` and `--text-shift`.
This example runs **6 arms** (Target-only, baseline, image_128, image_1024,
image_4096, image_nearest), so one sample/length gives six measured records.

`nearest` chooses a fixed per-request delta that places the last image Key at
`prompt_length - 1`, immediately before the first Draft Block. Image internal
spacing is preserved, so a 1024-token image still spans 1024 positions. Later
Blocks move away as generation progresses: nearest is not a moving mapping.
With multiple images the entire visual group shares one translation, placing
the final visual Key next to the first Block. This is not position compression.
Its environment switch is `SGLANG_DFLASH_PROMPT_IMAGE_NEAREST=1`; it takes
precedence over the fixed image shift. The runner resets it for every other arm.

CSV/JSONL results now include requested translations, `image_shift_mode`,
`effective_image_shift`, `effective_text_shift`, `eligible_text_tokens`,
`changed_tokens` (summed across measured Prefill chunks), and
`image_end_distance_before/after`. Distances refer to the first Block's first
Query. Baseline/Target-only have zero shifts; unmeasured eligibility/distance
fields are blank. Missing or inconsistent intervention diagnostics fail the run
rather than silently treating an unconfirmed mapping as effective.

Compare verification rounds and acceptance length against baseline on the same
input first. The sweep uses deterministic decoding and one timing measurement
per case; it is a mechanism probe, not a statistically established speedup or
a held-out result. If a favorable offset emerges, freeze it and evaluate it on
new images/questions before drawing a general conclusion.

## Image KV dependence diagnostic

Compare three Draft arms on exactly the same inputs:

* `baseline`: original Key positions, all Context KV visible.
* `image_nearest`: fixed nearest placement, all Context KV visible.
* `image_mask`: original Key positions, image Context KV disallowed in attention.

The mask is applied to Draft `DFlashVerifyInput.custom_mask` only. FlashInfer's
paged prefill wrapper consumes it as a query-major boolean **allow** mask;
False removes that Key from the softmax for every query and head. This is not
zeroing K or V. Text, generated history, and all tokens of the noncausal Draft
Block stay visible. Target prefill, Target verification, cache writes, physical
KV slots and hidden-state capture are unchanged. Visual information may still
flow through Target hidden states at text/generated positions, so this probes
dependence on explicit visual Context KV, not all visual information.

For matched execution, **all three arms** use FlashInfer Draft attention with
custom masks (all-allow for the two controls), both CUDA graph phases disabled,
single GPU, one request at a time, and no radix reuse. Unsupported compact,
sliding-window or causal Draft configurations fail rather than run a different
mask interpretation. This mode is a mechanism probe; its throughput is not
comparable to earlier CUDA graph runs or evidence of a deployable speedup.

```bash
export CUDA_VISIBLE_DEVICES=6
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model "$HOME/LMM/lmm_model/Qwen3.5-9B" \
  --draft "$HOME/LMM/lmm_model/Qwen3.5-9B-DFlash" \
  --samples smoke_rope/samples.jsonl \
  --output-dir "smoke_rope/image_kv_diag_$(date +%Y%m%d_%H%M%S)" \
  --port 31001 \
  --image-kv-diagnostic \
  --include-target-only \
  --context-tokens 0 16384 \
  --filler-file smoke_rope/neutral_context.txt \
  --max-new-tokens 256 \
  --server-args-json '["--mem-fraction-static","0.7"]'
```

Keep the CUDA 13 compatibility environment active. Start with the existing
single image as a smoke check: this command produces **8 measured records**
(4 arms x 2 lengths). Then replace the JSONL with 10–20 distinct image/question
pairs emphasizing OCR, counting, charts and spatial relations. Each line has
`id`, `image` and `question`; optional `answer` is preserved without automatic
scoring. Sixteen thousand filler tokens test textual separation from the
image, not long visual context. Short answers can terminate before 256 tokens;
inspect actual completion lengths and early-round acceptance separately.

`masks_<arm>.jsonl` confirms actual masked-Key counts on each Draft call.
The runner requires the mask count to equal the response's image-token count
in the masked arm (zero in controls), and requires coverage of every measured
verification round. Warmup and readiness traffic are excluded from extracted
files. CSV adds `masked_image_tokens`, `mask_draft_rounds`,
`image_kv_diagnostic`, `round_log`, and `timing_diagnostic_only`.

`rounds_<arm>.jsonl` includes request IDs, the first query's real position,
generated-token count before that round, accepted Draft candidates (excluding
bonus), committed tokens (including bonus), block size, and distance from the
last translated visual Key. Counts describe the verification decision before
scheduler EOS/output-budget truncation, so their sum can exceed the final
completion count. In the masked arm the distance is geometric only: the visual
Keys are not attended. Compare rounds at generated-token positions, not just
round indices, since different acceptance changes block boundaries.

`--round-log` can also be used on an ordinary shift sweep without timing events.
It still copies small tensors from GPU to CPU and perturbs latency. CUDA phase
times remain null unless `--profile` is also enabled. All three options mark
timings as diagnostic-only; leave them off for subsequent speed measurements.

Environment switches (runner resets them per arm):
`SGLANG_DFLASH_ROPE_DIAGNOSTIC=1` selects matched custom-mask execution,
`SGLANG_DFLASH_IMAGE_KV_MASK=1` additionally hides prompt image Keys,
and `SGLANG_DFLASH_ROPE_ROUND_LOG=1` enables acceptance/distance logging.

## Image/text/both: six interventions against one baseline

Use `--modality-kv-diagnostic` instead of `--image-kv-diagnostic` to run:

| Arm | Prompt image KV | Prompt text KV |
| --- | --- | --- |
| `baseline` | Original positions, visible | Original positions, visible |
| `image_nearest` | Fixed nearest translation | Unchanged |
| `image_mask` | Hidden from Draft attention | Unchanged |
| `text_nearest` | Unchanged | Fixed nearest translation of eligible older text |
| `text_mask` | Unchanged | All prompt text hidden from Draft attention |
| `both_nearest` | Image nearest translation | Text nearest translation |
| `both_mask` | Hidden | All prompt text hidden |

All seven Draft arms use the same eager FlashInfer custom-mask path. The old
three-arm image diagnostic and fixed-translation sweep remain available.
These two diagnostic flags are mutually exclusive and cannot be combined with
`--image-shift-sweep`. Fixed `--image-shift`/`--text-shift` values are ignored
by both diagnostic modes.

**Text nearest semantics:** the eligible group excludes image spans and the
last `--text-keep-recent` sequence positions of the prompt (default 256).
Translate this group uniformly until its last Key is at `prompt_length - 1`.
Text in that recent suffix is unchanged. Internal text distances are preserved;
this is not compression. On a long textual suffix, the effective translation
is typically only 256, even when the prompt is 32K long. With keep-recent=0,
the group's last Key usually already occupies that position, so nearest can be
a no-op. Inspect `effective_text_shift`, `eligible_text_tokens`, and
`eligible_text_end_distance_before/after`. These endpoint distances describe
the eligible older text, not the unchanged recent suffix. Translated groups
may share RoPE coordinates with recent text or each other; real cache indices
and attention visibility do not change.

**Text masking semantics:** mask every original prompt position outside image
spans, including special tokens, the question and recent prompt text. This
deliberately does not apply the nearest shift's keep-recent exclusion. Image
masking covers image spans. Both masking removes all prompt Context KV reads,
but generated history and the current Draft Block remain visible. Thus
`both_mask` is not a KV-free drafter, and visual/text information may still
arrive through generated-position Target hidden states.

```bash
CUDA_VISIBLE_DEVICES=7 python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model "$HOME/LMM/lmm_model/Qwen3.5-9B" \
  --draft "$HOME/LMM/lmm_model/Qwen3.5-9B-DFlash" \
  --samples smoke_rope/samples.jsonl \
  --output-dir "smoke_rope/modality_diag_$(date +%Y%m%d_%H%M%S)" \
  --port 31001 \
  --modality-kv-diagnostic \
  --include-target-only \
  --context-tokens 0 1024 4096 8192 16384 32768 \
  --text-keep-recent 256 \
  --filler-file smoke_rope/neutral_context.txt \
  --max-new-tokens 256 \
  --server-args-json '["--mem-fraction-static","0.7"]'
```

Use the existing virtual environment/CUDA 13 compatibility setup and localhost
proxy exclusions. This command uses physical GPU 7 only. One sample produces
**48 measured records**: 7 Draft arms plus Target-only, times 6 context lengths.
Without `--include-target-only`, it produces 42 records. Servers are started
and stopped automatically, once per arm.

CSV/JSONL adds `modality_kv_diagnostic`, `text_shift_mode`, `text_keep_recent`,
`masked_text_tokens` and `masked_prompt_tokens`. Runtime validation checks image,
text and total masked counts on every measured Draft call. The log marker
remains `DFLASH_IMAGE_MASK` for compatibility; extracted `masks_*.jsonl` now
contains both modalities. Round logs also include eligible-text endpoint
distances and `text_kv_masked`.

Per-case comparisons (same sample, context length, image size) are included in
each measured row:

* `baseline_spec_accept_length`: the matching baseline's acceptance length.
* `spec_accept_length_change_pct`: `(arm / baseline - 1) * 100`; positive is better.
* `spec_verify_ct_change`: `arm_verify - baseline_verify`; negative is fewer calls.

Baseline changes are zero; Target-only changes are blank. These comparisons
are not aggregated across samples and do not establish statistical significance.
Always inspect `matches_baseline` and `matches_target` alongside acceptance.
Known output mismatches in the exploratory image run still require investigation;
this extension does not fix or waive the correctness check. A matrix saves
results, then exits unsuccessfully if any output mismatch was found. Diagnostic
timings remain unsuitable for claiming production speedups.

Additional environment switches, explicitly reset per arm:
`SGLANG_DFLASH_PROMPT_TEXT_NEAREST=1` and `SGLANG_DFLASH_TEXT_KV_MASK=1`.

## TextVQA acceptance-only run

`load_samples` accepts either `question` or `prompt`. If both are supplied they
must agree. Questions are passed through unchanged; blank reference answers
are allowed. `textvqa_11.jsonl` contains the supplied 11 questions on 6 images,
using the original server-local image paths. Those images are not in this repo.

For this experiment omit `--include-target-only`: every arm remains DFlash,
and Target verification within each arm supplies acceptance decisions. Use
`--allow-output-mismatch` to retain all identity flags without failing the
completed matrix. This does not assert distributional correctness; it scopes
the run to the requested acceptance-only diagnostic.

`--max-new-tokens 256` alone is only an upper bound: VQA answers often end much
earlier. Add `--ignore-eos` for a fixed 256-token probe. This continues after
natural EOS and can produce repetitive or otherwise unnatural continuations;
interpret its acceptance as the fixed-length probe, not VQA answer accuracy.
Neither this flag nor the runner adds an explanation request to the question.
CSV records `ignore_eos`, `max_new_tokens`, actual `completion_tokens`, and
`fixed_output_budget_met` so unexpectedly short generations are visible.
Leave `--ignore-eos` off for a separate natural-answer experiment.

```bash
git pull --ff-only origin research/vlm-dflash-rope-shift
python examples/runtime/dflash_rope_shift/test_cpu.py

CUDA_VISIBLE_DEVICES=7 python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model "$HOME/LMM/lmm_model/Qwen3.5-9B" \
  --draft "$HOME/LMM/lmm_model/Qwen3.5-9B-DFlash" \
  --samples examples/runtime/dflash_rope_shift/textvqa_11.jsonl \
  --output-dir "smoke_rope/textvqa_diag_$(date +%Y%m%d_%H%M%S)" \
  --port 31001 \
  --modality-kv-diagnostic \
  --context-tokens 0 1024 4096 8192 16384 32768 \
  --text-keep-recent 256 \
  --filler-file smoke_rope/neutral_context.txt \
  --max-new-tokens 256 \
  --ignore-eos \
  --allow-output-mismatch \
  --server-args-json '["--mem-fraction-static","0.7"]'
```

Keep the established virtual environment, CUDA compatibility variables and
localhost proxy exclusions active. This gives **462 formal measurements**
(11 questions x 6 lengths x 7 DFlash arms), with 8-token warmups excluded.
For another prepared JSONL, replace only `--samples`. Results remain per-question
in `results.csv`/`results.jsonl` and per-round in `rounds_*.jsonl`.

The runner also writes `summary.csv`, grouping by context length, image size and
arm. Each of its 42 rows summarizes 11 questions:

* `pooled_accept_length`: total completion tokens / total verification calls.
* `pooled_accept_length_change_pct`: that ratio relative to paired baseline.
* `paired_accept_change_mean_pct`: mean of per-question relative changes.
* `win_count`, `tie_count`, `loss_count`: question-level comparisons to baseline.
* Sample count, unique-image hashes, output-length range, fixed-budget counts
  and output identity mismatch counts.

Questions sharing one image are not independent images. This small set supports
an initial TextVQA mechanism probe; it does not represent overall TextVQA
accuracy. Diagnostic throughput still must not be treated as serving speedup.

You can summarize an existing CSV locally without model loading:

```bash
python examples/runtime/dflash_rope_shift/summarize_results.py RUN/results.csv
```

## Generated Target-feature history window experiment

This experiment tests whether the recovery after masking the entire prompt
needs a long generated Target-feature history, or just a few recent positions.
It does not change RoPE, the Target, or the draft block. All arms use the same
eager FlashInfer custom-mask path, with no prefix reuse and one request at a time.

| Arm | Prompt image/text Context KV | Generated-history Context KV |
| --- | --- | --- |
| `baseline` | All visible | All visible |
| `both_mask` | All masked | All visible |
| `both_mask_gen1` | All masked | Last 1 position |
| `both_mask_gen4` | All masked | Last 4 positions |
| `both_mask_gen16` | All masked | Last 16 positions |

For prompt length P and current draft prefix Q, the generated history is
`[P, Q)`. A window N exposes only `[max(P, Q-N), Q)`. The complete current draft
block `[Q, Q+block_size)` remains visible, including its known anchor token;
it is excluded from the window count. Therefore even the optional window 0
is not a draft without the current token's information. The first round has
zero generated-history positions for every window. Early rounds retain
`min(N, Q-P)` positions. All generated KV is still materialized and cached:
this is an information-dependence experiment, not a memory-saving implementation.

Run in the established `.venv-rope` environment with CUDA compatibility variables
and localhost proxy exclusions active:

```bash
git pull --ff-only origin research/vlm-dflash-rope-shift
python examples/runtime/dflash_rope_shift/test_cpu.py

CUDA_VISIBLE_DEVICES=7 python examples/runtime/dflash_rope_shift/run_matrix.py \
  --model "$HOME/LMM/lmm_model/Qwen3.5-9B" \
  --draft "$HOME/LMM/lmm_model/Qwen3.5-9B-DFlash" \
  --samples examples/runtime/dflash_rope_shift/textvqa_11.jsonl \
  --output-dir "smoke_rope/generated_kv_diag_$(date +%Y%m%d_%H%M%S)" \
  --port 31001 \
  --generated-kv-diagnostic \
  --generated-kv-windows 1 4 16 \
  --context-tokens 0 1024 4096 8192 16384 32768 \
  --filler-file smoke_rope/neutral_context.txt \
  --max-new-tokens 256 --ignore-eos --allow-output-mismatch \
  --server-args-json '["--mem-fraction-static","0.7"]'
```

This runs **330 formal measurements**: 11 questions on 6 images x 6 context
lengths x 5 arms, each with a separate 8-token warmup that is not scored.
The context lengths refer to unrelated filler after the image, not the full
prompt length. There is no Target-only arm and output identity differences
are recorded without failing the completed acceptance-only matrix.
`--generated-kv-windows 0 1 4 16` optionally adds a sixth arm with no generated
history; `--dry-run` prints plans without model loading or serving.

Read `summary.csv` for pooled completion-tokens/verification acceptance versus
baseline and `vs_both_mask_change_pct` versus the full generated-history control.
Read `round_summary.csv` for accepted draft candidates per round in `first_round`
and position ranges `0-31`, `32-63`, `64-127`, `128-255` (and `256+` if present).
Ranges use the logged number of generated Target-feature positions before the
round, not the round ordinal. `first_round` duplicates the first entry in `0-31`
as a separate diagnostic; do not add those rows together. Raw round counts
exclude the bonus from accepted candidates and can include a final over-cap
verification block, so they are not capped output throughput or exact short-run
measurements. Each stage reports contributing samples and rounds.

`masks_*.jsonl` confirms the actual visible/masked generated-history count,
the unchanged visible block count, and complete prompt masking each round.
`results.csv` also reports min/max visible generated-history counts. The runner
fails if a requested window or block visibility is not confirmed in every
measured verification round. Baseline resets the window to -1 (all history),
even if an environment variable was inherited from another experiment.

Interpretation: if window 1 closely follows `both_mask`, one recent position is
sufficient for recovery on these probes; if 4/16 improve over 1, recovery needs
more history. A gap to full baseline can still come from inaccessible prompt
features. These tests do not prove that visual information specifically is
stored in the latest hidden state. The small, shared-image dataset and forced
post-EOS continuation remain the same limitations as the previous probe.

Rebuild stage summaries from a completed run without GPUs:

```bash
python examples/runtime/dflash_rope_shift/summarize_rounds.py RUN/results.csv
```
