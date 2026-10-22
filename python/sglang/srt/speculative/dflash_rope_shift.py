"""Fixed, prompt-only RoPE translations for controlled DFlash experiments.

This module is deliberately independent of torch. Positions here are sequence
indices, not Target M-RoPE coordinates or physical KV slots. A single translation
per modality is chosen from the *full* prompt, so chunked prefill and replay of
generated tokens use the same mapping. No query-relative re-rotation is needed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRopeShift:
    image_shift: int = 0
    text_shift: int = 0
    text_keep_recent: int = 256

    def __post_init__(self):
        if self.text_keep_recent < 0:
            raise ValueError("DFlash text_keep_recent must be nonnegative")

    @property
    def enabled(self):
        return self.image_shift != 0 or self.text_shift != 0

    def chunk_offsets(self, prompt_len, start, length, image_spans):
        """Return (offsets, diagnostics); image span endpoints are inclusive.

        Requested translations are bounded once per modality to [0, prompt_len).
        This preserves all within-group position differences, unlike per-token
        clamping. Diagnostics report requested AND effective translations. The
        recent-text exclusion is a fixed suffix of the prompt, not a moving
        decode window. Special tokens outside image spans belong to text.
        """
        if prompt_len < 1 or start < 0 or length < 0:
            raise ValueError("Invalid prompt/chunk bounds")
        image = [False] * prompt_len
        for lo, hi in image_spans:
            if not 0 <= lo <= hi < prompt_len:
                raise ValueError(f"Image span {(lo, hi)} outside prompt {prompt_len}")
            if any(image[lo : hi + 1]):
                raise ValueError(f"Overlapping image span {(lo, hi)}")
            image[lo : hi + 1] = [True] * (hi - lo + 1)

        image_positions = [p for p in range(prompt_len) if image[p]]
        text_positions = [
            p for p in range(max(0, prompt_len - self.text_keep_recent)) if not image[p]
        ]

        def bounded_shift(requested, positions):
            if not positions:
                return 0
            return min(max(requested, -positions[0]), prompt_len - 1 - positions[-1])

        image_delta = bounded_shift(self.image_shift, image_positions)
        text_delta = bounded_shift(self.text_shift, text_positions)
        offsets = []
        for p in range(start, start + length):
            if p >= prompt_len:
                offsets.append(0)  # re-prefill/retraction must not shift generated text
            elif image[p]:
                offsets.append(image_delta)
            elif p < prompt_len - self.text_keep_recent:
                offsets.append(text_delta)
            else:
                offsets.append(0)
        return offsets, {
            "prompt_tokens": prompt_len,
            "image_tokens": len(image_positions),
            "eligible_text_tokens": len(text_positions),
            "requested_image_shift": self.image_shift,
            "requested_text_shift": self.text_shift,
            "effective_image_shift": image_delta,
            "effective_text_shift": text_delta,
            "text_keep_recent": self.text_keep_recent,
            "chunk_start": start,
            "chunk_tokens": length,
            "changed_tokens": sum(delta != 0 for delta in offsets),
        }
