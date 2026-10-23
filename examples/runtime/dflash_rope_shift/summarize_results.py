"""Summarize paired DFlash arms: python summarize_results.py RUN/results.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def truth(value):
    return value is True or value == "True"


def summarize(rows):
    """Pool tokens/verification calls; also report per-question wins and losses.

    Groups have the same filler length and image size. Questions on the same
    image are counted separately, with unique-image counts reported explicitly.
    No timing speedup or task accuracy is inferred from these diagnostics.
    """
    rows = [r for r in rows if r["arm"] != "target_only"]

    def case(row):
        return (
            str(row["id"]),
            int(row["context_tokens_requested"]),
            int(row["image_max_edge"]),
        )

    baselines = {case(r): r for r in rows if r["arm"] == "baseline"}
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["arm"], *case(row))
        if key in seen:
            raise ValueError(f"Duplicate measurement: {key}")
        seen.add(key)
        if case(row) not in baselines:
            raise ValueError(f"Missing paired baseline: {case(row)}")
        groups[
            (
                int(row["context_tokens_requested"]),
                int(row["image_max_edge"]),
                row["arm"],
            )
        ].append(row)

    result = []
    for (context, edge, arm), group in sorted(groups.items()):
        controls = [baselines[case(r)] for r in group]
        tokens = [int(r["completion_tokens"]) for r in group]
        rounds = [int(r["spec_verify_ct"]) for r in group]
        if min(rounds) <= 0:
            raise ValueError("Each DFlash case must contain verification calls")
        base_tokens = sum(int(r["completion_tokens"]) for r in controls)
        base_rounds = sum(int(r["spec_verify_ct"]) for r in controls)
        pooled = sum(tokens) / sum(rounds)
        base_pooled = base_tokens / base_rounds
        changes = []
        for row, base in zip(group, controls, strict=True):
            value = int(row["completion_tokens"]) / int(row["spec_verify_ct"])
            reference = int(base["completion_tokens"]) / int(base["spec_verify_ct"])
            changes.append(100 * (value / reference - 1))
        result.append(
            {
                "context_tokens_requested": context,
                "image_max_edge": edge,
                "arm": arm,
                "samples": len(group),
                "unique_image_hashes": len({r["image_sha256"] for r in group}),
                "completion_tokens_total": sum(tokens),
                "completion_tokens_min": min(tokens),
                "completion_tokens_max": max(tokens),
                "verify_ct_total": sum(rounds),
                "pooled_accept_length": pooled,
                "baseline_pooled_accept_length": base_pooled,
                "pooled_accept_length_change_pct": 100 * (pooled / base_pooled - 1),
                "paired_accept_change_mean_pct": sum(changes) / len(changes),
                "win_count": sum(x > 1e-9 for x in changes),
                "tie_count": sum(abs(x) <= 1e-9 for x in changes),
                "loss_count": sum(x < -1e-9 for x in changes),
                "baseline_output_mismatches": sum(
                    r.get("matches_baseline") in (False, "False") for r in group
                ),
                "target_output_mismatches": sum(
                    r.get("matches_target") in (False, "False") for r in group
                ),
                "target_comparisons": sum(
                    r.get("matches_target") in (True, False, "True", "False")
                    for r in group
                ),
                "fixed_output_budget_met_count": sum(
                    truth(r.get("fixed_output_budget_met")) for r in group
                ),
                "timing_diagnostic_only": any(
                    truth(r.get("timing_diagnostic_only")) for r in group
                ),
            }
        )
    return result


def write_summary(rows, path):
    if not rows:
        raise ValueError("No DFlash measurements to summarize")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.results.open() as file:
        rows = summarize(list(csv.DictReader(file)))
    output = args.output or args.results.with_name("summary.csv")
    write_summary(rows, output)
    print(f"Saved {len(rows)} arm/context summaries to {output}")
