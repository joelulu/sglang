"""Summarize measured Draft rounds by generated position, not round ordinal.

Usage: python summarize_rounds.py RUN/results.csv
Raw accepted candidates exclude the bonus token. The last round can extend past
the returned output cap, so these stage means are not capped output throughput.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from summarize_results import write_summary


def summarize_rounds(rows, directory):
    groups = defaultdict(list)
    seen = set()
    arm_logs = {}
    for row in rows:
        if row["arm"] == "target_only":
            continue
        rid = row["rid"]
        if rid in seen:
            raise ValueError(f"Duplicate measured rid: {rid}")
        seen.add(rid)
        arm = row["arm"]
        if arm not in arm_logs:
            by_rid = defaultdict(list)
            with (directory / f"rounds_{arm}.jsonl").open() as file:
                for line in file:
                    entry = json.loads(line)
                    if len(entry["rids"]) != 1:
                        raise ValueError("Round summaries require single-request logs")
                    by_rid[entry["rids"][0]].append(entry)
            arm_logs[arm] = by_rid
        entries = arm_logs[arm][rid]
        if len(entries) != int(row["spec_verify_ct"]):
            raise ValueError(f"Missing verification-round records for {rid}")
        previous = -1
        for index, entry in enumerate(entries):
            position = entry["generated_tokens_before"][0]
            if position <= previous or (index == 0 and position != 0):
                raise ValueError(f"Invalid generated-position order for {rid}")
            previous = position
            stage = next(
                (
                    f"{low}-{high - 1}"
                    for low, high in [(0, 32), (32, 64), (64, 128), (128, 256)]
                    if low <= position < high
                ),
                "256+",
            )
            stages = [stage, "first_round"] if index == 0 else [stage]
            for label in stages:
                groups[
                    (
                        int(row["context_tokens_requested"]),
                        int(row["image_max_edge"]),
                        row["arm"],
                        label,
                    )
                ].append(
                    {
                        "rid": rid,
                        "accepted": entry["accepted_draft_tokens"][0],
                        "committed": entry["committed_tokens"][0],
                    }
                )
    result = []
    for (context, edge, arm, stage), entries in sorted(groups.items()):
        result.append(
            {
                "context_tokens_requested": context,
                "image_max_edge": edge,
                "arm": arm,
                "stage_by_generated_position": stage,
                "samples": len({e["rid"] for e in entries}),
                "rounds": len(entries),
                "accepted_draft_tokens_total": sum(e["accepted"] for e in entries),
                "accepted_draft_tokens_mean": sum(e["accepted"] for e in entries)
                / len(entries),
                "committed_tokens_mean": sum(e["committed"] for e in entries)
                / len(entries),
                "zero_accept_rounds": sum(e["accepted"] == 0 for e in entries),
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with args.results.open() as file:
        rows = list(csv.DictReader(file))
    output = args.results.with_name("round_summary.csv")
    write_summary(summarize_rounds(rows, args.results.parent), output)
    print(f"Saved per-stage acceptance to {output}")
