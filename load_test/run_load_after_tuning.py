"""Load runner for the after-tuning pass.

# Goal: Run the same load driver after a real serving/agent tuning change.
# Why: Phase 6 requires a fair before/after comparison with identical traffic
# generation and a recorded tuning note.

This reuses load_test/driver.py so the load-generation logic stays identical
between baseline and after-tuning runs. The default output file is
results/load_after_tuning.json.

Run only after making a real tuning change and restarting the affected service:

    uv run python load_test/run_load_after_tuning.py \
      --agent-url http://localhost:8001/answer \
      --rps 2 \
      --duration 120 \
      --tuning-note "restarted vLLM with max-num-seqs=32 and max-num-batched-tokens=4096"
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from load_test.driver import AGENT_URL_DEFAULT, drive

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "load_after_tuning.json"


def main() -> None:
    # Goal: Keep load-test parameters configurable but require a tuning note.
    # Why: "After tuning" is only meaningful if a real change is documented.
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=float, default=2.0, help="target requests/second")
    parser.add_argument("--duration", type=int, default=120, help="seconds to drive load")
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--tuning-note",
        required=True,
        help="Short description of the real tuning change made before this run.",
    )
    args = parser.parse_args()

    print("Writing after-tuning load results.")
    print(f"Tuning note: {args.tuning_note}")
    # Goal: Reuse the baseline load engine exactly.
    # Why: Any metric change should come from tuning, not a different driver.
    asyncio.run(drive(args))

    # Goal: Annotate the output JSON after the shared driver writes it.
    # Why: The report can identify the run type and the tuning action later.
    data = json.loads(args.out.read_text())
    data["run_type"] = "after_tuning"
    data["tuning_note"] = args.tuning_note
    args.out.write_text(json.dumps(data, indent=2))
    print(f"Annotated {args.out} with run_type and tuning_note")


if __name__ == "__main__":
    main()
