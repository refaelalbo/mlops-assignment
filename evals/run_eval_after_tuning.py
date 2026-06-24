"""Eval runner for the after-tuning pass.

# Goal: Run the same evaluation logic after a real tuning/change step.
# Why: Baseline and after-tuning quality must be comparable; only the output
# filename/run label should differ.

This intentionally reuses evals/run_eval.py so the scoring logic stays identical
between baseline and after-tuning runs. The only default difference is the output
file: results/eval_after_tuning.json.

Run after changing prompts, verifier rules, repair logic, model settings, or
serving parameters and restarting the agent:

    uv run python evals/run_eval_after_tuning.py --agent-url http://localhost:8001/answer
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evals.run_eval import AGENT_URL_DEFAULT, DEFAULT_EVAL_FILE, eval_one, summarize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_FILE = ROOT / "results" / "eval_after_tuning.json"


def main() -> None:
    # Goal: Mirror run_eval.py CLI while changing the default output path.
    # Why: Operational commands stay familiar between baseline and after-tuning.
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    # Goal: Load the exact same eval questions as the baseline run by default.
    # Why: A tuning comparison is invalid if the question set changes.
    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")
    print("Writing after-tuning eval results.")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        # Goal: Keep progress output identical to baseline eval.
        # Why: It helps compare terminal logs across runs.
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        # Goal: Mark this artifact as post-tuning.
        # Why: Later report automation or manual review can distinguish files.
        "run_type": "after_tuning",
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    # Goal: Save the same raw detail as baseline plus the run_type label.
    # Why: Accuracy regressions need per-question inspection.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
