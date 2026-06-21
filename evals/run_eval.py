"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)

    t0 = time.monotonic()
    agent_error = None
    agent_payload: dict = {}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                agent_url,
                json={
                    "question": question["question"],
                    "db": db_id,
                    "tags": {
                        "phase": "phase5_eval",
                        "db_id": db_id,
                    },
                },
            )
        resp.raise_for_status()
        agent_payload = resp.json()
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"

    latency = time.monotonic() - t0
    pred_sql = agent_payload.get("sql", "")
    pred_ok, pred_rows, pred_error = run_sql(db_id, pred_sql) if pred_sql else (False, None, "missing SQL")
    correct = gold_ok and pred_ok and matches(gold_rows, pred_rows)

    iteration_results: list[dict] = []
    attempt_no = 0
    for event in agent_payload.get("history", []):
        if event.get("node") not in {"generate_sql", "revise"}:
            continue
        attempt_no += 1
        sql = event.get("sql", "")
        ok, rows, err = run_sql(db_id, sql) if sql else (False, None, "missing SQL")
        iteration_results.append({
            "iteration": attempt_no,
            "sql": sql,
            "exec_ok": ok,
            "correct": gold_ok and ok and matches(gold_rows, rows),
            "error": err,
        })

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_error,
        "pred_sql": pred_sql,
        "pred_exec_ok": pred_ok,
        "pred_error": pred_error,
        "correct": correct,
        "agent_ok": agent_payload.get("ok", False),
        "agent_error": agent_payload.get("error") or agent_error,
        "iterations": agent_payload.get("iterations", 0),
        "latency_seconds": latency,
        "history": agent_payload.get("history", []),
        "iteration_results": iteration_results,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    agent_ok = sum(1 for r in results if r.get("agent_ok"))
    revised = sum(1 for r in results if (r.get("iterations") or 0) > 1)
    latencies = sorted(r["latency_seconds"] for r in results if "latency_seconds" in r)
    max_iter = max((r.get("iterations") or 0 for r in results), default=0)

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    per_iteration: dict[str, dict] = {}
    for i in range(1, max_iter + 1):
        iter_correct = 0
        iter_scored = 0
        for r in results:
            attempts = r.get("iteration_results", [])
            if not attempts:
                continue
            carried = attempts[min(i, len(attempts)) - 1]
            iter_scored += 1
            if carried.get("correct"):
                iter_correct += 1
        per_iteration[str(i)] = {
            "scored": iter_scored,
            "correct": iter_correct,
            "pass_rate": (iter_correct / iter_scored) if iter_scored else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "pass_rate": (correct / total) if total else 0.0,
        "agent_ok": agent_ok,
        "agent_ok_rate": (agent_ok / total) if total else 0.0,
        "revised": revised,
        "revision_rate": (revised / total) if total else 0.0,
        "avg_iterations": (
            sum(r.get("iterations") or 0 for r in results) / total
        ) if total else 0.0,
        "latency_p50": pct(0.50),
        "latency_p95": pct(0.95),
        "per_iteration": per_iteration,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
