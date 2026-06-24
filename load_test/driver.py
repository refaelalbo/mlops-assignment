"""Async load driver for the agent endpoint.

# Goal: Generate repeatable HTTP traffic against the /answer endpoint.
# Why: Phase 6 needs latency/error evidence under load, not only single-request
# manual checks.

Samples questions from load_test/perf_pool.jsonl and fires them at the
agent at the requested RPS for the requested duration, recording per-
request latency and outcome.

Run:
    uv run python load_test/driver.py --rps 8 --duration 300

Writes a JSON file (default results/load_test.json) with summary + raw
per-request data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"
DEFAULT_OUT = ROOT / "results" / "load_test.json"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    question: dict,
    results: list[dict],
) -> None:
    # Goal: Convert one perf-pool question into the API request shape.
    # Why: The agent endpoint expects "db", while eval rows store "db_id".
    payload = {"question": question["question"], "db": question["db_id"]}
    t0 = time.monotonic()
    status = "ok"
    err: str | None = None
    try:
        # Goal: Fire one request and read the body to release the connection.
        # Why: Load tests should include full HTTP response handling, not just
        # socket open time.
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            await resp.read()
            if resp.status != 200:
                status = "http_error"
                err = f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        # Goal: Separate server slowness from other client-side failures.
        # Why: Timeouts are an important SLO/stability signal.
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        status = "client_error"
        err = f"{type(e).__name__}: {e}"
    # Goal: Append one raw observation for later percentile/error summaries.
    # Why: Keeping raw results allows more analysis than only aggregate metrics.
    results.append({
        "latency_seconds": time.monotonic() - t0,
        "status": status,
        "error": err,
    })


async def drive(args: argparse.Namespace) -> None:
    # Goal: Fail early if data prep did not create the load-test pool.
    # Why: A load test without questions is a setup problem, not a service result.
    if not PERF_POOL.exists():
        raise SystemExit(f"{PERF_POOL} not found - run scripts/load_data.py first")
    questions = [json.loads(line) for line in PERF_POOL.read_text().splitlines() if line.strip()]
    if not questions:
        raise SystemExit(f"{PERF_POOL} is empty")

    # Goal: Use deterministic sampling from the perf pool.
    # Why: Baseline and after-tuning runs should send comparable traffic.
    rnd = random.Random(0)
    results: list[dict] = []
    interval = 1.0 / args.rps

    # Goal: Remove aiohttp's connector limit.
    # Why: The requested RPS schedule should control concurrency, not the client
    # connector's default cap.
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.monotonic()
        deadline = start + args.duration
        tasks: list[asyncio.Task] = []
        next_fire = start
        while time.monotonic() < deadline:
            # Goal: Schedule requests at fixed intervals.
            # Why: This approximates open-loop load at the requested RPS.
            q = rnd.choice(questions)
            tasks.append(asyncio.create_task(fire_one(session, args.agent_url, q, results)))
            next_fire += interval
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        # let in-flight finish (cap drain at 60s)
        if tasks:
            # Goal: Include completed in-flight requests but avoid waiting forever.
            # Why: Slow tail requests should influence wall-clock, but the driver
            # needs a bounded shutdown.
            await asyncio.wait(tasks, timeout=60.0)
        wall = time.monotonic() - start

    # Goal: Compute latency percentiles only over successful requests.
    # Why: Failed requests have short/odd latencies that would distort service
    # latency metrics.
    latencies = sorted(r["latency_seconds"] for r in results if r["status"] == "ok")

    def pct(p: float) -> float:
        # Goal: Return percentile latency from sorted successful latencies.
        # Why: P95/P99 expose tail behavior better than averages.
        if not latencies:
            return float("nan")
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    summary = {
        # Goal: Report requested and achieved load separately.
        # Why: The assignment SLO depends on actual achieved throughput.
        "requested_rps": args.rps,
        "duration_seconds": args.duration,
        "wall_clock_seconds": wall,
        "total_requests": len(results),
        "achieved_rps": (len(results) / wall) if wall > 0 else 0.0,
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "timeouts": sum(1 for r in results if r["status"] == "timeout"),
        "http_errors": sum(1 for r in results if r["status"] == "http_error"),
        "client_errors": sum(1 for r in results if r["status"] == "client_error"),
        "latency_p50": pct(0.50),
        "latency_p95": pct(0.95),
        "latency_p99": pct(0.99),
        "latency_max": latencies[-1] if latencies else float("nan"),
    }

    # Goal: Save aggregate metrics plus raw request outcomes.
    # Why: The final report uses summary values, while troubleshooting uses
    # individual errors and latencies.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


def main() -> None:
    # Goal: Expose load parameters as CLI flags.
    # Why: The same driver can run smoke tests, baseline loads, and tuning loads.
    p = argparse.ArgumentParser()
    p.add_argument("--rps", type=float, default=8.0, help="target requests/second")
    p.add_argument("--duration", type=int, default=300, help="seconds to drive load")
    p.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    asyncio.run(drive(args))


if __name__ == "__main__":
    main()
