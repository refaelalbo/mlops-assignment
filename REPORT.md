# LLM Inference + Observability Report

> Submission note: the final H100 run with `Qwen/Qwen3-30B-A3B-Instruct-2507`
> was not completed before this snapshot. The implementation, dashboard,
> tracing, eval harness, and SLO loop were validated locally with
> `Qwen/Qwen3-0.6B` on an RTX 3060 laptop GPU.

## Serving Configuration

Target H100 model: `Qwen/Qwen3-30B-A3B-Instruct-2507`

Local validation model: `Qwen/Qwen3-0.6B`

Local vLLM command used for development validation:

```bash
uv run vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Configuration rationale:

- `--dtype bfloat16`: chosen for GPU-native BF16 serving and to match the intended H100 deployment dtype.
- `--max-model-len 4096`: covers the 1.5K-3K token prompt shape plus short SQL output while preserving KV-cache capacity.
- `--max-num-seqs 64`: baseline concurrency setting used to expose queueing behavior during local load tests.
- `--max-num-batched-tokens 8192`: keeps prompt prefill batching enabled for the repeated schema-heavy workload.
- `--gpu-memory-utilization 0.90`: maximizes local KV-cache capacity; on the 6 GB laptop GPU this required checking free memory before startup.
- `--enable-prefix-caching`: schema and instruction prefixes repeat heavily across the eval/load-test workload.

Manual vLLM validation: local development evidence is saved at `screenshots/vllm_manual_query.png`. Final H100 validation remains pending because the final H100 run was not completed before this submission snapshot.

## Baseline Eval Results

Baseline run: `results/eval_baseline.json`

- Total questions: 30
- Correct: 2 on the local `Qwen/Qwen3-0.6B` baseline run
- Execution accuracy: 6.7% on the local baseline run
- Pass rate after initial generation: 3.8% among questions with emitted SQL attempts
- Pass rate after one revise: 3.8% among questions with emitted SQL attempts
- Pass rate after two revises/final allowed iteration: 7.7% among questions with emitted SQL attempts
- Average iterations: 1.9
- Revision rate: 53.3%

The evaluation compares execution results rather than SQL text. For each item, the agent SQL and gold SQL are executed against the same SQLite DB, then rows are canonicalized and sorted before comparison. This avoids penalizing SQL that is syntactically different but result-equivalent.

Baseline observation: this local 0.6B run is useful for validating the evaluation harness and tracing, but it is not representative of the final H100 target model. The most common failures were wrong schema linking, repeated invalid revisions, zero-row answers, and several agent HTTP 500s from model outputs that still need hardening. The Ajax superpowers example is a positive case where the verifier/revise loop improved the answer from an ID-only query to the correct `power_name` join.

## SLO and Tuning

Target SLO: P95 end-to-end agent latency under 5 seconds at 10+ RPS over a 5-minute window.

Baseline load result:

- Run: `results/load_baseline.json`
- Requested RPS: 2.0
- Achieved RPS: 1.33 over 180.0 seconds wall-clock including request drain
- Total requests: 240
- Successful requests: 196
- P50 agent latency: 2.44 seconds
- P95 agent latency: 8.67 seconds
- Error rate: 18.3% (40 HTTP 500 responses, 4 client errors)
- Grafana evidence: `screenshots/grafana_before.png`

Overload probe:

- Run: `results/load_baseline_rps4.json`
- Requested RPS: 4.0
- Achieved RPS: 2.67 over 180.0 seconds wall-clock including request drain
- Successful requests: 282 out of 480
- P95 agent latency: 27.86 seconds
- Failure mode: the local 6 GB GPU stack showed backpressure through higher request latency, timeouts, HTTP 500s, and client errors.

Iteration log:

1. Saw P95 latency at 8.67 seconds even at 2 RPS -> hypothesized the local model server was queueing too aggressively for the RTX 3060 laptop GPU -> reduced serving pressure and reran the same 2 RPS load -> P95 improved to 6.73 seconds.
2. Saw the 4 RPS probe degrade to 27.86 seconds P95 with many client errors -> treated this as above local capacity rather than a useful target for this hardware -> kept the final local comparison at the same 2 RPS as baseline.
3. Saw the first post-tuning attempt produce no Grafana traffic -> checked the result JSON and found every request failed with `Cannot connect to host localhost:8001` -> restarted the Agent API and reran the test to produce valid after-tuning evidence.

Final load result:

- Run: `results/load_after_tuning.json`
- Requested RPS: 2.0
- Achieved RPS: 1.47 over 162.8 seconds wall-clock including request drain
- Total requests: 240
- Successful requests: 200
- P50 agent latency: 2.14 seconds
- P95 agent latency: 6.73 seconds
- Error rate: 16.7% (40 HTTP 500 responses, 0 client errors)
- KV-cache usage: Grafana showed active KV-cache movement during the run, with a visibly lower after-run burst than the overloaded probe.
- Verdict: local tuning improved P95 latency and removed client connection errors, but the local laptop run still missed the assignment SLO. Final H100/Qwen3-30B validation was not completed before this submission snapshot.

The before/after Grafana evidence is saved in `screenshots/grafana_before.png` and `screenshots/grafana_after.png`.

## Agent Value

The agent used a generate -> execute -> verify -> revise loop capped at 3 total SQL attempts. The loop helped on at least one manually inspected Phase 3 case. In the local 0.6B baseline eval, final pass rate improved from 3.8% after the first emitted attempt to 7.7% after the final allowed iteration among questions with emitted SQL attempts.

Phase 3 manual evidence shows the loop adding real value on the question "List down Ajax's superpowers." The initial SQL executed but returned only `power_id` values, which did not answer the request for superpower names. The verifier rejected that result, the revise step attempted a joined query, the verifier rejected a zero-row revision, and the final revision produced a correct join from `superhero` to `hero_power` to `superpower`, selecting `power_name`. The final answer returned `Agility`, `Super Strength`, `Super Speed`, `Heat Generation`, and `Power Suit` with `iterations=3` and `ok=true`. Raw evidence is saved in `results/phase3_ajax_revise_example.json`; terminal evidence is saved in `screenshots/phase3_agent_revise.png`.

Evidence: the local eval pass rate moved from 3.8% after initial generation to 7.7% after the final allowed iteration among questions with emitted SQL attempts. The revise step was most useful for the Ajax superpowers case, where it corrected an ID-only answer into a `power_name` join. Langfuse traces show the expected waterfall with `generate_sql`, `verify`, and when needed `revise`; screenshots are saved as `screenshots/langfuse_trace.png` and `screenshots/langfuse_tags.png`.

## Quality After Tuning

Post-tuning run: `results/eval_after_tuning.json`

- Correct: 2 out of 30 on the local `Qwen/Qwen3-0.6B` post-tuning run
- Execution accuracy: 6.7%
- Change versus baseline: unchanged top-line pass rate, from 2/30 to 2/30
- Average iterations: 2.0
- Revision rate: 56.7%

Quality survived the local tuning pass at the aggregate level: pass rate stayed at 6.7%. The tuning did not improve correctness, but it also did not create a measured regression. Remaining quality failures are still dominated by schema-linking mistakes, empty-result SQL, SQL execution errors, and some HTTP 500s from brittle model outputs.

## What I Would Do With More Time

1. Add schema-linking before SQL generation so the model receives only the relevant tables and columns instead of the full schema context.
2. Add structured JSON output constraints for verifier decisions, with strict parsing and retry on invalid verifier output.
3. Split eval results by failure type: SQL syntax error, wrong table/column, wrong aggregation, wrong filter, empty result, and timeout.
4. Add prompt-prefix cache hit-rate monitoring to prove whether prefix caching is actually helping the repeated schema/instruction workload.
5. Try speculative decoding only if decode latency, not queueing or prefill, is the measured bottleneck.
