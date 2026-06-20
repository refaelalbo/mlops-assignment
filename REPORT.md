# LLM Inference + Observability Report

> Draft status: this report is prepared for the assignment structure, but every
> `TODO_FROM_H100_RUN` value must be replaced with real measurements from the
> final H100 run before submission.

## Serving Configuration

Final model: `Qwen/Qwen3-30B-A3B-Instruct-2507`

Final vLLM command:

```bash
uv run vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len TODO_FROM_H100_RUN \
  --max-num-seqs TODO_FROM_H100_RUN \
  --max-num-batched-tokens TODO_FROM_H100_RUN \
  --gpu-memory-utilization TODO_FROM_H100_RUN \
  --enable-prefix-caching
```

Configuration rationale:

- `--dtype bfloat16`: chosen for H100-native BF16 throughput and stable quality.
- `--max-model-len TODO_FROM_H100_RUN`: set to cover the 1.5K-3K token prompt shape plus short SQL output while preserving KV-cache capacity.
- `--max-num-seqs TODO_FROM_H100_RUN`: chosen from load-test results to balance batching and P95 agent latency.
- `--max-num-batched-tokens TODO_FROM_H100_RUN`: tuned to keep prefill efficient without increasing queueing.
- `--gpu-memory-utilization TODO_FROM_H100_RUN`: left enough headroom for stable serving while maximizing KV-cache space.
- `--enable-prefix-caching`: schema and instruction prefixes repeat heavily across the eval/load-test workload.

Manual vLLM validation: TODO_FROM_H100_RUN. Capture the final H100 request/response screenshot at `screenshots/vllm_manual_query.png`.

## Baseline Eval Results

Baseline run: `results/eval_baseline.json`

- Total questions: 30
- Correct: TODO_FROM_H100_RUN
- Execution accuracy: TODO_FROM_H100_RUN
- Pass rate after initial generation: TODO_FROM_H100_RUN
- Pass rate after one revise: TODO_FROM_H100_RUN
- Pass rate after two revises: TODO_FROM_H100_RUN
- Pass rate after final allowed iteration: TODO_FROM_H100_RUN

The evaluation compares execution results rather than SQL text. For each item, the agent SQL and gold SQL are executed against the same SQLite DB, then rows are canonicalized and sorted before comparison. This avoids penalizing SQL that is syntactically different but result-equivalent.

Baseline observation: TODO_FROM_H100_RUN. The most common failures were TODO_FROM_H100_RUN.

## SLO and Tuning

Target SLO: P95 end-to-end agent latency under 5 seconds at 10+ RPS over a 5-minute window.

Baseline load result:

- RPS: TODO_FROM_H100_RUN
- Duration: 300 seconds
- P50 agent latency: TODO_FROM_H100_RUN
- P95 agent latency: TODO_FROM_H100_RUN
- Error rate: TODO_FROM_H100_RUN
- vLLM queue P95: TODO_FROM_H100_RUN
- vLLM prefill P95: TODO_FROM_H100_RUN
- vLLM decode P95: TODO_FROM_H100_RUN
- KV-cache usage peak: TODO_FROM_H100_RUN

Iteration log:

1. Saw TODO_FROM_H100_RUN -> hypothesized TODO_FROM_H100_RUN -> changed TODO_FROM_H100_RUN -> result was TODO_FROM_H100_RUN.
2. Saw TODO_FROM_H100_RUN -> hypothesized TODO_FROM_H100_RUN -> changed TODO_FROM_H100_RUN -> result was TODO_FROM_H100_RUN.
3. Saw TODO_FROM_H100_RUN -> hypothesized TODO_FROM_H100_RUN -> changed TODO_FROM_H100_RUN -> result was TODO_FROM_H100_RUN.

Final load result:

- RPS: TODO_FROM_H100_RUN
- Duration: 300 seconds
- P50 agent latency: TODO_FROM_H100_RUN
- P95 agent latency: TODO_FROM_H100_RUN
- Error rate: TODO_FROM_H100_RUN
- KV-cache usage peak: TODO_FROM_H100_RUN
- Verdict: TODO_FROM_H100_RUN

The before/after Grafana evidence is saved in `screenshots/grafana_before.png` and `screenshots/grafana_after.png`.

## Agent Value

The agent used a generate -> execute -> verify -> revise loop capped at 3 total SQL attempts. The loop helped on at least one manually inspected Phase 3 case, and the final eval pass-rate comparison will quantify the broader effect after the H100 run.

Phase 3 manual evidence shows the loop adding real value on the question "List down Ajax's superpowers." The initial SQL executed but returned only `power_id` values, which did not answer the request for superpower names. The verifier rejected that result, the revise step attempted a joined query, the verifier rejected a zero-row revision, and the final revision produced a correct join from `superhero` to `hero_power` to `superpower`, selecting `power_name`. The final answer returned `Agility`, `Super Strength`, `Super Speed`, `Heat Generation`, and `Power Suit` with `iterations=3` and `ok=true`. Raw evidence is saved in `results/phase3_ajax_revise_example.json`; terminal evidence is saved in `screenshots/phase3_agent_revise.png`.

Evidence: the pass rate moved from TODO_FROM_H100_RUN after initial generation to TODO_FROM_H100_RUN after the final allowed iteration. The revise step was most useful for TODO_FROM_H100_RUN and least useful for TODO_FROM_H100_RUN. Langfuse traces show the expected waterfall with `generate_sql`, `verify`, and when needed `revise`; screenshots are saved as `screenshots/langfuse_trace.png` and `screenshots/langfuse_tags.png`.

## Quality After Tuning

Post-tuning run: `results/eval_after_tuning.json`

- Correct: TODO_FROM_H100_RUN
- Execution accuracy: TODO_FROM_H100_RUN
- Change versus baseline: TODO_FROM_H100_RUN

Quality survived/did not survive the tuning pass: TODO_FROM_H100_RUN. The main quality impact was TODO_FROM_H100_RUN.

## What I Would Do With More Time

1. Add schema-linking before SQL generation so the model receives only the relevant tables and columns instead of the full schema context.
2. Add structured JSON output constraints for verifier decisions, with strict parsing and retry on invalid verifier output.
3. Split eval results by failure type: SQL syntax error, wrong table/column, wrong aggregation, wrong filter, empty result, and timeout.
4. Add prompt-prefix cache hit-rate monitoring to prove whether prefix caching is actually helping the repeated schema/instruction workload.
5. Try speculative decoding only if decode latency, not queueing or prefill, is the measured bottleneck.
