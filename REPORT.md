# LLM Inference + Observability Report

## Serving Configuration

Final run target:

- Cloud: Nebius H100 VM
- GPU: NVIDIA H100 80GB HBM3
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Evidence:
  - `screenshots/phase6_h100_nebius_vm.png`
  - `screenshots/phase6_h100_nvidia_smi_vllm.png`

Baseline H100 vLLM command:

```bash
uv run vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Tuned H100 vLLM command:

```bash
uv run vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Configuration rationale:

- `--dtype bfloat16`: H100 supports BF16 efficiently and it matches the intended target deployment dtype.
- `--max-model-len 4096`: covers the schema-heavy BIRD prompts and short SQL outputs while preserving KV-cache headroom.
- `--max-num-seqs` and `--max-num-batched-tokens`: baseline used `64` / `8192`; after tuning used `32` / `4096` to reduce serving pressure and queueing risk.
- `--gpu-memory-utilization 0.90`: uses most H100 memory while leaving a small reserve for runtime overhead.
- `--enable-prefix-caching`: the workload repeatedly sends similar system, schema, and instruction prefixes.

The local development model was `Qwen/Qwen3-0.6B` on an RTX 3060 laptop GPU. Those local results are kept as development evidence, but the final reported metrics below come from the Nebius H100 run.

## Baseline Eval Results

Baseline H100 run: `results/eval_baseline_h100.json`

- Total questions: 30
- Correct: 17
- Execution accuracy: 56.7%
- Agent OK rate: 76.7%
- Revision rate: 60.0%
- Average iterations: 1.83
- P50 latency: 0.600 seconds
- P95 latency: 2.250 seconds
- Per-iteration pass rate:
  - Iteration 1: 10/30, 33.3%
  - Iteration 2: 17/30, 56.7%
  - Iteration 3: 17/30, 56.7%
- Evidence: `screenshots/phase6_h100_eval_baseline_grafana.png`

The evaluation compares execution results rather than SQL text. For each item, the agent SQL and gold SQL are executed against the same SQLite DB, then rows are canonicalized and sorted before comparison. This avoids penalizing SQL that is syntactically different but result-equivalent.

## SLO and Tuning

Target SLO: P95 end-to-end agent latency under 5 seconds at 10+ RPS over a 5-minute window.

The final measured H100 load tests used the assignment driver at 2 RPS for 120 seconds. The stack did not reach the 10+ RPS SLO target in this run, so the SLO is not claimed as met.

Baseline H100 load:

- Run: `results/load_baseline_h100.json`
- Requested RPS: 2.0
- Achieved RPS: 1.33 over 180.0 seconds wall-clock including request drain
- Total requests: 240
- Successful requests: 206
- Timeouts: 0
- HTTP errors: 30
- Client errors: 4
- P50 agent latency: 0.870 seconds
- P95 agent latency: 4.894 seconds
- P99 agent latency: 7.879 seconds
- Max latency: 8.908 seconds
- Grafana evidence: `screenshots/phase6_h100_load_baseline_grafana.png`

Tuned H100 load:

- Run: `results/load_after_tuning_h100.json`
- Requested RPS: 2.0
- Achieved RPS: 1.33 over 180.0 seconds wall-clock including request drain
- Total requests: 240
- Successful requests: 207
- Timeouts: 0
- HTTP errors: 30
- Client errors: 3
- P50 agent latency: 0.904 seconds
- P95 agent latency: 5.851 seconds
- P99 agent latency: 9.083 seconds
- Max latency: 10.308 seconds
- Grafana evidence: `screenshots/phase6_h100_load_after_tuning_grafana.png`

Iteration log:

1. The baseline H100 run used `max_num_seqs=64` and `max_num_batched_tokens=8192`.
2. The tuning pass restarted vLLM with `max_num_seqs=32` and `max_num_batched_tokens=4096` to reduce concurrent serving pressure.
3. The tuned load kept zero timeouts and slightly reduced client errors from 4 to 3, but P95 latency regressed from 4.894s to 5.851s.

Verdict: the H100 tuning was stability-oriented, not a latency win. It slightly reduced client errors but did not improve P95 latency, and the final measured run does not meet the stated 10+ RPS SLO.

## Agent Value

The agent used a generate -> execute -> verify -> revise loop capped at 3 total SQL attempts.

On the H100 baseline eval, pass rate improved from 33.3% after the first attempt to 56.7% after the revision loop. On the H100 post-tuning eval, pass rate improved from 30.0% after the first attempt to 53.3% after revision. This shows the verifier/revise loop adds measurable value beyond a single generation pass.

Manual Phase 3 evidence also shows the loop adding real value on the question "List down Ajax's superpowers." The initial SQL executed but returned only `power_id` values, which did not answer the request for superpower names. The verifier rejected that result, the revise step attempted a joined query, and the final revision produced a correct join from `superhero` to `hero_power` to `superpower`, selecting `power_name`. Raw evidence is saved in `results/phase3_ajax_revise_example.json`; terminal evidence is saved in `screenshots/phase3_agent_revise.png`.

Langfuse traces show the expected waterfall with `generate_sql`, `verify`, and when needed `revise`; screenshots are saved as `screenshots/langfuse_trace.png`, `screenshots/langfuse_tags.png`, and `screenshots/langfuse_tags_eval_baseline.png`.

## Quality After Tuning

Post-tuning H100 eval: `results/eval_after_tuning_h100.json`

- Correct: 16 out of 30
- Execution accuracy: 53.3%
- Change versus H100 baseline: 17/30 -> 16/30, a 1-question regression
- Agent OK rate: 76.7%, unchanged from baseline
- Average iterations: 1.83, unchanged from baseline
- Revision rate: 60.0%, unchanged from baseline
- P50 latency: 0.587 seconds, slightly better than baseline 0.600 seconds
- P95 latency: 2.239 seconds, slightly better than baseline 2.250 seconds
- Evidence: `screenshots/phase6_h100_eval_after_tuning_grafana.png`

Quality did not improve after the reduced-pressure tuning: execution accuracy moved from 56.7% to 53.3%. The latency change during eval was negligible and slightly favorable, but the load-test P95 regressed. The final interpretation is that this tuning was not a net improvement for the H100 configuration; it is included because the assignment asks for a measured before/after loop and because it exposed the stability/latency tradeoff clearly.

## What I Would Do With More Time

1. Add schema-linking before SQL generation so the model receives only relevant tables and columns instead of the full schema context.
2. Add structured JSON output constraints for verifier decisions, with strict parsing and retry on invalid verifier output.
3. Split eval results by failure type: SQL syntax error, wrong table/column, wrong aggregation, wrong filter, empty result, and timeout.
4. Tune the H100 serving parameters around measured queueing and GPU utilization rather than only reducing concurrency.
5. Run a true 10+ RPS H100 test after hardening HTTP 500 failure paths in the agent.
