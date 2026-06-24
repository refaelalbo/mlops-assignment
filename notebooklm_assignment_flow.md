# MLOps Task 2 Learning Flow

This document is a clean study source for NotebookLM. It explains the assignment
from Phase 0 to Phase 6, what each phase was trying to prove, what was
implemented, what evidence was collected, and what the final H100 results mean.

It is for personal learning, not for submission. Secrets are intentionally not
included.

## Assignment Goal

The assignment was to build and observe an LLM-powered text-to-SQL service.

The full system had to:

- Serve an open-source LLM through vLLM.
- Build an agent that converts natural-language questions into SQL.
- Execute generated SQL against local BIRD-style SQLite databases.
- Add a verifier/revision loop so the agent can repair bad SQL.
- Trace agent behavior with Langfuse.
- Expose vLLM metrics to Prometheus and Grafana.
- Run evaluation and load tests before and after tuning.
- Produce a final report with measured quality, latency, throughput, and SLO
  interpretation.

The final measured target was a Nebius H100 VM running
`Qwen/Qwen3-30B-A3B-Instruct-2507`. Earlier local runs on
`Qwen/Qwen3-0.6B` were useful for development, but the final reported results
came from the H100 run.

## Phase 0 - Environment and Project Setup

Phase 0 established the project layout and runtime dependencies.

The main project folder was `mlops-assignment`. Important components:

- `agent/`: FastAPI service and SQL agent logic.
- `evals/`: evaluation runner and eval set.
- `load_test/`: load-test driver.
- `infra/`: Prometheus and Grafana provisioning.
- `scripts/`: data-loading and helper scripts.
- `results/`: JSON evidence from evals and load tests.
- `screenshots/`: Grafana, Langfuse, vLLM, and H100 evidence.
- `REPORT.md`: final assignment report.

The key setup steps were:

```bash
uv sync
docker compose up -d
uv run python scripts/load_data.py
```

Why `scripts/load_data.py` mattered:

- The agent executes SQL against local SQLite databases.
- Without loading data, requests fail with errors such as:
  `DB superhero not found at data/bird/superhero.sqlite`.

For the final H100 run, the Nebius VM also needed enough disk space because the
30B model cache was large. The boot disk was resized to about 200 GiB before the
final run.

## Phase 1 - vLLM Serving

Phase 1 served the model behind an OpenAI-compatible API using vLLM.

The final H100 baseline command was:

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

The tuned H100 command was:

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

Important serving choices:

- `bfloat16`: H100 supports BF16 efficiently.
- `max-model-len=4096`: enough context for schema-heavy prompts while limiting
  KV-cache pressure.
- `enable-prefix-caching`: useful because many requests share similar system,
  instruction, and schema prefixes.
- `gpu-memory-utilization=0.90`: uses most H100 memory but leaves runtime
  headroom.
- Tuning reduced `max_num_seqs` from 64 to 32 and `max_num_batched_tokens` from
  8192 to 4096 to reduce serving pressure.

One H100 startup issue occurred:

```text
fatal error: Python.h: No such file or directory
```

The fix was:

```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

Proof that the H100 was used:

- `screenshots/phase6_h100_nebius_vm.png`
- `screenshots/phase6_h100_nvidia_smi_vllm.png`

## Phase 2 - Prometheus and Grafana

Phase 2 added observability for the serving layer.

Prometheus scraped vLLM metrics from:

```text
http://host.docker.internal:8000/metrics
```

Grafana visualized:

- Latency panels.
- Throughput and queue metrics.
- Request success.
- Prompt and generation tokens.
- Requests running and waiting.
- KV-cache usage.
- Generated tokens per second.

The core purpose of this phase was to make performance visible during evals and
load tests. Without Grafana, it would be hard to explain whether a run was idle,
queued, overloaded, or actually serving traffic.

Key dashboard evidence:

- `screenshots/grafana_serving.png`
- `screenshots/phase6_h100_eval_baseline_grafana.png`
- `screenshots/phase6_h100_load_baseline_grafana.png`
- `screenshots/phase6_h100_load_after_tuning_grafana.png`
- `screenshots/phase6_h100_eval_after_tuning_grafana.png`

## Phase 3 - Agent, Execution, Verification, and Revision

Phase 3 built the actual text-to-SQL agent.

The FastAPI service exposed:

```text
POST /answer
```

The agent loop was:

1. Generate SQL from question and schema context.
2. Execute the SQL against the target SQLite database.
3. Verify whether the SQL result answers the user question.
4. If verification fails, revise the SQL.
5. Stop after a maximum of 3 total SQL attempts.

The important implementation idea was that successful SQL execution is not
enough. A query can execute and still answer the wrong question. The verifier
was added to catch that.

Manual Phase 3 example:

- Question: `List down Ajax's superpowers.`
- Initial SQL executed but returned only `power_id` values.
- The verifier rejected it because the question asked for superpower names.
- The revised SQL joined `superhero`, `hero_power`, and `superpower`.
- The final SQL selected `power_name`.

Evidence:

- `results/phase3_ajax_revise_example.json`
- `screenshots/phase3_agent_revise.png`

This phase is the clearest proof that the revise loop added value beyond simple
SQL generation.

## Phase 4 - Langfuse Tracing

Phase 4 added tracing for the agent workflow.

Langfuse was used to see:

- The overall request trace.
- `generate_sql` spans.
- SQL execution details.
- `verify` spans.
- `revise` spans when the first SQL attempt failed.
- Tags and metadata for evaluation runs.

The private keys belonged in the VM `.env` file only. They should not be
committed or uploaded as public documentation. The real values are copied from
the Langfuse project settings after creating the project/API keys in the
Langfuse UI.

Exact required `.env` shape, using realistic dummy values:

```env
LANGFUSE_SECRET_KEY="sk-lf-00000000-0000-0000-0000-000000000000"
LANGFUSE_PUBLIC_KEY="pk-lf-00000000-0000-0000-0000-000000000000"
LANGFUSE_HOST="http://localhost:3001"
LANGFUSE_BASE_URL="http://localhost:3001"
```

For the final Nebius VM run, the `.env` also needed the vLLM/OpenAI-compatible
settings:

```env
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
OPENAI_API_KEY="not-needed"
LANGFUSE_SECRET_KEY="sk-lf-00000000-0000-0000-0000-000000000000"
LANGFUSE_PUBLIC_KEY="pk-lf-00000000-0000-0000-0000-000000000000"
LANGFUSE_HOST="http://localhost:3001"
LANGFUSE_BASE_URL="http://localhost:3001"
```

NotebookLM should receive this shape, not the real secret values. The important
learning point is which variables are required and which hostnames are used
inside the VM.

Evidence:

- `screenshots/langfuse_trace.png`
- `screenshots/langfuse_tags.png`
- `screenshots/langfuse_tags_eval_baseline.png`

Learning point:

Langfuse was not only for screenshots. It helped validate the internal behavior
of the agent. If the final answer was wrong, the trace made it possible to see
whether the failure came from generation, execution, verification, or revision.

## Phase 5 - Evaluation

Phase 5 measured SQL-answer quality.

The evaluation runner loaded questions from:

```text
evals/eval_set.jsonl
```

It called:

```text
POST /answer
```

For each question, the eval compared execution results rather than raw SQL text.
This was important because two different SQL queries can be semantically
equivalent if they return the same rows.

Evaluation command:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_baseline_h100.json
```

Post-tuning evaluation command:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_after_tuning_h100.json
```

Final H100 baseline eval:

- Result file: `results/eval_baseline_h100.json`
- Total: 30
- Correct: 17
- Pass rate: 56.7%
- Agent OK rate: 76.7%
- Revision rate: 60.0%
- Average iterations: 1.83
- P50 latency: 0.600s
- P95 latency: 2.250s
- Iteration 1 pass rate: 10/30, 33.3%
- Iteration 2 pass rate: 17/30, 56.7%
- Iteration 3 pass rate: 17/30, 56.7%

Final H100 after-tuning eval:

- Result file: `results/eval_after_tuning_h100.json`
- Total: 30
- Correct: 16
- Pass rate: 53.3%
- Agent OK rate: 76.7%
- Revision rate: 60.0%
- Average iterations: 1.83
- P50 latency: 0.587s
- P95 latency: 2.239s
- Iteration 1 pass rate: 9/30, 30.0%
- Iteration 2 pass rate: 16/30, 53.3%
- Iteration 3 pass rate: 16/30, 53.3%

Learning point:

The revision loop clearly helped. On the H100 baseline, quality improved from
33.3% after the first attempt to 56.7% after revision. On the post-tuning run,
quality improved from 30.0% to 53.3%. However, tuning did not improve final
quality.

## Phase 6 - Load Test and SLO Tuning

Phase 6 tested service behavior under concurrent traffic and compared baseline
versus tuned serving parameters.

Baseline load command:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 2 \
  --duration 120 \
  --out results/load_baseline_h100.json
```

After-tuning load command:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 2 \
  --duration 120 \
  --out results/load_after_tuning_h100.json
```

The target SLO was:

```text
P95 end-to-end agent latency under 5 seconds at 10+ RPS over a 5-minute window.
```

The final measured H100 load tests used 2 RPS for 120 seconds. Therefore, the
10+ RPS SLO was not met and should not be claimed as met.

H100 baseline load result:

- Result file: `results/load_baseline_h100.json`
- Requested RPS: 2.0
- Achieved RPS: 1.33
- Wall-clock seconds: 180.0 including request drain
- Total requests: 240
- Successful requests: 206
- Timeouts: 0
- HTTP errors: 30
- Client errors: 4
- P50 latency: 0.870s
- P95 latency: 4.894s
- P99 latency: 7.879s
- Max latency: 8.908s

H100 after-tuning load result:

- Result file: `results/load_after_tuning_h100.json`
- Requested RPS: 2.0
- Achieved RPS: 1.33
- Wall-clock seconds: 180.0 including request drain
- Total requests: 240
- Successful requests: 207
- Timeouts: 0
- HTTP errors: 30
- Client errors: 3
- P50 latency: 0.904s
- P95 latency: 5.851s
- P99 latency: 9.083s
- Max latency: 10.308s

Tuning interpretation:

- Tuning reduced serving pressure.
- Client errors improved slightly from 4 to 3.
- Timeouts remained 0.
- P95 latency regressed from 4.894s to 5.851s.
- Eval accuracy regressed from 17/30 to 16/30.

Final verdict:

The H100 tuning was stability-oriented, not a latency win. It slightly reduced
client errors but did not improve P95 latency or final quality.

## Final H100 Run Workflow

The clean operational model used four terminals.

### Terminal A - Local WSL Tunnel

This ran on the local laptop in WSL and stayed open:

```bash
ssh -i ~/.ssh/id_ed25519 \
  -L 13000:localhost:3000 \
  -L 13001:localhost:3001 \
  -L 19090:localhost:9090 \
  -L 18001:localhost:8001 \
  refae@<vm-public-ip>
```

Local browser URLs:

- Grafana: `http://localhost:13000`
- Langfuse: `http://localhost:13001`
- Prometheus: `http://localhost:19090`
- Agent docs: `http://localhost:18001/docs`

### Terminal B - vLLM on the Nebius VM

This ran vLLM directly on the H100 VM.

Baseline used:

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

After tuning, vLLM was stopped with `Ctrl+C` and restarted with:

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

### Terminal C - Agent API on the Nebius VM

This ran the FastAPI agent:

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

### Terminal D - Eval and Load Commands on the Nebius VM

This ran data loading, evals, and load tests:

```bash
uv run python scripts/load_data.py
uv run python evals/run_eval.py --agent-url http://localhost:8001/answer --out results/eval_baseline_h100.json
uv run python load_test/driver.py --agent-url http://localhost:8001/answer --rps 2 --duration 120 --out results/load_baseline_h100.json
uv run python load_test/driver.py --agent-url http://localhost:8001/answer --rps 2 --duration 120 --out results/load_after_tuning_h100.json
uv run python evals/run_eval.py --agent-url http://localhost:8001/answer --out results/eval_after_tuning_h100.json
```

The run order matters:

1. Start Docker services.
2. Load data.
3. Start baseline vLLM.
4. Start the agent.
5. Run baseline eval.
6. Run baseline load.
7. Stop vLLM.
8. Restart vLLM with tuned parameters.
9. Run after-tuning load.
10. Run after-tuning eval.
11. Save JSONs and screenshots.
12. Copy H100 result files back to the local project.

## Code Reading Flow

The Python files now include learning comments using this format:

```python
# Goal: what this block accomplishes.
# Why: the engineering reason this block exists.
```

Read the code in this order:

1. `scripts/load_data.py`

This explains how the BIRD data is downloaded, split into eval/load JSONL files,
and surfaced as `data/bird/<db_id>.sqlite`.

2. `agent/schema.py`

This explains how `db_id="superhero"` becomes `data/bird/superhero.sqlite`, and
how the project renders tables, columns, primary keys, and foreign keys into
schema text for the LLM.

3. `agent/prompts.py`

This shows the three prompt roles: generate SQL, verify the executed result, and
revise SQL after a verifier complaint.

4. `agent/execution.py`

This shows how generated SQL is executed in read-only SQLite mode and converted
into structured rows, columns, row counts, and errors.

5. `agent/graph.py`

This is the main agent workflow:

```text
attach_schema
-> generate_sql
-> execute
-> verify
-> revise if needed
-> execute again
-> verify again
```

6. `agent/server.py`

This wraps the graph in FastAPI so eval/load/manual calls can use
`POST /answer`.

7. `evals/run_eval.py`

This explains execution-accuracy scoring: run gold SQL, run predicted SQL, then
compare canonicalized rows rather than comparing SQL text.

8. `load_test/driver.py`

This explains open-loop traffic generation, latency measurement, success/error
classification, and P50/P95/P99 summaries.

9. `tests/test_graph_repairs.py`

This shows regression tests for deterministic repair rules that were added after
observing repeated text-to-SQL failure patterns.

The comments are educational only. They do not change executable logic and do
not require rerunning the H100 experiment.

## Final Results Summary

Quality:

| Run | Correct | Pass Rate | P50 Eval Latency | P95 Eval Latency |
|---|---:|---:|---:|---:|
| H100 baseline eval | 17/30 | 56.7% | 0.600s | 2.250s |
| H100 after tuning eval | 16/30 | 53.3% | 0.587s | 2.239s |

Load:

| Run | OK / Total | Timeouts | Client Errors | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| H100 baseline load | 206/240 | 0 | 4 | 0.870s | 4.894s | 7.879s |
| H100 after tuning load | 207/240 | 0 | 3 | 0.904s | 5.851s | 9.083s |

Main conclusion:

The revision loop improved SQL quality, but the serving tuning did not improve
the final H100 outcome. The tuned run had slightly fewer client errors but worse
P95 load latency and slightly worse eval accuracy.

## Final Submission Artifacts

Final H100 result JSONs:

- `results/eval_baseline_h100.json`
- `results/load_baseline_h100.json`
- `results/load_after_tuning_h100.json`
- `results/eval_after_tuning_h100.json`

Final H100 screenshots:

- `screenshots/phase6_h100_nebius_vm.png`
- `screenshots/phase6_h100_nvidia_smi_vllm.png`
- `screenshots/phase6_h100_eval_baseline_grafana.png`
- `screenshots/phase6_h100_load_baseline_grafana.png`
- `screenshots/phase6_h100_load_after_tuning_grafana.png`
- `screenshots/phase6_h100_eval_after_tuning_grafana.png`

Other useful evidence:

- `screenshots/phase3_agent_revise.png`
- `screenshots/langfuse_trace.png`
- `screenshots/langfuse_tags.png`
- `screenshots/langfuse_tags_eval_baseline.png`

Final report:

- `REPORT.md`

Final clean submission zip:

- `mlops-assignment_refael_albo_H100.zip`

## Nebius Cleanup

After copying all result JSONs and screenshots back locally, the Nebius VM was
deleted.

Final cleanup proof from the console:

- Standalone VMs: 0
- Disks: No disks

This matters because stopping or deleting the VM prevents continuing H100
compute charges, and deleting the boot disk prevents storage charges.

## What To Ask NotebookLM

Good questions for learning:

- Explain the assignment phase by phase.
- Why was the verifier/revision loop useful?
- What is the difference between SQL execution success and answer correctness?
- Why did the H100 tuning not count as a performance win?
- What evidence proves the final run used H100?
- Which files are final evidence and which are development-only evidence?
- What would be the next engineering improvements?
- How would schema linking improve text-to-SQL accuracy?
- Why does the final report not claim the 10+ RPS SLO was met?

## Lessons Learned

1. Observability is only useful when tied to a concrete run sequence.
2. Load results are only meaningful if the baseline and after-tuning runs use
   the same driver settings.
3. An after-tuning run must happen after a real serving change.
4. A screenshot is useful only if it corresponds to the exact run being reported.
5. SQL agents need execution checks, but execution checks alone are not enough.
6. The revision loop improved answer quality, but it also adds latency.
7. H100 made the large model feasible, but did not automatically satisfy the
   throughput SLO.
8. Final reporting should be honest: include regressions and failed SLOs instead
   of overstating success.
