# MLOps Task 2: Elaborated Runbook and Answer Notes

This file is an elaborated version of `MLOps_Task2_Answer.md`. It keeps the
same assignment flow, but makes the local-vs-Nebius setup explicit and adds the
Phase 3 agent implementation/run steps from our troubleshooting.

Final screenshots and metrics must come from the Nebius H100 run with
`Qwen/Qwen3-30B-A3B-Instruct-2507`. Local WSL results with `Qwen/Qwen3-0.6B`
are useful for development only.

## Required Final Deliverables

```text
REPORT.md
infra/grafana/provisioning/dashboards/serving.json
agent/graph.py
agent/prompts.py
evals/run_eval.py
results/eval_baseline.json
results/eval_after_tuning.json
screenshots/vllm_manual_query.png
screenshots/grafana_serving.png
screenshots/langfuse_trace.png
screenshots/langfuse_tags.png
screenshots/grafana_eval_run.png
screenshots/grafana_before.png
screenshots/grafana_after.png
results/phase3_ajax_revise_example.json
screenshots/phase3_agent_revise.png
```

## Phase 0: Setup

There are two separate environments:

- Local WSL/Docker Desktop: development with `Qwen/Qwen3-0.6B`.
- Nebius H100 VM: final screenshots, evals, load test, and report numbers with
  `Qwen/Qwen3-30B-A3B-Instruct-2507`.

Local Docker containers, Grafana UI edits, model cache, `.env`, and unpushed git
commits do not automatically appear on Nebius.

Because the upstream repository is not yours, pushing directly to
`GlebBerjoskin/mlops-assignment` may fail with HTTP 403. Use one of these:

- Fork the repo, push your changes to your fork, then clone your fork on Nebius.
- Manually copy changed files to Nebius.

Important changed files for the current work:

```text
pyproject.toml
uv.lock
REPORT.md
agent/graph.py
agent/prompts.py
agent/server.py
infra/grafana/provisioning/dashboards/serving.json
```

Dependency compatibility note:

- vLLM 0.10.2 works with `transformers>=4.55.2,<5.0`.
- The earlier crash, `Qwen2Tokenizer has no attribute all_special_tokens_extended`,
  came from using a Transformers 5.x-compatible resolution with vLLM/Qwen.
- After `uv sync`, confirm:

```bash
uv run python -c "import transformers; print(transformers.__version__)"
```

Expected: `4.x`, for example `4.57.6`.

Local setup:

```bash
cd /mnt/c/dev/study___nebius_acdemy/part_3_MLOps/MLOps_task2/mlops-assignment
uv sync
cp .env.example .env
uv run python scripts/load_data.py
docker compose up -d
```

Local `.env` while serving the small model:

```env
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen3-0.6B
OPENAI_API_KEY=not-needed
```

Nebius setup:

```bash
git clone https://github.com/<your-github-user>/mlops-assignment
cd mlops-assignment
uv sync
cp .env.example .env
uv run python scripts/load_data.py
docker compose up -d
```

Nebius `.env` for the final run:

```env
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
OPENAI_API_KEY=not-needed
```

Port forwarding from Nebius to your laptop:

```bash
ssh -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    <user>@<vm-host>
```

Expected browser URLs:

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Langfuse: `http://localhost:3001`
- vLLM OpenAI-compatible API docs: `http://localhost:8000/docs`
- Agent API docs: `http://localhost:8001/docs`

## Phase 1: vLLM Serving

Local WSL development command for a 6 GB RTX 3060:

```bash
uv run vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.75 \
  --enable-prefix-caching
```

If local vLLM fails with:

```text
Free memory on device (4.98/6.0 GiB) on startup is less than desired GPU memory utilization (0.9, 5.4 GiB).
```

then `--gpu-memory-utilization 0.90` is too high for the available local VRAM.
Use the local command above, or go more conservative:

```bash
uv run vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.65 \
  --enable-prefix-caching
```

The line below means vLLM is alive and Prometheus is scraping `/metrics`:

```text
INFO: 127.0.0.1:49868 - "GET /metrics HTTP/1.1" 200 OK
```

`nvidia-smi` in WSL can show memory in use while listing no WSL process because
Windows-side display or compute processes may own GPU memory. Check Windows
Task Manager or PowerShell `nvidia-smi` if local memory looks inconsistent.

H100 serving command for final metrics:

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

Do not include `--disable-log-requests`; in vLLM 0.10.2 it is deprecated.

Serving flag rationale:

- `--dtype bfloat16`: H100 supports BF16 efficiently.
- `--max-model-len 4096`: enough for schema-heavy text-to-SQL prompts without
  over-reserving KV cache.
- `--max-num-seqs 64`: allows concurrent agent calls while protecting tail
  latency.
- `--max-num-batched-tokens 8192`: balances prompt-heavy prefill batching with
  P95 latency.
- `--gpu-memory-utilization 0.90`: uses most H100 memory while leaving a small
  stability margin.
- `--enable-prefix-caching`: useful because schema and instruction prefixes
  repeat across eval/load-test requests.

Manual validation:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Root path behavior:

- `http://localhost:8000` returning `{"detail":"Not Found"}` is normal.
- `http://localhost:8000/docs` shows FastAPI/OpenAI-compatible endpoints.
- Useful endpoints are `/health`, `/metrics`, `/v1/models`, and
  `/v1/chat/completions`.

Small local request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

Final H100 screenshot request should use:

```text
Qwen/Qwen3-30B-A3B-Instruct-2507
```

Save the final vLLM screenshot as:

```text
screenshots/vllm_manual_query.png
```

## Phase 2: Grafana Dashboard

Docker Compose starts Prometheus, Grafana, Langfuse, and storage services. It
does not start vLLM or the agent server. Prometheus is configured to scrape
vLLM on `host.docker.internal:8000`.

Prometheus can look static if you only open `http://localhost:9090`. Use:

- `Status -> Target health` to confirm the vLLM target is `UP`.
- The expression browser to query metrics like `up`,
  `vllm:num_requests_running`, or `vllm:request_success_total`.

The dashboard should answer:

- Is end-to-end request latency inside target?
- Is latency coming from queueing, prefill, decode, or output-token generation?
- Is KV cache close to saturation?
- What request/token throughput is the server handling?

Dashboard persistence:

- Changing `VLLM_MODEL` from the local 0.6B model to the H100 30B model does
  not require rebuilding panels.
- Local Grafana UI edits do not automatically appear on Nebius.
- Save/export the final dashboard JSON into
  `infra/grafana/provisioning/dashboards/serving.json`, then commit/copy that
  file to Nebius before the final run.

How to add the latency panel:

1. Open `http://localhost:3000`.
2. Open the provisioned `vLLM serving` dashboard.
3. Select `Add -> Visualization`.
4. Choose the Prometheus datasource.
5. Use a `Time series` visualization.
6. Paste one PromQL query into query `A`.
7. Click `+ Query` for each additional latency query if using one combined
   panel.
8. Set unit to seconds.
9. Set `Legend` or `Legend format` for each query to a readable alias.

Readable legend examples:

```text
p95 e2e
p95 queue
p95 prefill
p95 decode
p95 ttft
p95 per output token
```

Changing the query ref ID from `A`, `B`, `C` to a better name is not enough.
The graph legend uses the Prometheus series name unless `Legend format` is set.
If the legend field is hidden, switch the Prometheus query editor from Builder
mode to Code mode or expand the query options.

Latency queries:

```promql
histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_queue_time_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_prefill_time_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_decode_time_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le))
```

The repo dashboard currently uses `vllm:time_per_output_token_seconds_bucket`
for the last query. If using another vLLM version, verify the exact metric name
in Prometheus first.

Throughput and queue queries:

```promql
sum(rate(vllm:request_success_total[5m]))
sum(rate(vllm:prompt_tokens_total[5m]))
sum(rate(vllm:generation_tokens_total[5m]))
sum(vllm:num_requests_running)
sum(vllm:num_requests_waiting)
sum by (reason) (vllm:num_requests_waiting_by_reason)
```

KV-cache queries:

```promql
vllm:kv_cache_usage_perc
rate(vllm:num_preemptions_total[5m])
```

Save:

```text
infra/grafana/provisioning/dashboards/serving.json
screenshots/grafana_serving.png
```

## Phase 3: Agent

Files changed for Phase 3:

```text
agent/graph.py
agent/prompts.py
```

The intended graph:

```text
question + schema
  -> generate_sql
  -> execute
  -> verify
  -> if ok: end
  -> if not ok and under iteration cap: revise -> execute -> verify
```

Implementation intent:

- `generate_sql_node`: create first SQL from question and schema.
- `execute_node`: run SQL against the SQLite DB.
- `verify_node`: inspect the question, SQL, execution result, and error state.
- `revise_node`: repair SQL using the verifier issue and previous failure.
- `route_after_verify`: end if `verify_ok=true` or `iteration>=MAX_ITERATIONS`;
  otherwise route to `revise`.

Implementation checklist:

1. In `agent/prompts.py`, fill:

```python
GENERATE_SQL_SYSTEM
GENERATE_SQL_USER
VERIFY_SYSTEM
VERIFY_USER
REVISE_SYSTEM
REVISE_USER
```

2. In `agent/graph.py`, implement:

```python
verify_node()
revise_node()
route_after_verify()
```

3. Add robust parsing:

- `_extract_sql(...)` should strip Qwen `<think>...</think>` blocks before
  extracting SQL, including unterminated `<think>` output.
- The verifier should ask for JSON and defensively extract the first JSON object
  from the LLM response.
- Add deterministic verifier guards for obvious semantic failures that a small
  local model may accept too easily, such as returning only ID columns when the
  user asks for names or list entries.
- Ensure the HTTP response reports `ok=false` when the final SQL executes but
  the verifier rejected the final result.

Verifier behavior:

- Return `ok=false` when SQL errored.
- Return `ok=false` when rows are empty but the question asks to list/show/find
  existing records.
- Return `ok=false` when selected columns do not answer the question.
- Return `ok=false` when aggregation, grouping, ordering, or limit is clearly
  missing.
- Return `ok=true` when rows plausibly answer the question.

After editing, run:

```bash
python -m py_compile agent/graph.py agent/prompts.py
```

Start vLLM first, then start the agent in another terminal:

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

Validate the agent server:

```bash
curl http://localhost:8001/health
```

Expected:

```json
{"status":"ok"}
```

If `http://localhost:8001` gives `ERR_CONNECTION_REFUSED`, the agent server is
not running. This is expected until `uvicorn agent.server:app` is started.

Test one eval-set question:

```bash
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"List down Ajax'\''s superpowers.","db":"superhero"}'
```

Expected response shape:

```json
{
  "sql": "...",
  "rows": [["..."]],
  "iterations": 3,
  "ok": true,
  "history": [
    {"node": "generate_sql", "sql": "..."},
    {"node": "verify", "ok": false, "issue": "..."},
    {"node": "revise", "issue": "...", "sql": "..."},
    {"node": "verify", "ok": true, "issue": ""}
  ]
}
```

Test more questions from the generated eval set:

```bash
head -5 evals/eval_set.jsonl
```

Save one example that triggers revision:

```bash
mkdir -p results
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"List down Ajax'\''s superpowers.","db":"superhero"}' \
  | tee results/phase3_ajax_revise_example.json
```

The current saved Phase 3 proof is:

```text
Question: List down Ajax's superpowers.
Initial SQL issue: returned only power_id values instead of superpower names.
Second issue: revised SQL used the wrong entity string and returned zero rows.
Final SQL: joins superhero -> hero_power -> superpower and selects power_name.
Final result: Agility, Super Strength, Super Speed, Heat Generation, Power Suit.
Iterations: 3.
Raw proof: results/phase3_ajax_revise_example.json.
Screenshot proof: screenshots/phase3_agent_revise.png.
```

For the assignment, `iterations > 1` is the proof that the `verify -> revise`
loop did real work. The returned `history` field from `POST /answer` is the
best place to collect this evidence.







## Phase 4: Langfuse Tracing

Goal: prove that agent runs are traced, inspectable, and tagged. The required
evidence is a Langfuse trace waterfall showing `generate_sql`, `verify`, and
when applicable `revise`.

Langfuse runs at `http://localhost:3001`, but it will look empty/static until:

1. You sign up in the local Langfuse UI.
2. You create a project.
3. You add the project keys to `.env`.
4. You restart the agent server.
5. You send requests through `POST /answer`.

What to do:

1. Open Langfuse:

```text
http://localhost:3001
```

2. Create a local account and project.

3. Copy the project API keys into `.env`:

```env
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=your_public_key_here
LANGFUSE_SECRET_KEY=your_secret_key_here
```

4. Restart the agent server so `agent/server.py` reloads `.env`:

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

The current `agent/server.py` already initializes the Langfuse callback handler
when the keys are present and passes tags as metadata:

```python
config = {
    "callbacks": [_lf_handler] if _lf_handler is not None else [],
    "metadata": req.tags,
}
final = graph.invoke(state, config=config)
```

5. Send several tagged agent requests:

```bash
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "List down Ajax'\''s superpowers.",
    "db": "superhero",
    "tags": {
      "phase": "phase4_trace_check",
      "run_type": "manual",
      "model": "Qwen/Qwen3-0.6B"
    }
  }'
```

For the final H100 run, use:

```json
{
  "tags": {
    "run_type": "baseline",
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507"
  }
}
```

6. Open Langfuse and inspect:

- The trace list should show new traces after requests.
- Trace metadata/tags should include `run_type`, `model`, and any phase label.
- A trace should contain the LangGraph/LLM call waterfall.
- At least one trace should include a revise path if you have a question that
  triggers `iterations > 1`.

Troubleshooting:

- Empty Langfuse UI usually means the agent was not restarted after adding
  keys, or no `/answer` request was sent after keys were configured.
- If Langfuse keys are wrong, the agent may fail during callback initialization
  or trace upload.
- If `POST /answer` works but no traces appear, confirm `.env` is in the repo
  root and that `LANGFUSE_HOST=http://localhost:3001`.

Expected Langfuse evidence:

- Trace list with metadata/tags visible.
- Trace waterfall showing `generate_sql`, `verify`, and sometimes `revise`.
- Prompt, response, latency, and token data for the LLM calls.

Save:

```text
screenshots/langfuse_trace.png
screenshots/langfuse_tags.png
```







------------------------------------------------------------------
------------------------------------------------------------------
• Do Phase 4 in this order.

  1. Open Langfuse

  Browser:

  http://localhost:3001

  If it asks to create an account, create a local account.

  Then:

  - Create a project
  - Go to project settings / API keys
  - Copy:
      - public key
      - secret key

  2. Put keys in .env

  In WSL, from repo root:

  cd /mnt/c/dev/study___nebius_acdemy/part_3_MLOps/MLOps_task2/mlops-assignment
  nano .env

  Set:

  LANGFUSE_HOST=http://localhost:3001
  LANGFUSE_PUBLIC_KEY=your_public_key_here
  LANGFUSE_SECRET_KEY=your_secret_key_here

  Keep your vLLM settings as they are.

  3. Restart agent server

  Stop current agent server with CTRL+C.

  Then:

  uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

  If Langfuse config is bad, the agent may show an error. If it starts normally, continue.

  4. Send a tagged request

  In another WSL terminal:

  cd /mnt/c/dev/study___nebius_acdemy/part_3_MLOps/MLOps_task2/mlops-assignment

  curl -X POST http://localhost:8001/answer \
    -H "Content-Type: application/json" \
    -d '{
      "question":"List down Ajax'\''s superpowers.",
      "db":"superhero",
      "tags":{
        "phase":"phase4_trace_check",
        "run_type":"manual",
        "model":"Qwen/Qwen3-0.6B"
      }
    }'

  You want response with:

  "ok": true

  and preferably:

  "iterations": 3

  5. Check Langfuse

  Go back to:

  http://localhost:3001

  Look for a new trace.

  Open it. You want to see:

  - trace metadata/tags
  - LLM calls / LangGraph steps
  - generate_sql
  - verify
  - revise if visible
  - timings/token usage if available

  6. Save screenshots

  Save two screenshots:

  mlops-assignment/screenshots/langfuse_trace.png
  mlops-assignment/screenshots/langfuse_tags.png

  What each should show:

  langfuse_trace.png

  Opened trace/waterfall with the agent steps or LLM calls.

  langfuse_tags.png

  Trace list or trace details showing metadata/tags like phase, run_type, model.

  After saving them, verify that the files exist in `mlops-assignment/screenshots/`.
------------------------------------------------------------------
------------------------------------------------------------------











## Phase 5: Evaluation

Goal: measure execution accuracy, not textual SQL similarity. A prediction is
correct if the final agent SQL and the gold SQL return the same canonicalized
rows on the same SQLite DB.

The current `evals/run_eval.py` scaffold still requires Phase 5 work. Before
running the eval, implement:

```python
eval_one()
summarize()
```

What `eval_one()` should do:

1. Read `question`, `db_id`, and `gold_sql` from one eval record.
2. Call the agent HTTP endpoint:

```json
{
  "question": "...",
  "db": "...",
  "tags": {
    "phase": "phase5_eval",
    "run_type": "baseline"
  }
}
```

3. Extract from the response:

- final `sql`
- final `rows`
- `iterations`
- `history`
- `ok`
- `error`

4. Execute `gold_sql` locally with the provided `run_sql(db_id, gold_sql)`.

5. Execute the agent final SQL locally with the provided
   `run_sql(db_id, agent_sql)`.

6. Compare with the provided `matches(gold_rows, pred_rows)`.

7. Return a per-question dict containing at least:

```json
{
  "question": "...",
  "db_id": "...",
  "gold_sql": "...",
  "pred_sql": "...",
  "correct": true,
  "iterations": 1,
  "agent_ok": true,
  "agent_error": null,
  "gold_error": null,
  "pred_error": null,
  "history": []
}
```

What `summarize()` should report:

- total questions
- number correct
- pass rate
- average iterations
- count of questions that used revise, meaning `iterations > 1`
- per-iteration pass rates if possible

Per-iteration pass rate is important for the report because it answers whether
the agent loop added value. If iteration 1 and final iteration have the same
pass rate, the revise loop did not help. If final pass rate is higher, cite that.

Run eval only after vLLM and the agent server are both running:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_baseline.json
```

Watch Grafana during the run. This eval calls the agent for each question, and
each agent run may call vLLM more than once because of verify/revise.

Quick checks after eval:

```bash
cat results/eval_baseline.json
```

Look for:

- `summary.pass_rate`
- number of failed questions
- examples with `iterations > 1`
- repeated failure reasons in `history`

Use the failures to improve prompts or verifier rules, then rerun:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_after_tuning.json
```

Save:

```text
results/eval_baseline.json
results/eval_after_tuning.json
screenshots/grafana_eval_run.png
```




------------
------------
• Phase 5 is now included.

  What I did:

  - Implemented evals/run_eval.py.
  - Confirmed results/eval_baseline.json exists.
  - Read the eval summary:
      - Total: 30
      - Correct: 2
      - Local baseline execution accuracy: 6.7%
      - Revision rate: 53.3%
      - Avg iterations: 1.9
      - P50 latency: 1.30s
      - P95 latency: 26.41s
  - Updated REPORT.md to include these local Phase 5 results clearly as local Qwen/Qwen3-0.6B evidence, not final H100 results.
  - Rebuilt the zip.
------------
------------









## Phase 6: Load Test and SLO Tuning

Goal: tune the serving and agent system until it satisfies the target SLO, then
explain the tuning using Grafana evidence rather than guessing.

Target SLO:

```text
P95 end-to-end agent latency < 5 seconds at 10+ RPS for 5 minutes.
```

Preflight checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8000/v1/models
```

Start with a short smoke load test before the full 5-minute run:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 2 \
  --duration 30 \
  --out results/load_smoke_2rps.json
```

Then try intermediate load:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 5 \
  --duration 120 \
  --out results/load_5rps.json
```

Final SLO run:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 10 \
  --duration 300 \
  --out results/load_10rps.json
```

The load driver output summary includes:

- `achieved_rps`
- `ok`
- `timeouts`
- `http_errors`
- `client_errors`
- `latency_p50`
- `latency_p95`
- `latency_p99`
- `latency_max`

For the SLO, use `latency_p95` from `results/load_10rps.json` as the
end-to-end agent P95. Use Grafana to explain why it was fast or slow.

Use Grafana to decide what to change:

- High queue time: reduce concurrency, reduce agent iterations, or tune
  `--max-num-seqs`.
- High prefill time: reduce prompt/schema length or tune
  `--max-num-batched-tokens`.
- High decode time or per-output-token time: limit answer length and keep SQL
  output short.
- High KV-cache usage or preemptions: reduce `--max-model-len`,
  `--max-num-seqs`, or increase available GPU memory.

Concrete tuning levers:

- If P95 is high and vLLM queue time is high:
  reduce `--max-num-seqs`, reduce request rate for diagnosis, or shorten the
  agent loop by keeping `MAX_ITERATIONS=3`.
- If prefill dominates:
  reduce prompt verbosity, trim schema rendering if possible, lower
  `--max-model-len`, or tune `--max-num-batched-tokens`.
- If decode dominates:
  make prompts demand SQL only, keep `max_tokens` small if configured, and
  remove explanations.
- If many requests use revise:
  improve `GENERATE_SQL_*` prompts so fewer requests need extra LLM calls.
- If KV cache usage is near saturation:
  lower `--max-model-len`, lower `--max-num-seqs`, or reduce concurrency.
- If agent latency is high but vLLM latency is acceptable:
  inspect Langfuse for sequential extra calls, revise loops, and slow
  verification prompts.

Record each tuning iteration in `REPORT.md`:

```text
Saw X -> hypothesized Y -> changed Z -> result was W.
```

Example:

```text
Saw p95 queue time spike above 2s at 10 RPS -> hypothesized too many concurrent sequences for the latency target -> changed max-num-seqs from 64 to 48 -> p95 agent latency dropped in the next load run.
```

Minimum useful evidence:

- Screenshot before the tuning change:
  `screenshots/grafana_before.png`
- Screenshot after the tuning change:
  `screenshots/grafana_after.png`
- Final load result:
  `results/load_10rps.json`
- Final eval result after tuning:
  `results/eval_after_tuning.json`

Save:

```text
results/load_smoke_2rps.json
results/load_5rps.json
results/load_10rps.json
results/eval_after_tuning.json
screenshots/grafana_before.png
screenshots/grafana_after.png
```

## Final Ordered Phase 1-3 Guide

This is the compact final checklist for the first three phases. The detailed
notes above explain the same work in more depth; this section is the practical
order to follow.

### Phase 1: vLLM Serving, Final Order

Goal: get a model server running on `:8000`, prove it answers OpenAI-compatible
requests, and expose `/metrics` for Prometheus.

1. Sync dependencies:

```bash
uv sync
uv run python -c "import transformers; print(transformers.__version__)"
```

Why: vLLM 0.10.2 needs a compatible Transformers 4.x resolution. The local fix
was `transformers>=4.55.2,<5.0`; this avoids the Qwen tokenizer crash seen with
an incompatible resolution.

2. Load the assignment data:

```bash
uv run python scripts/load_data.py
```

Why: the agent and eval phases depend on the generated SQLite DBs,
`evals/eval_set.jsonl`, and `load_test/perf_pool.jsonl`.

3. Start Docker services:

```bash
docker compose up -d
```

Why: this starts Grafana, Prometheus, Langfuse, and their backing services. It
does not start vLLM or the agent.

4. Start local vLLM for development:

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

If the 6 GB GPU reports insufficient free memory, lower
`--gpu-memory-utilization`, `--max-num-seqs`, or `--max-model-len`.

Why: local `Qwen/Qwen3-0.6B` is fast enough for implementation and debugging.
The final assignment target is H100 with `Qwen/Qwen3-30B-A3B-Instruct-2507`.

5. Validate vLLM:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Why: `http://localhost:8000` returning `{"detail":"Not Found"}` is normal. The
useful endpoints are `/health`, `/metrics`, `/v1/models`, and
`/v1/chat/completions`.

6. Save manual serving evidence:

```text
screenshots/vllm_manual_query.png
```

Why: the screenshot proves the serving endpoint was reachable and produced a
reasonable response.

### Phase 2: Grafana Dashboard, Final Order

Goal: build a Grafana dashboard that can explain latency, throughput, queueing,
token generation, and KV-cache behavior during eval/load.

1. Confirm Prometheus can scrape vLLM.

Open:

```text
http://localhost:9090
```

Use `Status -> Target health` and simple queries such as:

```promql
up
vllm:num_requests_running
vllm:request_success_total
```

Why: if Prometheus is not scraping vLLM, Grafana panels will be empty or stale.

2. Open Grafana:

```text
http://localhost:3000
```

Why: Grafana is the visual evidence layer for Phase 2 and Phase 6.

3. Add latency panels using P95 histogram queries:

```promql
histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_queue_time_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_prefill_time_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_decode_time_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(vllm:request_time_per_output_token_seconds_bucket[5m])) by (le))
```

Why: end-to-end latency alone says the system is slow; queue/prefill/decode/TTFT
show where it is slow.

4. Set readable legend names, not only query names.

Why: Grafana may show raw PromQL unless each query has an explicit legend/alias.
A reviewer should understand the graph without reading PromQL.

5. Add throughput, running request, generated-token, and KV-cache panels.

Useful queries:

```promql
rate(vllm:request_success_total[5m])
vllm:num_requests_running
rate(vllm:generation_tokens_total[5m])
vllm:kv_cache_usage_perc
rate(vllm:num_preemptions_total[5m])
```

Why: these panels explain whether the bottleneck is request volume, generation
work, queueing, or memory/cache pressure.

6. Save dashboard JSON and screenshot:

```text
infra/grafana/provisioning/dashboards/serving.json
screenshots/grafana_serving.png
```

Why: the JSON proves the dashboard is reproducible; the screenshot proves it
visibly reacts to traffic.

### Phase 3: Agent, Final Order

Goal: build the SQL agent loop:

```text
generate_sql -> execute -> verify -> revise if needed -> execute -> verify
```

1. Implement prompts in `agent/prompts.py`.

Why: Qwen can emit `<think>` blocks and explanations unless the prompt demands
SQL-only or JSON-only outputs. Tight prompts reduce parser failures.

2. Implement graph nodes in `agent/graph.py`:

```python
verify_node()
revise_node()
route_after_verify()
```

Why: `generate_sql_node()` alone only produces a first guess. The assignment
expects a verifier and revision loop.

3. Add robust output parsing.

Required behavior:

- strip `<think>` blocks before SQL extraction;
- extract full `SELECT ...` statements, not only the first word;
- extract JSON defensively from verifier output;
- mark the final response `ok=false` if the verifier rejects it.

Why: small local Qwen outputs can be verbose or malformed. The agent must fail
cleanly instead of returning `<think>` text as SQL.

4. Add deterministic verification guards for obvious failures.

Example: reject ID-only SQL when the question asks for names/list entries.

Why: the local model accepted `power_id` values for "List down Ajax's
superpowers." The deterministic guard forced a revision to select
`superpower.power_name`.

5. Compile-check:

```bash
python -m py_compile agent/graph.py agent/prompts.py agent/server.py
```

Why: this catches syntax errors before starting the service.

6. Start the agent:

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

7. Health-check:

```bash
curl http://localhost:8001/health
```

Expected:

```json
{"status":"ok"}
```

Why: `ERR_CONNECTION_REFUSED` on `:8001` means the agent is not running. This is
the most common cause of invalid Phase 5 and Phase 6 results.

8. Test the Ajax revise case:

```bash
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"List down Ajax'\''s superpowers.","db":"superhero"}'
```

Why: this is a concrete proof case. The successful final answer joins
`superhero -> hero_power -> superpower` and selects `power_name`, returning
`Agility`, `Super Strength`, `Super Speed`, `Heat Generation`, and `Power Suit`.

9. Save proof:

```text
results/phase3_ajax_revise_example.json
screenshots/phase3_agent_revise.png
```

Why: this gives both raw machine-readable proof and screenshot proof that the
verify/revise loop did real work.

## Final Ordered Phase 4-6 Guide

This section is the cleaned, final version of the latest Phase 4, Phase 5, and
Phase 6 guidance. It exists because the live troubleshooting added important
lessons that are easy to forget: restart the agent after changing `.env`, reject
connection-refused eval/load results, and always verify load JSON before trusting
Grafana screenshots.

### Phase 4: Langfuse Tracing, Final Order

Goal: prove that agent runs are traceable, inspectable, and filterable. The
evidence should show the `generate_sql`, `execute`, `verify`, and optional
`revise` waterfall.

1. Open Langfuse:

```text
http://localhost:3001
```

Why: Docker starts Langfuse, but it has no useful traces until a project exists
and the agent sends requests to it.

2. Create a local account and project, then copy the public and secret API keys.

Why: Langfuse uses project keys to accept trace uploads. Without them, the agent
may still answer questions, but the tracing UI will stay empty.

3. Add the keys to `.env` in the repo root:

```env
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=your_public_key_here
LANGFUSE_SECRET_KEY=your_secret_key_here
```

Why: `LANGFUSE_HOST` is the runtime variable used by the callback handler.
`LANGFUSE_BASE_URL` matches the UI wording and is mapped by `agent/server.py`.
Do not commit or zip `.env` because it contains secrets.

4. Restart the agent server:

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

Why: the agent loads `.env` at startup. If `.env` changes while the agent is
already running, the old process will not see the Langfuse keys.

5. Confirm health:

```bash
curl http://localhost:8001/health
```

Expected:

```json
{"status":"ok"}
```

Why: if the agent is down, there will be no traces, no valid eval, and no valid
load test.

6. Send a tagged request:

```bash
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question":"List down Ajax'\''s superpowers.",
    "db":"superhero",
    "tags":{
      "phase":"phase4_trace_check",
      "run_type":"manual",
      "model":"Qwen/Qwen3-0.6B"
    }
  }'
```

Why: tags make the trace filterable. This matters later when the trace list has
manual checks, baseline eval requests, and tuning requests mixed together.

7. Inspect Langfuse.

Look for:

- a new `LangGraph` trace;
- visible input/output, latency, and token counts;
- graph or LLM observations for generation and verification;
- a revise path when `iterations > 1`.

Why: the assignment asks for agent tracing, not only a final answer. The opened
trace proves the graph was instrumented.

8. Save screenshots:

```text
screenshots/langfuse_trace.png
screenshots/langfuse_tags.png
screenshots/langfuse_tags_eval_baseline.png
```

Why:

- `langfuse_trace.png` proves one detailed trace waterfall exists.
- `langfuse_tags.png` proves manual trace metadata/tags exist.
- `langfuse_tags_eval_baseline.png` proves the baseline eval generated many
  traces, not just one hand-picked manual request.

Troubleshooting:

- Empty Langfuse usually means the agent was not restarted after editing `.env`.
- If `POST /answer` works but traces do not appear, verify `.env` is in the repo
  root and includes `LANGFUSE_HOST=http://localhost:3001`.
- If `curl http://localhost:8001/health` fails, start the agent before debugging
  Langfuse.

### Phase 5: Evaluation, Final Order

Goal: measure execution accuracy. Do not compare SQL strings directly. Compare
the rows returned by the predicted SQL and the gold SQL on the same SQLite DB.

1. Keep vLLM running on `:8000`.

Why: each agent call can invoke vLLM more than once because the graph may run
`generate_sql`, `verify`, and `revise`.

2. Keep the agent running on `:8001`.

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

Why: `evals/run_eval.py` calls the agent HTTP endpoint. If the agent is down,
the eval JSON is only measuring an outage.

3. Confirm health before eval:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

Why: this prevents invalid eval output with `ConnectError` or connection refused
for every question.

4. Run baseline eval:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_baseline.json
```

Why: baseline quality is the reference point for Phase 6. Tuning is not useful
if it improves latency but destroys correctness.

5. Inspect the summary:

```bash
cat results/eval_baseline.json
```

Important fields:

- `summary.total`
- `summary.correct`
- `summary.pass_rate`
- `summary.agent_ok_rate`
- `summary.revision_rate`
- `summary.avg_iterations`
- `summary.per_iteration`

Why: `per_iteration` answers whether the verify/revise loop adds value. In the
local run, the final allowed iteration improved pass rate among emitted SQL
attempts from 3.8% to 7.7%.

6. Keep one clear revise example:

```text
results/phase3_ajax_revise_example.json
screenshots/phase3_agent_revise.png
```

Why: the Ajax example is easy to explain: the first query returned `power_id`
values, the verifier rejected the ID-only answer, and the final revision joined
to `superpower` and selected `power_name`.

7. After tuning, rerun eval:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_after_tuning.json
```

Why: this proves whether quality survived the serving changes.

Latest local results:

```text
Baseline eval: results/eval_baseline.json
Total: 30
Correct: 2
Execution accuracy: 6.7%
Revision rate: 53.3%
Average iterations: 1.9
P50 latency: 1.30s
P95 latency: 26.41s
Per-iteration pass rate: 3.8% -> 3.8% -> 7.7%
```

```text
Post-tuning eval: results/eval_after_tuning.json
Total: 30
Correct: 2
Execution accuracy: 6.7%
Revision rate: 56.7%
Average iterations: 2.0
Quality change: unchanged top-line pass rate, 2/30 -> 2/30
```

Troubleshooting:

- If `agent_error` says `ConnectError`, the agent was not running.
- If many items are HTTP 500, inspect the agent terminal and Langfuse traces for
  brittle model output, invalid SQL, or parser failures.
- A low pass rate is still valid evidence if the harness ran correctly. Report
  the failure modes honestly.

### Phase 6: Load Test and SLO Tuning, Final Order

Goal: measure latency under load, diagnose the bottleneck from metrics, change
one thing, and prove whether the result moved.

Target SLO:

```text
P95 end-to-end agent latency < 5 seconds at 10+ RPS for 5 minutes.
```

For final grading, this should come from the H100 run with
`Qwen/Qwen3-30B-A3B-Instruct-2507`. The current saved numbers are local laptop
development evidence with `Qwen/Qwen3-0.6B`.

1. Confirm health:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl http://localhost:8001/health
```

Why: the first after-tuning attempt was invalid because the agent was down.
`results/load_after_tuning.json` had 240/240 client errors and Grafana showed no
new live traffic.

2. Run baseline load:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 2 \
  --duration 120 \
  --out results/load_baseline.json
```

Why: baseline load establishes the before number. It must contain real
successful requests.

3. Save the before Grafana screenshot:

```text
screenshots/grafana_before.png
```

Why: the JSON says how slow the system was; Grafana explains why.

4. Optionally run an overload probe:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 4 \
  --duration 120 \
  --out results/load_baseline_rps4.json
```

Why: the overload probe shows what breaks first. Locally, 4 RPS pushed P95 to
27.86s and caused many failures, so it was above useful laptop capacity.

5. Choose one tuning change from the dashboard.

Use these interpretations:

- High queue time: reduce concurrency pressure or lower `--max-num-seqs`.
- High prefill time: shorten prompts/schema or tune `--max-num-batched-tokens`.
- High decode time: force SQL-only output and reduce output tokens.
- High KV-cache usage/preemptions: lower `--max-model-len` or
  `--max-num-seqs`.
- High agent latency with acceptable vLLM latency: inspect Langfuse for extra
  sequential calls and revise loops.

Why: changing one thing gives a defensible `saw X -> hypothesized Y -> changed Z
-> result W` explanation.

6. Restart vLLM if serving flags changed, then recheck health.

Why: vLLM flags are startup configuration. A changed command in notes does not
affect the running server.

7. Rerun the same load:

```bash
uv run python load_test/driver.py \
  --agent-url http://localhost:8001/answer \
  --rps 2 \
  --duration 120 \
  --out results/load_after_tuning.json
```

Why: same RPS and duration make the before/after comparison valid.

8. Save the after Grafana screenshot:

```text
screenshots/grafana_after.png
```

Why: it proves traffic reached vLLM after the tuning change. In the valid local
after screenshot, the traffic burst appears around `00:45-00:50`.

9. Validate the JSON before trusting the screenshot:

Reject the run if:

- `ok` is 0;
- `client_errors` equals total requests;
- errors contain `Cannot connect to host localhost:8001`;
- latency percentiles are `NaN`.

Why: Grafana can show old traffic in the selected time window. The load-driver
JSON is the source of truth for whether the run reached the agent.

10. Rerun post-tuning eval:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8001/answer \
  --out results/eval_after_tuning.json
```

Why: this checks whether latency tuning preserved quality.

Latest local Phase 6 evidence:

```text
Baseline load: results/load_baseline.json
Requested RPS: 2.0
Total requests: 240
OK: 196
HTTP errors: 40
Client errors: 4
P50: 2.44s
P95: 8.67s
```

```text
Overload probe: results/load_baseline_rps4.json
Requested RPS: 4.0
Total requests: 480
OK: 282
Timeouts: 8
HTTP errors: 59
Client errors: 131
P95: 27.86s
```

```text
After tuning: results/load_after_tuning.json
Requested RPS: 2.0
Total requests: 240
OK: 200
HTTP errors: 40
Client errors: 0
P50: 2.14s
P95: 6.73s
```

Interpretation:

- P95 improved from 8.67s to 6.73s.
- Client connection errors improved from 4 to 0.
- Quality stayed unchanged at 2/30.
- The local run still missed the SLO, so the report should be honest that final
  H100 validation was not completed in this snapshot.

Save:

```text
results/load_baseline.json
results/load_baseline_rps4.json
results/load_after_tuning.json
results/eval_after_tuning.json
screenshots/grafana_before.png
screenshots/grafana_after.png
screenshots/phase6___grafana_before.png
screenshots/phase6___grafana_after.png
```

## Report Notes

In `REPORT.md`, use final H100 values if the H100 run is available. If the
submission uses the local fallback evidence, be explicit that the values are
from `Qwen/Qwen3-0.6B` on the RTX 3060 laptop GPU:

```text
Baseline load P50 agent latency: 2.44s
Baseline load P95 agent latency: 8.67s
After-tuning P50 agent latency: 2.14s
After-tuning P95 agent latency: 6.73s
Eval pass rate before tuning: 6.7%
Eval pass rate after tuning: 6.7%
H100 final run: not completed in the current submission snapshot
```

Agent value paragraph should cite per-iteration pass rate and at least one
specific revise example from `history` or Langfuse.
