# Implementing a Text-to-SQL Agent in a Company

This guide explains how to adapt the assignment SQL agent pattern for a real
company database. The assignment version is a useful prototype, but a company
deployment needs stronger guardrails before it can safely answer free-text
questions with SQL.

The final assignment rerun used a Nebius H100 VM with
`Qwen/Qwen3-30B-A3B-Instruct-2507`. That run is useful for company planning
because it shows both the value and the limits of the prototype:

- The verifier/revision loop improved SQL quality.
- The service produced observable traces and Grafana metrics.
- H100 made the larger model feasible.
- The measured run still did not satisfy the stated 10+ RPS SLO.
- The tuning change reduced serving pressure but was not a net latency win.

The recommended production flow is:

```text
user question
-> identify relevant schema and business definitions
-> generate SQL
-> validate SQL safety
-> dry-run or explain
-> execute with read-only permissions and limits
-> verify result against the question
-> return SQL, result, and caveats
```

## Core Principle

Never let an LLM directly execute arbitrary SQL against a production database.
The model should propose SQL. Your application should validate, limit, execute,
log, and explain it.

## What The Assignment Proved

The assignment was not only a model-serving exercise. It proved an end-to-end
pattern:

```text
question
-> schema context
-> LLM SQL generation
-> SQL execution
-> verifier decision
-> optional SQL revision
-> final answer
-> traces, metrics, and result JSONs
```

The most important positive result was the revision loop. On the final H100
baseline evaluation, the first generated SQL attempt solved 10 out of 30
questions. After verification and revision, the final score reached 17 out of
30. That is a meaningful improvement and supports using a verifier/reviser in a
company version.

The most important negative result was the SLO. The target was P95 agent latency
under 5 seconds at 10+ RPS over a 5-minute window. The final H100 load tests
used 2 RPS for 120 seconds and achieved about 1.33 RPS after request drain.
Therefore the company version should not assume that a bigger GPU alone solves
production throughput.

Final H100 metrics:

| Run | Result |
|---|---|
| Baseline eval | 17/30 correct, 56.7%, P95 2.250s |
| After-tuning eval | 16/30 correct, 53.3%, P95 2.239s |
| Baseline load | 206/240 OK, 0 timeouts, P95 4.894s |
| After-tuning load | 207/240 OK, 0 timeouts, P95 5.851s |

The tuning pass reduced `max_num_seqs` from 64 to 32 and
`max_num_batched_tokens` from 8192 to 4096. It slightly reduced client errors
from 4 to 3, but P95 load latency regressed. In a company, this should be
reported as a stability-oriented experiment, not as a performance win.

## Production Architecture Based On The Assignment

A company version should keep the assignment's core agent loop, but split it
into explicit services and control points:

```text
User / BI tool
-> API gateway and auth
-> Text-to-SQL service
-> schema retriever
-> LLM gateway / vLLM cluster
-> SQL safety validator
-> dry-run / query planner
-> read-only warehouse executor
-> verifier and answer composer
-> audit log, Langfuse traces, Prometheus metrics
```

The assignment ran as one local FastAPI service plus local Docker observability
services. That is fine for a prototype. In production, the sensitive parts
should be separated:

- Authentication and authorization should happen before schema retrieval.
- Schema retrieval should only return objects the user is allowed to query.
- SQL validation should run before database execution.
- Database execution should use a read-only role on a warehouse or replica.
- Traces and metrics should be retained for audit and debugging.

## How The H100 Result Changes Company Planning

The H100 run changes the planning assumptions in three ways.

First, model size matters. The local `Qwen/Qwen3-0.6B` run was useful for
development, but final planning should use the H100 `Qwen/Qwen3-30B-A3B`
results because they better represent the intended deployment model.

Second, serving capacity still needs real load testing. The H100 was powerful
enough to host the model, but the measured agent service did not reach the
assignment SLO. A production rollout needs load testing at the actual target:
10+ RPS, 5 minutes or longer, with representative questions and schemas.

Third, tuning must be judged by measured tradeoffs. Reducing concurrency reduced
pressure and slightly reduced client errors, but worsened load P95. A company
rollout should compare multiple tuning profiles, not stop after one change.

Recommended next H100 tuning matrix:

```text
Profile A: max_num_seqs=64, max_num_batched_tokens=8192
Profile B: max_num_seqs=32, max_num_batched_tokens=4096
Profile C: max_num_seqs=48, max_num_batched_tokens=8192
Profile D: max_num_seqs=32, max_num_batched_tokens=8192
Profile E: max_num_seqs=16, max_num_batched_tokens=4096
```

Each profile should be tested with:

```text
quality eval
2 RPS smoke load
10 RPS SLO load
error breakdown
Grafana screenshot
Langfuse trace sampling
```

## 1. Use Read-Only Access

What to do:

1. Create a dedicated database user for the SQL agent.
2. Grant only `SELECT` privileges on approved schemas, views, or tables.
3. Deny write and admin operations:
   - `INSERT`
   - `UPDATE`
   - `DELETE`
   - `DROP`
   - `ALTER`
   - `CREATE`
   - `TRUNCATE`
4. Prefer a read replica, analytics database, or warehouse instead of the primary
   production OLTP database.

Why:

Even if the model generates bad SQL, the database permission layer should make
destructive actions impossible. This is the most important guardrail because it
does not depend on prompt quality.

Example PostgreSQL pattern:

```sql
CREATE ROLE text_to_sql_reader LOGIN PASSWORD 'use-a-secret-manager';

GRANT CONNECT ON DATABASE analytics_db TO text_to_sql_reader;
GRANT USAGE ON SCHEMA analytics TO text_to_sql_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO text_to_sql_reader;

ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
GRANT SELECT ON TABLES TO text_to_sql_reader;
```

Better pattern:

```text
LLM agent -> read-only analytics views -> warehouse/read replica
```

Avoid:

```text
LLM agent -> production transactional database with broad user permissions
```

## 2. Whitelist Allowed SQL

What to do:

1. Parse generated SQL with a real SQL parser when possible.
2. Allow only read-only statements:
   - `SELECT`
   - optionally `WITH ... SELECT`
3. Reject multiple statements.
4. Reject comments if they are not needed.
5. Reject dangerous keywords even if permissions should block them.
6. Reject references to blocked schemas or tables.

Why:

String matching alone is brittle, but a basic whitelist still catches many bad
outputs before they reach the database. It also protects against prompt
injection, accidental model drift, and parser mistakes.

Example validation rules:

```text
Allowed:
- SELECT ...
- WITH cte AS (...) SELECT ...

Rejected:
- INSERT ...
- UPDATE ...
- DELETE ...
- DROP ...
- ALTER ...
- CREATE ...
- TRUNCATE ...
- CALL ...
- EXEC ...
- COPY ...
- SELECT ...; DELETE ...
```

Python implementation idea:

```python
FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "call", "exec", "execute", "copy", "merge",
}

def validate_sql_safety(sql: str) -> tuple[bool, str]:
    normalized = sql.strip().lower()
    if not normalized:
        return False, "empty SQL"
    if normalized.count(";") > 1:
        return False, "multiple statements are not allowed"
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False, "only SELECT/WITH queries are allowed"
    for word in FORBIDDEN:
        if word in normalized.split():
            return False, f"forbidden keyword: {word}"
    return True, ""
```

Production improvement:

Use a parser such as `sqlglot` to inspect statement type, tables, columns, and
query structure instead of relying only on string checks.

Company addition:

The validator should return structured reasons. That makes it possible to report
blocked queries clearly and measure which guardrails are triggered most often.

Example:

```json
{
  "allowed": false,
  "reason": "blocked_table",
  "detail": "table payroll.employee_salary is not available to this user"
}
```

## 3. Use Schema Retrieval Instead of Sending the Full Schema

What to do:

1. Store schema metadata in a searchable index:
   - table names
   - column names
   - column types
   - descriptions
   - primary keys
   - foreign keys
   - example values if safe
   - business definitions
2. For each user question, retrieve only relevant tables and relationships.
3. Send the model a compact schema context, not the whole database.
4. Include the table grain for each table.

Why:

Company schemas can have hundreds or thousands of tables. Sending everything
causes long prompts, higher latency, higher cost, and more wrong joins. Relevant
schema retrieval improves both accuracy and speed.

Example schema card:

```text
Table: orders
Grain: one row per customer order
Primary key: order_id
Columns:
- order_id: unique order id
- customer_id: customer who placed the order
- order_created_at: timestamp when order was created
- order_status: pending, paid, shipped, cancelled
- total_amount_usd: total paid amount in USD

Relationships:
- orders.customer_id -> customers.customer_id
```

Retrieval approach:

```text
question
-> embed/search table and column descriptions
-> choose top candidate tables
-> add directly connected foreign-key tables
-> pass compact schema cards to the LLM
```

Good prompt content:

```text
Use only the tables and columns listed below.
If the question cannot be answered from this schema, say so.
Do not invent table names or column names.
```

Assignment lesson:

The final report explicitly listed schema-linking as the first improvement to
make with more time. This is the highest-value company upgrade because it can
improve accuracy, reduce prompt size, and reduce latency at the same time.

## 4. Include Relationships and Primary Keys

What to do:

1. Document primary keys for every table.
2. Document foreign-key relationships.
3. Document many-to-one and many-to-many join paths.
4. Document table grain.
5. Document known bridge tables.

Why:

Most text-to-SQL mistakes come from wrong joins, duplicate multiplication, or
using an ID column when a name/description column is needed. Explicit
relationships reduce these errors.

Example relationship metadata:

```text
customers.customer_id is primary key.
orders.order_id is primary key.
orders.customer_id joins to customers.customer_id.
order_items.order_id joins to orders.order_id.
order_items.product_id joins to products.product_id.
```

Example table grain metadata:

```text
customers: one row per customer
orders: one row per order
order_items: one row per product line item within an order
products: one row per product
```

Why grain matters:

If the question asks "total revenue by customer", joining `orders` to
`order_items` may duplicate order-level totals unless the query aggregates at
the correct grain. The verifier should check for this class of mistake.

Recommended join instruction:

```text
When joining tables, use only the listed relationships.
Do not infer joins from similarly named columns unless listed.
Respect table grain to avoid duplicate counting.
```

Assignment lesson:

The Phase 3 Ajax example showed why names and relationships matter. The first
query returned IDs, but the question needed superpower names. The revised query
used the correct join path and selected the human-readable column. In a company
schema, the same issue appears constantly with customer IDs, account IDs,
product IDs, and employee IDs.

## 5. Add Query Limits, Timeouts, and Cost Controls

What to do:

1. Add a default `LIMIT` for exploratory detail queries.
2. Do not add `LIMIT` blindly to aggregate queries where it changes meaning.
3. Set database query timeout.
4. Set maximum returned rows.
5. Set maximum scanned data if the warehouse supports it.
6. Prefer `EXPLAIN` or dry-run before execution for expensive warehouses.

Why:

Generated SQL can accidentally scan huge tables, return millions of rows, or run
for too long. Limits protect the database and user experience.

Example application policy:

```text
Default row limit: 100
Maximum row limit: 1000
Query timeout: 30 seconds
Blocked if estimated scan exceeds budget
```

Example PostgreSQL timeout:

```sql
SET statement_timeout = '30s';
```

Example BigQuery controls:

```text
Use dry_run=True.
Reject query if estimated bytes processed exceeds the configured budget.
Set maximum_bytes_billed.
```

Limit policy:

```text
If query returns raw rows and has no LIMIT, add LIMIT 100.
If query is COUNT/SUM/AVG/GROUP BY, do not add LIMIT unless the SQL has ORDER BY
and asks for top/bottom results.
```

Company addition:

Separate model latency from database latency. The assignment load-test numbers
were end-to-end agent latency. In production, track at least:

```text
schema retrieval latency
LLM generation latency
SQL validation latency
database execution latency
verification latency
total response latency
```

This split makes tuning decisions more accurate. If P95 is high because of
database execution, changing vLLM batching will not fix the real bottleneck.

## 6. Run SQL in a Safe Environment

What to do:

1. Use a read replica or analytics warehouse.
2. Avoid the primary transactional database for generated SQL.
3. Route queries through a service layer, not direct client DB access.
4. Log every question, generated SQL, validation result, execution time, row
   count, and user id.
5. Add alerting for repeated failures, expensive queries, and blocked attempts.

Why:

Even read-only queries can hurt production if they scan large tables or lock
resources. A safe environment isolates generated SQL from critical application
traffic.

Recommended architecture:

```text
User
-> Text-to-SQL API
-> schema retriever
-> LLM
-> SQL validator
-> read-only query executor
-> analytics database/read replica
-> verifier
-> response
```

Audit log fields:

```text
timestamp
user_id
question
retrieved_schema_ids
generated_sql
validation_status
blocked_reason
execution_latency_ms
row_count
error
model_name
trace_id
```

Why audit logs matter:

They let you debug bad answers, investigate data access, measure adoption, and
prove compliance if the system touches sensitive company data.

Observability based on the assignment:

Use two observability layers:

1. Application and agent traces, such as Langfuse.
2. Serving and infrastructure metrics, such as Prometheus and Grafana.

Langfuse answers:

```text
What did the model see?
What SQL did it generate?
Did verification pass?
Was revision triggered?
Which step failed?
```

Grafana answers:

```text
Was vLLM serving traffic?
Was latency increasing?
Were requests queued?
Were tokens/sec stable?
Was KV-cache pressure visible?
```

## 7. Add Human Approval for Sensitive Queries

What to do:

1. Classify tables and columns by sensitivity:
   - public/internal
   - confidential
   - PII
   - financial
   - health
   - payroll
   - security logs
2. Detect sensitive columns in generated SQL.
3. Require approval or block export-like queries.
4. Mask or aggregate sensitive values when possible.
5. Enforce user-level authorization before SQL generation and before execution.

Why:

A syntactically correct query can still violate privacy or company policy.
Permissions, classification, and approval workflows prevent the agent from
becoming an uncontrolled data export tool.

Example sensitive-data policy:

```text
Allowed without approval:
- aggregate revenue by month
- count customers by region
- average response time by support team

Requires approval:
- list customer emails
- export transaction-level financial data
- show employee compensation
- join user identity with behavioral events

Blocked:
- secrets, tokens, passwords
- unrestricted PII exports
- queries outside the user's authorized business domain
```

Example response when approval is needed:

```text
This question requires access to sensitive customer-level data. I can provide an
aggregated version, or you can request approval for row-level access.
```

## Company Evaluation Plan

The assignment used 30 evaluation questions and compared execution results
against gold SQL. A company version should use the same idea, but the eval set
must be built from real business questions.

Recommended eval categories:

```text
simple lookup
aggregation
filtering by date
top/bottom ranking
multi-table join
many-to-many join
business metric definition
sensitive-data request
question that cannot be answered from allowed schema
ambiguous question
```

Track these metrics:

```text
execution accuracy
first-attempt accuracy
final-after-revision accuracy
revision rate
invalid SQL rate
blocked-query rate
P50/P95/P99 latency
database timeout rate
user-visible error rate
```

The company should keep a regression suite. Any prompt change, model change,
schema-retrieval change, or vLLM tuning change should rerun the eval before
being deployed.

## Company SLO Plan

The assignment target was under 5 seconds P95 at 10+ RPS, but the final H100 run
did not prove that target. A company should define staged SLOs instead of one
large goal.

Example staged SLOs:

```text
Internal alpha:
- 1-2 RPS
- P95 under 8 seconds
- no destructive SQL possible
- all queries logged

Internal beta:
- 5 RPS
- P95 under 6 seconds
- 70%+ execution accuracy on approved eval set
- less than 2% user-visible errors

Production target:
- 10+ RPS
- P95 under 5 seconds
- 80%+ execution accuracy on approved eval set
- zero unauthorized data access
```

This is more realistic than declaring production readiness after one H100 run.

## Company Rollout Plan

Recommended rollout:

1. Start with one department and one analytics domain.
2. Use read-only curated views, not raw production tables.
3. Build a 50-100 question eval set from real analyst questions.
4. Run the service in shadow mode where it generates SQL but does not execute
   for end users.
5. Let analysts compare generated SQL against known-good queries.
6. Enable execution for low-risk aggregate questions.
7. Add approval for sensitive or row-level requests.
8. Expand table coverage only after accuracy and audit results are stable.

Shadow mode is especially important. It lets the company collect failure cases
without exposing users to wrong answers.

## Production Guardrail Checklist

Use this checklist before connecting the system to real company data:

```text
[ ] Dedicated read-only DB user
[ ] Approved schemas/tables only
[ ] SQL parser or strict SQL validator
[ ] SELECT/WITH only
[ ] Multiple statements blocked
[ ] Dangerous keywords blocked
[ ] Query timeout configured
[ ] Row limit configured
[ ] Cost/dry-run checks for warehouse queries
[ ] Schema retrieval implemented
[ ] Primary keys and foreign keys documented
[ ] Table grain documented
[ ] Sensitive columns classified
[ ] User authorization enforced
[ ] Human approval path for sensitive data
[ ] Full audit logging
[ ] Langfuse or equivalent tracing
[ ] Evaluation set with company-specific questions
[ ] Monitoring for latency, errors, and blocked queries
[ ] Error taxonomy for wrong answers
[ ] Separate latency metrics per pipeline stage
[ ] H100 or GPU serving profile load-tested at target RPS
[ ] Rollback plan for prompt/model/serving changes
```

## Recommended Development Plan

1. Start with a small approved analytics schema.
2. Build schema cards for 5-10 important tables.
3. Add SQL generation with no execution.
4. Add SQL validation.
5. Add execution against a read-only replica.
6. Add verifier and revision loop.
7. Build a company-specific eval set.
8. Add tracing and audit logging.
9. Run with internal power users.
10. Expand schemas only after accuracy and safety are measured.
11. Run a production-like load test at the target RPS.
12. Publish a model card or internal system card with known limitations.

## What Not To Do

Avoid these patterns:

```text
LLM -> raw SQL -> production database
```

```text
Full company schema in every prompt
```

```text
Admin database user for generated SQL
```

```text
No logging because the tool is "only internal"
```

```text
Returning large PII result sets directly to chat
```

Also avoid:

```text
Claiming an SLO was met when the test ran below the target RPS
```

```text
Calling a tuning change successful just because one metric improved
```

```text
Using screenshots from a different run than the reported JSON results
```

## Summary

Yes, the assignment SQL agent can become a company text-to-SQL assistant. The
production version should be built as a governed query system: schema retrieval,
strict SQL validation, read-only execution, limits, tracing, authorization, and
human approval for sensitive data.

The final H100 rerun makes the lesson more concrete. The architecture is viable,
and the verifier/revision loop clearly adds value, but the prototype is not yet
production-ready. A company implementation needs stronger schema linking,
structured validation, better error analysis, staged SLOs, and load testing at
the actual target throughput.

The model should help write SQL. The application must remain responsible for
safety.
