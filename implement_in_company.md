# Implementing a Text-to-SQL Agent in a Company

This guide explains how to adapt the assignment SQL agent pattern for a real
company database. The assignment version is a useful prototype, but a company
deployment needs stronger guardrails before it can safely answer free-text
questions with SQL.

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

## Summary

Yes, the assignment SQL agent can become a company text-to-SQL assistant. The
production version should be built as a governed query system: schema retrieval,
strict SQL validation, read-only execution, limits, tracing, authorization, and
human approval for sensitive data.

The model should help write SQL. The application must remain responsible for
safety.
