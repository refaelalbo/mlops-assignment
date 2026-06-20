"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are a careful text-to-SQL assistant.
Return exactly one SQLite SELECT query and nothing else.
Use only tables and columns from the provided schema.
Quote identifiers with double quotes when they contain spaces, punctuation, or reserved words.
Do not use markdown fences. Do not explain your answer.
Do not output hidden reasoning, chain-of-thought, or <think> blocks.
Your first token must be SELECT."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

Write the SQLite query that answers the question.
/no_think"""


VERIFY_SYSTEM = """You verify whether an executed SQL result plausibly answers a question.
Return only a compact JSON object with this exact shape:
{"ok": true, "issue": ""}
or
{"ok": false, "issue": "short reason"}

Mark ok=false when:
- the SQL execution errored,
- the result has zero rows but the question asks to list/show/find existing records,
- the selected columns clearly do not answer the question,
- the question asks for names, titles, labels, descriptions, or list entries but the SQL returns only IDs or foreign keys,
- the SQL ignores an important condition from the question.

Do not require revision merely because the result is small, a count is zero, or the query uses a different valid SQL style.
Do not output hidden reasoning, chain-of-thought, or <think> blocks.
Your first token must be {."""

VERIFY_USER = """Question:
{question}

SQL:
{sql}

Execution result:
{execution}

Does the execution result plausibly answer the question? Return only JSON.
/no_think"""


REVISE_SYSTEM = """You revise SQLite queries after a verifier found a problem.
Return exactly one corrected SQLite SELECT query and nothing else.
Use only tables and columns from the provided schema.
Preserve the user's requested filters, joins, ordering, aggregation, and limits.
Do not use markdown fences. Do not explain your answer.
Do not output hidden reasoning, chain-of-thought, or <think> blocks.
Your first token must be SELECT."""

REVISE_USER = """Schema:
{schema}

Question:
{question}

Previous SQL:
{sql}

Execution result:
{execution}

Verifier issue:
{issue}

Write a corrected SQLite query.
/no_think"""
