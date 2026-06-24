"""SQL execution helper (provided complete).

# Goal: Run generated SQL safely and return structured execution evidence.
# Why: The agent cannot judge an answer from SQL text alone; it needs rows,
# columns, row counts, and errors from the database.

execute_sql() runs the agent's SQL against the target DB in read-only mode
and returns a structured ExecutionResult. The verify node consumes this
to decide whether the answer looks plausible.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agent.schema import db_path


@dataclass
class ExecutionResult:
    # Goal: Keep execution output in a small, typed container.
    # Why: Graph nodes can pass one object around instead of loose tuples.
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(self, max_rows: int = 10) -> str:
        """Compact text rendering for prompt context."""
        # Goal: Convert database output into concise verifier prompt text.
        # Why: The verifier LLM needs enough evidence without receiving huge
        # result sets that increase latency and prompt noise.
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "OK: 0 rows returned."
        cols = ", ".join(self.columns or [])
        # Goal: Preview only the first rows.
        # Why: Verification usually needs shape and examples, not every row.
        preview = "\n".join(
            " | ".join(str(c) for c in row) for row in (self.rows or [])[:max_rows]
        )
        more = f"\n... ({self.row_count - max_rows} more rows)" if self.row_count > max_rows else ""
        return f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"


def execute_sql(db_id: str, sql: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    """Run SQL against db_id's sqlite, return result or error."""
    # Goal: Resolve db_id to its SQLite file before execution.
    # Why: The API speaks in logical ids; sqlite3 needs a concrete path.
    path = db_path(db_id)
    try:
        # Goal: Use SQLite read-only URI mode.
        # Why: Generated SQL should not be able to modify the assignment DBs.
        with sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        ) as conn:
            # Goal: Execute once and collect both column names and rows.
            # Why: Correctness checks compare rows, while verifier prompts also
            # need selected column names to catch "IDs instead of names" errors.
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except Exception as e:  # noqa: BLE001
        # Goal: Return database errors as data instead of crashing the graph.
        # Why: Failed SQL is expected during agent iteration and should trigger
        # verification/revision rather than a server process failure.
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
