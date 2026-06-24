"""Schema-rendering helper (provided complete).

# Goal: Convert a db_id such as "superhero" into schema text the LLM can read.
# Why: Text-to-SQL quality depends on showing the model valid tables, columns,
# and relationships before asking it to write SQL.

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

# Goal: Anchor all DB lookups to the repository, not the current shell folder.
# Why: The FastAPI server, eval runner, and tests may be launched from different
# working directories, but they must resolve the same SQLite files.
ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    # Goal: Translate the API/eval database id into a concrete SQLite path.
    # Why: Requests pass compact ids like "superhero"; execution needs a file.
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    # Goal: Make table and column names safe to embed in generated DDL text.
    # Why: Some BIRD columns contain spaces/punctuation or reserved words.
    return '"' + ident.replace('"', '""') + '"'


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    # Goal: Render a SQLite database as CREATE TABLE statements for the prompt.
    # Why: LLMs reason better from familiar DDL than from raw PRAGMA tuples.
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    # Goal: Open the DB read-only while introspecting schema metadata.
    # Why: Schema rendering should never mutate assignment data.
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        # Goal: Collect user tables in stable order.
        # Why: Stable prompt order makes runs easier to debug and compare.
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            # Goal: Reconstruct column names, types, primary keys, and NOT NULL.
            # Why: This gives the model enough structure to avoid invented fields.
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                col_lines.append(line)
            # Goal: Include declared foreign keys as relationship hints.
            # Why: Join paths such as hero_power -> superhero/superpower are what
            # let the model turn free text into valid multi-table SQL.
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                col_lines.append(
                    f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                )
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    # Goal: Expose which db_id values are available after data loading.
    # Why: Useful for diagnostics and for building UI/API validation later.
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
