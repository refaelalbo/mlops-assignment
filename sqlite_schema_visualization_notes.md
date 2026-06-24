# SQLite Schema Visualization Notes

This note captures the practical workflow used while inspecting the assignment
database `db_id="superhero"`.

The actual SQLite file is:

```text
mlops-assignment/data/bird/superhero.sqlite
```

In the assignment code, `db_id="superhero"` maps to:

```text
mlops-assignment/data/bird/superhero.sqlite
```

through `agent/schema.py`.

## Tool 1 - SQLite Viewer by Florian Klampfer

Use this VS Code extension for quick visual table browsing.

What it is good for:

- Opening `.sqlite` files directly in VS Code.
- Browsing table names.
- Clicking tables and seeing rows.
- Quickly checking columns and sample values.

Example tables visible in `superhero.sqlite`:

```text
alignment
attribute
colour
gender
hero_attribute
hero_power
publisher
race
superhero
superpower
```

Limitation:

- In the version tested, it did not expose a clear SQL query panel from the
  command palette.
- It is mainly a viewer/exporter, not the best tool for relationship diagrams.

## Tool 2 - SQLite by alexcvzz

Use this VS Code extension when you want a SQLite Explorer and query execution.

Workflow:

1. Press `Ctrl+Shift+P`.
2. Run:

```text
SQLite: Open Database
```

3. Select:

```text
mlops-assignment/data/bird/superhero.sqlite
```

4. Open the VS Code sidebar.
5. Expand:

```text
SQLITE EXPLORER
-> superhero.sqlite
```

6. Click the `New Query` button.
7. Paste and run SQL.

Query to show foreign-key relationships:

```sql
SELECT
  m.name AS table_name,
  p."from" AS column_name,
  p."table" AS referenced_table,
  p."to" AS referenced_column
FROM sqlite_master m
JOIN pragma_foreign_key_list(m.name) p
WHERE m.type = 'table'
ORDER BY m.name, p.id;
```

Expected relationship output includes:

```text
hero_attribute.hero_id -> superhero.id
hero_attribute.attribute_id -> attribute.id
hero_power.power_id -> superpower.id
hero_power.hero_id -> superhero.id
superhero.skin_colour_id -> colour.id
superhero.race_id -> race.id
superhero.publisher_id -> publisher.id
superhero.hair_colour_id -> colour.id
superhero.gender_id -> gender.id
superhero.eye_colour_id -> colour.id
superhero.alignment_id -> alignment.id
```

This is the relationship part of the schema.

## Tool 3 - DBeaver

Use DBeaver when you want a real visual schema graph / ER diagram.

Workflow:

1. Open DBeaver.
2. Create a SQLite connection.
3. Select:

```text
mlops-assignment/data/bird/superhero.sqlite
```

4. Open the database.
5. Use the ER diagram / view diagram feature.

The graph should show tables as boxes and foreign-key lines between them.

Important graph path for the Ajax example:

```text
superhero.id
-> hero_power.hero_id

hero_power.power_id
-> superpower.id
```

So the natural-language question:

```text
List down Ajax's superpowers.
```

maps to this SQL:

```sql
SELECT sp.power_name
FROM superhero s
JOIN hero_power hp ON s.id = hp.hero_id
JOIN superpower sp ON hp.power_id = sp.id
WHERE s.superhero_name = 'Ajax';
```

Expected result:

```text
Agility
Heat Generation
Power Suit
Super Speed
Super Strength
```

## What Counts As The Schema

The schema has two parts:

1. Tables and columns.
2. Foreign-key relationships between tables.

For example, `superhero` has columns:

```text
id
superhero_name
full_name
gender_id
eye_colour_id
hair_colour_id
skin_colour_id
race_id
publisher_id
alignment_id
height_cm
weight_kg
```

And the relationship graph says:

```text
superhero.gender_id -> gender.id
superhero.eye_colour_id -> colour.id
superhero.hair_colour_id -> colour.id
superhero.skin_colour_id -> colour.id
superhero.race_id -> race.id
superhero.publisher_id -> publisher.id
superhero.alignment_id -> alignment.id
```

Bridge tables:

```text
hero_power
hero_attribute
```

These exist because one superhero can have many powers/attributes, and one
power/attribute can belong to many superheroes.

## Why This Matters For Text-to-SQL

The LLM needs the schema to know how to join tables.

For the Ajax example, a weak query might return IDs:

```sql
SELECT power_id
FROM hero_power
WHERE hero_id = (
  SELECT id
  FROM superhero
  WHERE superhero_name = 'Ajax'
);
```

That SQL executes, but it does not answer the question because the user asked
for superpower names, not power IDs.

The correct schema-aware query joins through the bridge table and returns
`superpower.power_name`.

This is exactly why the assignment used:

```text
generate SQL
-> execute SQL
-> verify answer
-> revise SQL if needed
```

The verifier caught that returning only IDs was not enough.

## Quick Terminal Alternative

If VS Code tools are unclear, print relationships directly:

```powershell
python -c "import sqlite3; con=sqlite3.connect('mlops-assignment/data/bird/superhero.sqlite'); cur=con.cursor(); rows=cur.execute(\"SELECT m.name AS table_name, p.'from' AS column_name, p.'table' AS referenced_table, p.'to' AS referenced_column FROM sqlite_master m JOIN pragma_foreign_key_list(m.name) p WHERE m.type='table' ORDER BY m.name, p.id\").fetchall(); [print(f'{t}.{c} -> {rt}.{rc}') for t,c,rt,rc in rows]"
```

Print all table creation SQL:

```powershell
python -c "import sqlite3; con=sqlite3.connect('mlops-assignment/data/bird/superhero.sqlite'); cur=con.cursor(); rows=cur.execute(\"SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name\").fetchall(); [print(r[0] + '\\n') for r in rows]"
```
