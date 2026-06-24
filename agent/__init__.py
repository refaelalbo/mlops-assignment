# Goal: Mark agent/ as the package containing the text-to-SQL service logic.
# Why: Imports such as `from agent.graph import graph` depend on this package
# boundary when scripts run from the repository root.
