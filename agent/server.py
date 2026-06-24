"""FastAPI wrapper exposing the agent over HTTP.

# Goal: Turn the LangGraph SQL agent into a small web service.
# Why: Eval and load-test scripts need a stable HTTP boundary instead of calling
# Python functions directly.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Goal: Load local .env values before constructing the LLM/tracing stack.
# Why: vLLM and Langfuse endpoints/keys are environment-configured.
load_dotenv()

# Langfuse's SDK reads LANGFUSE_HOST. Some setup snippets call the same value
# LANGFUSE_BASE_URL, so accept both to avoid silently missing traces.
if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

# Goal: Import the graph after environment normalization.
# Why: graph.py reads model endpoint settings at import time.
from agent.graph import AgentState, graph  # noqa: E402

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler

    # Goal: Attach Langfuse tracing to LangChain/LangGraph calls.
    # Why: The assignment requires evidence of generate/verify/revise spans.
    _lf_handler = CallbackHandler()


app = FastAPI()


class AnswerRequest(BaseModel):
    # Goal: Define the JSON contract accepted by POST /answer.
    # Why: FastAPI validates user input before the agent runs.
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    # Goal: Define the stable response shape consumed by eval/load scripts.
    # Why: Downstream metrics depend on SQL, rows, iteration count, and history.
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    # Goal: Provide a cheap readiness check for tunnels, load tests, and docs.
    # Why: It separates "server is up" from slower LLM/database behavior.
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    # Goal: Seed the graph state from the HTTP request.
    # Why: LangGraph passes this state through schema, generation, execution,
    # verification, and optional revision nodes.
    state = AgentState(question=req.question, db_id=req.db)
    # Goal: Pass tracing callbacks and user/run metadata into the graph.
    # Why: Tags make Langfuse traces filterable by phase, db_id, and run type.
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": req.tags,
    }
    try:
        # Goal: Run the full agent workflow synchronously for this HTTP request.
        # Why: The caller expects one final SQL/result after all revisions finish.
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    # Goal: Normalize graph output into the public API response fields.
    # Why: Graph internals can hold Python objects; HTTP returns JSON.
    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")
    verify_ok = final.get("verify_ok", False)
    verify_issue = final.get("verify_issue", "")

    if execution is None:
        # Goal: Fail explicitly if the graph never reached SQL execution.
        # Why: This points debugging at graph wiring rather than SQL quality.
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        # Goal: Surface database execution errors without pretending success.
        # Why: Eval and load summaries need to distinguish bad SQL from bad HTTP.
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )
    if not verify_ok:
        # Goal: Return executed rows but mark the answer as rejected.
        # Why: A query can execute and still fail the user's semantic request.
        return AnswerResponse(
            sql=sql,
            rows=[list(r) for r in (execution.rows or [])],
            iterations=iteration,
            ok=False,
            error=verify_issue or "agent verifier rejected the final result",
            history=history,
        )

    # Goal: Return the verified successful answer.
    # Why: This is the only path counted as an agent-level success.
    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
