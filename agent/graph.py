"""
LangGraph state machine for fraud investigation.

Flow: START → ingest_transaction → query_graph → check_memory
            → reason_about_patterns → explain_verdict → END

The graph is compiled with:
  - AerospikeSaver as checkpointer  → enables checkpoint-based resume
  - AerospikeStore as store         → enables cross-session memory

Thread ID f"fraud-{transaction_id}" means re-submitting the same tx_id
after a crash resumes from the last saved checkpoint node.
"""

import json
import os
import time
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent.prompts import REASONING_PROMPT, SYSTEM_PROMPT
from agent.tools import (
    check_account_memory,
    extract_account_ids,
    fetch_transaction_context,
    find_flagged_neighbors,
    record_flagged_account,
    transaction_exists,
)

load_dotenv()

_LLM_MODEL = "gemini-2.5-flash"
_RETRY_ATTEMPTS = 3
_RETRY_WAIT = 15  # seconds between retries on 429


def _is_rate_limit(exc: Exception) -> bool:
    err = str(exc)
    return "429" in err or "RESOURCE_EXHAUSTED" in err


def _invoke_with_retry(messages: list, **kwargs):
    """Invoke gemini-2.5-flash, retrying up to 3x on 429 with a 15s wait."""
    llm = ChatGoogleGenerativeAI(model=_LLM_MODEL, **kwargs)
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if _is_rate_limit(e) and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_WAIT)
                continue
            raise


def _invoke_structured_with_retry(messages: list, schema, **kwargs):
    """Same as _invoke_with_retry but uses with_structured_output."""
    llm = ChatGoogleGenerativeAI(model=_LLM_MODEL, **kwargs).with_structured_output(schema)
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if _is_rate_limit(e) and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_WAIT)
                continue
            raise


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FraudInvestigationState(TypedDict):
    transaction_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    graph_context: Optional[dict]
    memory_context: Optional[dict]
    similar_cases: Optional[list]
    verdict: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    graph_path: Optional[list]
    risk_factors: Optional[list]
    investigation_complete: bool


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class FraudVerdict(BaseModel):
    verdict: str = Field(description="One of: FRAUD, SUSPICIOUS, CLEAN")
    confidence: float = Field(
        description="Confidence score 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    explanation: str = Field(
        description="Natural language explanation for a compliance officer"
    )
    graph_path_summary: Optional[str] = Field(
        default=None,
        description="Key graph path or combination of signals that triggered this verdict"
    )
    risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Specific risk signals identified (empty list if CLEAN)"
    )


# ---------------------------------------------------------------------------
# Node 1: ingest_transaction
# ---------------------------------------------------------------------------

def ingest_transaction(state: FraudInvestigationState) -> dict:
    tx_id = state["transaction_id"]
    # CSV mode: graph_context pre-built from uploaded data — skip Gremlin lookup
    if state.get("graph_context") is not None:
        return {
            "messages": [
                HumanMessage(content=f"Investigate transaction: {tx_id}"),
                AIMessage(content=f"Transaction `{tx_id}` loaded from uploaded dataset."),
            ],
            "investigation_complete": False,
        }
    if not transaction_exists(tx_id):
        return {
            "messages": [
                AIMessage(content=f"ERROR: Transaction '{tx_id}' not found in graph. "
                                   "Please run data/seed_data.py first.")
            ],
            "investigation_complete": True,
            "verdict": "ERROR",
        }
    return {
        "messages": [
            HumanMessage(content=f"Investigate transaction: {tx_id}"),
            AIMessage(content=f"Starting fraud investigation for transaction `{tx_id}`."),
        ],
        "investigation_complete": False,
    }


# ---------------------------------------------------------------------------
# Node 2: query_graph
# ---------------------------------------------------------------------------

def query_graph(state: FraudInvestigationState) -> dict:
    tx_id = state["transaction_id"]

    # CSV mode: graph_context already built from uploaded data — skip Gremlin
    if state.get("graph_context") is not None:
        ctx = state["graph_context"]
        n_neighbors = len(ctx.get("direct_context", {}).get("neighbors", []))
        n_paths = len(ctx.get("flagged_paths", []))
        return {
            "messages": [AIMessage(
                content=(
                    f"Graph built from uploaded data — "
                    f"{n_neighbors} connected entity/entities, "
                    f"{n_paths} flagged path(s) discovered."
                )
            )]
        }

    direct_context = fetch_transaction_context(tx_id)
    flagged_paths = find_flagged_neighbors(tx_id, max_hops=3)

    graph_context = {
        "direct_context": direct_context,
        "flagged_paths": flagged_paths,
    }

    n_paths = len(flagged_paths)
    n_neighbors = len(direct_context.get("neighbors", []))
    msg = (
        f"Graph traversal complete. "
        f"Found {n_neighbors} direct neighbor(s) and "
        f"{n_paths} path(s) to flagged account(s)."
    )
    return {
        "graph_context": graph_context,
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Node 3: check_memory  (store injected by LangGraph via param annotation)
# ---------------------------------------------------------------------------

def check_memory(state: FraudInvestigationState, store: BaseStore) -> dict:
    graph_context = state.get("graph_context") or {}
    account_ids = extract_account_ids(graph_context)

    prior_flags = check_account_memory(store, account_ids) if account_ids else {}

    if prior_flags:
        flagged_names = list(prior_flags.keys())
        msg = (
            f"Memory recall: {len(prior_flags)} account(s) from this graph "
            f"were previously flagged: {flagged_names}"
        )
    else:
        msg = "Memory recall: no accounts in this graph have been flagged in prior investigations."

    return {
        "memory_context": prior_flags,
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Node 4a: search_similar_cases
# ---------------------------------------------------------------------------

def search_similar_cases(state: FraudInvestigationState, store: BaseStore) -> dict:
    from agent.vector_memory import build_pattern_text, embed_pattern, search_similar_patterns
    try:
        pattern_text = build_pattern_text(
            state.get("graph_context") or {}, state["transaction_id"]
        )
        embedding = embed_pattern(pattern_text)
        similar = search_similar_patterns(store, embedding, top_k=3)
    except Exception as e:
        return {
            "similar_cases": [],
            "messages": [AIMessage(content=f"Vector similarity search skipped: {e}")],
        }

    if similar:
        lines = [f"- `{c['transaction_id']}` — **{c['verdict']}** — {c['similarity']:.0%} similar" for c in similar]
        msg = "Vector similarity search found similar past cases:\n" + "\n".join(lines)
    else:
        msg = "Vector similarity search: no stored patterns yet (cold start)."

    return {
        "similar_cases": similar,
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Node 4b: reason_about_patterns
# ---------------------------------------------------------------------------

def reason_about_patterns(state: FraudInvestigationState) -> dict:
    graph_data = json.dumps(state.get("graph_context") or {}, indent=2, default=str)
    memory_data = json.dumps(state.get("memory_context") or {}, indent=2, default=str)

    similar = state.get("similar_cases") or []
    if similar:
        similar_lines = [
            f"- {c['transaction_id']} | verdict={c['verdict']} | similarity={c['similarity']:.2f}\n  {c['pattern_text'][:200]}"
            for c in similar
        ]
        similar_cases_text = "\n".join(similar_lines)
    else:
        similar_cases_text = "No similar past cases found."

    prompt = REASONING_PROMPT.format(
        graph_data=graph_data,
        memory_data=memory_data,
        similar_cases=similar_cases_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = _invoke_with_retry(messages, temperature=0, max_output_tokens=2048)
    reasoning_text = response.content

    return {
        "messages": [AIMessage(content=reasoning_text)],
    }


# ---------------------------------------------------------------------------
# Node 5: explain_verdict
# ---------------------------------------------------------------------------

def explain_verdict(state: FraudInvestigationState, store: BaseStore) -> dict:

    # Build a concise prompt from everything gathered so far
    graph_data = json.dumps(state.get("graph_context") or {}, indent=2, default=str)
    memory_data = json.dumps(state.get("memory_context") or {}, indent=2, default=str)

    # Include prior reasoning messages for context
    prior_reasoning = ""
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and len(msg.content) > 100:
            prior_reasoning = msg.content
            break  # use the first substantive AI message (reason node output)

    verdict_prompt = f"""Based on the analysis below, produce a structured fraud verdict.

Graph data:
{graph_data}

Prior memory:
{memory_data}

Reasoning so far:
{prior_reasoning}

Return a structured verdict with: verdict (FRAUD/SUSPICIOUS/CLEAN), confidence (0-1),
explanation, graph_path_summary, and risk_factors list."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": verdict_prompt},
    ]

    try:
        result: FraudVerdict = _invoke_structured_with_retry(
            messages, FraudVerdict, temperature=0, max_output_tokens=4096
        )
    except Exception as parse_err:
        # Structured output failed (truncated JSON / schema mismatch).
        # Fall back to a plain LLM call and extract the verdict manually.
        plain_prompt = verdict_prompt + (
            "\n\nRespond with ONLY a JSON object using these exact keys: "
            "verdict, confidence, explanation, graph_path_summary, risk_factors"
        )
        plain_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": plain_prompt},
        ]
        raw = _invoke_with_retry(plain_messages, temperature=0, max_output_tokens=4096)
        raw_text = raw.content.strip()
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        try:
            data = json.loads(raw_text)
            result = FraudVerdict(
                verdict=data.get("verdict", "SUSPICIOUS"),
                confidence=float(data.get("confidence", 0.5)),
                explanation=data.get("explanation", str(parse_err)),
                graph_path_summary=data.get("graph_path_summary") or "",
                risk_factors=data.get("risk_factors") or [],
            )
        except Exception:
            # Last resort: return SUSPICIOUS with the raw text as explanation
            result = FraudVerdict(
                verdict="SUSPICIOUS",
                confidence=0.5,
                explanation=raw_text[:500] if raw_text else str(parse_err),
                graph_path_summary="",
                risk_factors=[],
            )

    # Persist to AerospikeStore if flagged
    if result.verdict in ("FRAUD", "SUSPICIOUS"):
        account_ids = extract_account_ids(state.get("graph_context") or {})
        for acc_id in account_ids:
            record_flagged_account(
                store=store,
                account_id=acc_id,
                verdict=result.verdict,
                reason=result.graph_path_summary or "",
                transaction_id=state["transaction_id"],
            )

    # Store pattern vector for all verdicts (CLEAN included — useful negative examples)
    try:
        from agent.vector_memory import build_pattern_text, embed_pattern, store_pattern_vector
        pattern_text = build_pattern_text(
            state.get("graph_context") or {}, state["transaction_id"]
        )
        embedding = embed_pattern(pattern_text)
        store_pattern_vector(store, state["transaction_id"], result.verdict, pattern_text, embedding)
    except Exception:
        pass  # non-fatal

    # Extract the path from graph_context for UI visualization
    graph_path = []
    flagged_paths = (state.get("graph_context") or {}).get("flagged_paths", [])
    if flagged_paths:
        graph_path = flagged_paths[0]  # show the first flagged path
    else:
        # Fall back to direct neighbors
        direct = (state.get("graph_context") or {}).get("direct_context", {})
        neighbors = direct.get("neighbors", [])
        tx = direct.get("transaction")
        if tx:
            graph_path.append({"type": "vertex", "id": tx["id"], "label": tx["label"], "props": tx.get("props", {})})
        for n in neighbors:
            graph_path.append({"type": "vertex", "id": n["id"], "label": n["label"], "props": n.get("props", {})})

    verdict_msg = (
        f"**Verdict: {result.verdict}** "
        f"(confidence: {result.confidence:.0%})\n\n"
        f"{result.explanation}"
    )

    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "graph_path": graph_path,
        "risk_factors": result.risk_factors,
        "investigation_complete": True,
        "messages": [AIMessage(content=verdict_msg)],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_fraud_graph(checkpointer, store):
    """
    Compile the LangGraph state machine with Aerospike checkpointer and store.

    checkpointer: AerospikeSaver — persists execution state for resume
    store:        AerospikeStore — long-term cross-session memory
    """
    builder = StateGraph(FraudInvestigationState)

    builder.add_node("ingest_transaction", ingest_transaction)
    builder.add_node("query_graph", query_graph)
    builder.add_node("check_memory", check_memory)
    builder.add_node("search_similar_cases", search_similar_cases)
    builder.add_node("reason_about_patterns", reason_about_patterns)
    builder.add_node("explain_verdict", explain_verdict)

    builder.add_edge(START, "ingest_transaction")
    builder.add_edge("ingest_transaction", "query_graph")
    builder.add_edge("query_graph", "check_memory")
    builder.add_edge("check_memory", "search_similar_cases")
    builder.add_edge("search_similar_cases", "reason_about_patterns")
    builder.add_edge("reason_about_patterns", "explain_verdict")
    builder.add_edge("explain_verdict", END)

    return builder.compile(checkpointer=checkpointer, store=store)
