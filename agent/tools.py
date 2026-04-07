"""
Gremlin query tools for the fraud investigation agent.

All tools are SYNCHRONOUS — do not introduce async here; it conflicts with
Streamlit's event loop and the synchronous LangGraph stream().

A new DriverRemoteConnection is opened per investigation and closed in a
finally block. This avoids thread-safety issues in Streamlit.
"""

import os
from datetime import datetime
from typing import Optional

from gremlin_python.driver import serializer
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T
from langgraph.store.base import BaseStore

GREMLIN_URL = os.getenv("GREMLIN_URL", "ws://localhost:8182/gremlin")


# ---------------------------------------------------------------------------
# Gremlin connection factory
# ---------------------------------------------------------------------------

def open_gremlin_connection():
    """Open a new Gremlin WebSocket connection. Caller must close it."""
    conn = DriverRemoteConnection(
        GREMLIN_URL,
        "g",
        message_serializer=serializer.GraphBinarySerializersV1(),  # NOT V4
    )
    g = traversal().withRemote(conn)
    return g, conn


# ---------------------------------------------------------------------------
# Path serialization helper
# ---------------------------------------------------------------------------

def path_to_dict(path) -> list:
    """
    Convert a Gremlin Path object to a plain list of dicts.

    When .path() is used with .by(__.project(...)) the elements are already
    plain Python dicts — not raw Vertex/Edge objects. Handle both cases.
    """
    result = []
    for element in path.objects:
        if isinstance(element, dict):
            # Already projected by .by(__.project(...)) — add type if missing
            if "type" not in element:
                element = {"type": "vertex", **element}
            result.append(element)
        elif hasattr(element, "in_vertex"):
            # Raw Edge object
            result.append({
                "type": "edge",
                "label": element.label,
                "from": element.out_vertex.id,
                "to": element.in_vertex.id,
            })
        elif hasattr(element, "properties"):
            # Raw Vertex object
            props = {k: v for k, v in element.properties.items()}
            result.append({
                "type": "vertex",
                "id": element.id,
                "label": element.label,
                "props": props,
            })
        else:
            result.append({"type": "unknown", "value": str(element)})
    return result


# ---------------------------------------------------------------------------
# Tool 1: 1-hop neighborhood of a transaction
# ---------------------------------------------------------------------------

def fetch_transaction_context(transaction_id: str) -> dict:
    """
    Fetch the transaction vertex and its directly connected vertices
    (accounts, merchants, devices) via all edge types.
    Returns a dict with 'transaction' and 'neighbors' keys.
    """
    g, conn = open_gremlin_connection()
    try:
        # Get the transaction vertex itself
        tx_vertices = (
            g.V()
             .has("transaction_id", transaction_id)
             .project("id", "label", "props")
             .by(T.id)
             .by(T.label)
             .by(__.value_map().by(__.unfold()))
             .to_list()
        )

        if not tx_vertices:
            return {"error": f"Transaction '{transaction_id}' not found in graph", "neighbors": []}

        # Get all directly connected vertices
        neighbors = (
            g.V()
             .has("transaction_id", transaction_id)
             .both_e()
             .other_v()
             .project("id", "label", "props")
             .by(T.id)
             .by(T.label)
             .by(__.value_map().by(__.unfold()))
             .to_list()
        )

        return {
            "transaction": tx_vertices[0],
            "neighbors": neighbors,
        }
    except Exception as e:
        return {"error": str(e), "neighbors": []}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 2: Multi-hop traversal to find flagged accounts
# ---------------------------------------------------------------------------

def find_flagged_neighbors(transaction_id: str, max_hops: int = 3) -> list:
    """
    Traverse up to max_hops from the transaction vertex, emitting any
    flagged accounts found along the way. Returns the full graph path
    (vertices + edges) for each flagged account found.

    Uses simplePath() to prevent cycles on ring-shaped subgraphs.
    Capped at 100 results as a safety net for large graphs.
    """
    g, conn = open_gremlin_connection()
    try:
        raw_paths = (
            g.V()
             .has("transaction_id", transaction_id)
             .repeat(
                 __.both_e().other_v().simple_path()
             )
             .emit(__.has("is_flagged", True))
             .times(max_hops)
             .has("is_flagged", True)
             .path()
             .by(
                 __.project("id", "label", "props")
                   .by(T.id)
                   .by(T.label)
                   .by(__.value_map().by(__.unfold()))
             )
             .limit(100)
             .to_list()
        )
        return [path_to_dict(p) for p in raw_paths]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        conn.close()


def transaction_exists(transaction_id: str) -> bool:
    """Check whether a transaction vertex exists in the graph."""
    g, conn = open_gremlin_connection()
    try:
        count = g.V().has("transaction_id", transaction_id).count().next()
        return count > 0
    except Exception:
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 3: Check account memory (AerospikeStore)
# ---------------------------------------------------------------------------

def check_account_memory(store: BaseStore, account_ids: list) -> dict:
    """
    Query AerospikeStore for prior flags on the given account IDs.
    Returns a dict of {account_id: stored_value} for any accounts found.

    Note: store.get() returns an Item object; access .value for the payload.
    """
    results = {}
    for acc_id in account_ids:
        try:
            item = store.get(
                namespace=("fraud_agent", "flagged_accounts"),
                key=acc_id,
            )
            if item is not None:
                results[acc_id] = item.value
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Tool 4: Record a flagged account to AerospikeStore
# ---------------------------------------------------------------------------

def record_flagged_account(
    store: BaseStore,
    account_id: str,
    verdict: str,
    reason: str,
    transaction_id: str,
) -> None:
    """
    Persist a flagged account to AerospikeStore so future investigations
    can recall this prior flag automatically (cross-session memory).
    """
    store.put(
        namespace=("fraud_agent", "flagged_accounts"),
        key=account_id,
        value={
            "verdict": verdict,
            "reason": reason,
            "transaction_id": transaction_id,
            "flagged_at": datetime.utcnow().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Helper: extract account IDs from graph context
# ---------------------------------------------------------------------------

def extract_account_ids(graph_context: Optional[dict]) -> list:
    """
    Pull all account vertex IDs from the graph_context dict produced by
    fetch_transaction_context and find_flagged_neighbors.
    """
    if not graph_context:
        return []

    ids = set()

    # From direct neighbors
    for neighbor in graph_context.get("neighbors", []):
        if neighbor.get("label") == "account":
            ids.add(neighbor["id"])

    # From flagged paths
    for path in graph_context.get("flagged_paths", []):
        for element in path:
            if isinstance(element, dict) and element.get("label") == "account":
                ids.add(element["id"])

    return list(ids)
