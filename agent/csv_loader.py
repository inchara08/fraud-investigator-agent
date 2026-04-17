"""
CSV → NetworkX graph → graph_context builder.

Converts a flat transaction CSV into the same graph_context format that
query_graph produces via Gremlin, so the rest of the LangGraph pipeline
(check_memory, search_similar_cases, reason_about_patterns, explain_verdict)
runs completely unchanged in CSV mode.

Expected CSV columns (only transaction_id, sender_id, amount are required):
    transaction_id, sender_id, sender_name, receiver_id, receiver_name,
    amount, currency, timestamp, merchant_name, merchant_category,
    merchant_country, device_id, sender_is_flagged, sender_flag_reason,
    receiver_is_flagged, receiver_flag_reason
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import networkx as nx
import pandas as pd

REQUIRED_COLUMNS = {"transaction_id", "sender_id", "amount"}

TEMPLATE_CSV = """transaction_id,sender_id,sender_name,receiver_id,receiver_name,amount,currency,timestamp,merchant_name,merchant_category,merchant_country,device_id,sender_is_flagged,sender_flag_reason,receiver_is_flagged,receiver_flag_reason
tx_001,acc_alice,Alice Chen,acc_bob,Bob Smith,49.99,USD,2024-01-15 10:30:00,Amazon,retail,US,dev_iphone_x,False,,False,
tx_002,acc_carol,Carol Davis,acc_offshore,Pacific Trade Ltd,9800.00,USD,2024-01-15 23:47:00,Pacific Trade Ltd,wire_transfer,KY,dev_android_y,True,money_laundering,False,
tx_003,acc_dave,Dave Kim,acc_eve,Eve Torres,250.00,USD,2024-01-16 09:15:00,Cayman Settlements,wire_transfer,KY,dev_iphone_x,False,,True,structuring
tx_004,acc_frank,Frank Lee,acc_grace,Grace Park,120.00,USD,2024-01-16 14:22:00,FreshMart Groceries,retail,US,dev_tablet_z,False,,False,
"""


def get_template_csv() -> str:
    return TEMPLATE_CSV


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return list of missing required column names."""
    return sorted(REQUIRED_COLUMNS - set(df.columns))


def _flag_value(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val) if pd.notna(val) else False


def _str_or_none(val) -> Optional[str]:
    if pd.isna(val) or str(val).strip() == "":
        return None
    return str(val).strip()


def _risk_score(is_flagged: bool) -> float:
    if is_flagged:
        return round(0.85 + random.uniform(0, 0.1), 2)
    return round(0.05 + random.uniform(0, 0.2), 2)


def build_graph_from_df(df: pd.DataFrame) -> nx.DiGraph:
    """
    Parse a transaction DataFrame and build a NetworkX DiGraph.

    Vertices: transaction, account (sender/receiver), merchant, device
    Edges:    INITIATED, TO, USES
    Each node stores its properties as attributes (id, label, props dict).
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        tx_id = str(row["transaction_id"]).strip()
        sender_id = str(row["sender_id"]).strip()
        amount = float(row["amount"]) if pd.notna(row.get("amount")) else 0.0
        currency = _str_or_none(row.get("currency")) or "USD"
        timestamp = _str_or_none(row.get("timestamp")) or ""
        merchant_name = _str_or_none(row.get("merchant_name"))
        merchant_category = _str_or_none(row.get("merchant_category")) or "unknown"
        merchant_country = _str_or_none(row.get("merchant_country")) or "unknown"
        device_id = _str_or_none(row.get("device_id"))
        receiver_id = _str_or_none(row.get("receiver_id"))
        sender_name = _str_or_none(row.get("sender_name")) or sender_id
        receiver_name = _str_or_none(row.get("receiver_name")) or receiver_id
        sender_flagged = _flag_value(row.get("sender_is_flagged", False))
        sender_flag_reason = _str_or_none(row.get("sender_flag_reason")) or ""
        receiver_flagged = _flag_value(row.get("receiver_is_flagged", False))
        receiver_flag_reason = _str_or_none(row.get("receiver_flag_reason")) or ""

        # --- Transaction vertex ---
        if tx_id not in G:
            G.add_node(tx_id, label="transaction", props={
                "transaction_id": tx_id,
                "amount": amount,
                "currency": currency,
                "timestamp": timestamp,
                "merchant_category": merchant_category,
                "status": "uploaded",
            })

        # --- Sender account vertex ---
        if sender_id not in G:
            G.add_node(sender_id, label="account", props={
                "account_id": sender_id,
                "name": sender_name,
                "is_flagged": sender_flagged,
                "flag_reason": sender_flag_reason,
                "risk_score": _risk_score(sender_flagged),
            })
        elif sender_flagged and not G.nodes[sender_id]["props"].get("is_flagged"):
            # Upgrade flag status if this row has flag info
            G.nodes[sender_id]["props"]["is_flagged"] = True
            G.nodes[sender_id]["props"]["flag_reason"] = sender_flag_reason

        # Sender → Transaction
        G.add_edge(sender_id, tx_id, label="INITIATED")

        # --- Receiver account vertex ---
        if receiver_id:
            if receiver_id not in G:
                G.add_node(receiver_id, label="account", props={
                    "account_id": receiver_id,
                    "name": receiver_name,
                    "is_flagged": receiver_flagged,
                    "flag_reason": receiver_flag_reason,
                    "risk_score": _risk_score(receiver_flagged),
                })
            elif receiver_flagged and not G.nodes[receiver_id]["props"].get("is_flagged"):
                G.nodes[receiver_id]["props"]["is_flagged"] = True
                G.nodes[receiver_id]["props"]["flag_reason"] = receiver_flag_reason

            G.add_edge(tx_id, receiver_id, label="TO")

        # --- Merchant vertex ---
        if merchant_name:
            if merchant_name not in G:
                G.add_node(merchant_name, label="merchant", props={
                    "merchant_id": merchant_name,
                    "merchant_name": merchant_name,
                    "category": merchant_category,
                    "country": merchant_country,
                    "is_flagged": False,
                })
            G.add_edge(tx_id, merchant_name, label="TO")

        # --- Device vertex + USES edges ---
        if device_id:
            if device_id not in G:
                G.add_node(device_id, label="device", props={
                    "device_id": device_id,
                    "device_type": "unknown",
                    "is_flagged": False,
                })
            G.add_edge(sender_id, device_id, label="USES")
            if receiver_id:
                G.add_edge(receiver_id, device_id, label="USES")

    return G


def _node_to_vertex_dict(G: nx.DiGraph, node_id: str) -> dict:
    data = G.nodes[node_id]
    return {
        "type": "vertex",
        "id": node_id,
        "label": data.get("label", "unknown"),
        "props": data.get("props", {}),
    }


def _edge_to_dict(from_id: str, to_id: str, G: nx.DiGraph) -> dict:
    edge_data = G.edges.get((from_id, to_id), {})
    return {
        "type": "edge",
        "label": edge_data.get("label", "CONNECTED"),
        "from": from_id,
        "to": to_id,
    }


def _find_flagged_paths(G: nx.DiGraph, start: str, max_hops: int = 3) -> list:
    """
    BFS from start node up to max_hops, emitting full vertex/edge paths
    whenever a node with is_flagged=True is reached. Tracks visited nodes
    per path to avoid cycles (equivalent to Gremlin simplePath()).
    """
    paths = []
    # Queue entries: (current_node, path_so_far, visited_set)
    # path_so_far is a list of vertex/edge dicts
    queue = deque()
    queue.append((start, [_node_to_vertex_dict(G, start)], {start}))

    while queue:
        current, path, visited = queue.popleft()

        if len(path) > max_hops * 2 + 1:  # vertices + edges interleaved
            continue

        # Explore all neighbors (both directions for undirected-like traversal)
        neighbors = list(G.successors(current)) + list(G.predecessors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue

            neighbor_data = G.nodes[neighbor]
            neighbor_props = neighbor_data.get("props", {})

            # Build edge dict (check both directions)
            if G.has_edge(current, neighbor):
                edge = _edge_to_dict(current, neighbor, G)
            else:
                edge = _edge_to_dict(neighbor, current, G)

            new_path = path + [edge, _node_to_vertex_dict(G, neighbor)]
            new_visited = visited | {neighbor}

            if neighbor_props.get("is_flagged"):
                paths.append(new_path)
            else:
                queue.append((neighbor, new_path, new_visited))

    return paths[:100]  # safety cap, matches Gremlin limit


def get_graph_context(G: nx.DiGraph, tx_id: str) -> Optional[dict]:
    """
    Build the exact graph_context dict format that query_graph produces.

    Returns None if tx_id is not in the graph.
    """
    if tx_id not in G:
        return None

    # Direct context: transaction + 1-hop neighbors
    tx_props = G.nodes[tx_id].get("props", {})
    transaction_dict = {
        "id": tx_id,
        "label": "transaction",
        "props": tx_props,
    }

    neighbors = []
    for node in list(G.successors(tx_id)) + list(G.predecessors(tx_id)):
        node_data = G.nodes[node]
        neighbors.append({
            "id": node,
            "label": node_data.get("label", "unknown"),
            "props": node_data.get("props", {}),
        })

    direct_context = {
        "transaction": transaction_dict,
        "neighbors": neighbors,
    }

    # Flagged paths: multi-hop BFS
    flagged_paths = _find_flagged_paths(G, tx_id, max_hops=3)

    return {
        "direct_context": direct_context,
        "flagged_paths": flagged_paths,
    }
