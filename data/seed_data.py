"""
Seed data for fraud-investigator-agent.
Creates 4 demo scenarios in Aerospike Graph (or TinkerPop dev mode).
Safe to run multiple times — all operations use mergeV / mergeE (idempotent).

Usage:
    python data/seed_data.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from gremlin_python.driver import serializer
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T, Merge, Direction

GREMLIN_URL = os.getenv("GREMLIN_URL", "ws://localhost:8182/gremlin")


def get_traversal():
    connection = DriverRemoteConnection(
        GREMLIN_URL,
        "g",
        message_serializer=serializer.GraphBinarySerializersV1(),  # NOT V4
    )
    g = traversal().withRemote(connection)
    return g, connection


# ---------------------------------------------------------------------------
# Helpers — idempotent vertex / edge upserts
# ---------------------------------------------------------------------------

def upsert_vertex(g, vid: str, label: str, props: dict):
    """Create or update a vertex by ID."""
    create_props = {T.id: vid, T.label: label, **props}
    update_props = props  # on_match only updates properties, not label/id
    (
        g.merge_v({T.id: vid})
         .option(Merge.on_create, create_props)
         .option(Merge.on_match, update_props)
         .iterate()
    )


def upsert_edge(g, from_id: str, edge_label: str, to_id: str):
    """Create an edge between two vertices if it doesn't already exist."""
    (
        g.merge_e({T.label: edge_label, Direction.OUT: from_id, Direction.IN: to_id})
         .option(Merge.out_v, {T.id: from_id, T.label: "vertex"})
         .option(Merge.in_v, {T.id: to_id, T.label: "vertex"})
         .iterate()
    )


# ---------------------------------------------------------------------------
# Scenario 1: Clean transaction
# alice → INITIATED → tx_clean_001 → TRANSACTED_WITH → merchant_amazon
# bob   → RECEIVED  → tx_clean_001
# No flags anywhere. Expected verdict: CLEAN ~0.95
# ---------------------------------------------------------------------------

def seed_clean(g):
    print("  Seeding tx_clean_001 (CLEAN scenario)...")
    upsert_vertex(g, "acc_alice", "account", {
        "account_id": "acc_alice",
        "name": "Alice Nguyen",
        "is_flagged": False,
        "risk_score": 0.05,
    })
    upsert_vertex(g, "acc_bob", "account", {
        "account_id": "acc_bob",
        "name": "Bob Patel",
        "is_flagged": False,
        "risk_score": 0.03,
    })
    upsert_vertex(g, "tx_clean_001", "transaction", {
        "transaction_id": "tx_clean_001",
        "amount": 49.99,
        "currency": "USD",
        "timestamp": "2024-03-15T10:23:00Z",
        "merchant_category": "retail",
        "status": "completed",
    })
    upsert_vertex(g, "merchant_amazon", "merchant", {
        "merchant_id": "merchant_amazon",
        "merchant_name": "Amazon",
        "category": "retail",
        "country": "US",
    })
    upsert_edge(g, "acc_alice", "INITIATED", "tx_clean_001")
    upsert_edge(g, "acc_bob", "RECEIVED", "tx_clean_001")
    upsert_edge(g, "tx_clean_001", "TRANSACTED_WITH", "merchant_amazon")


# ---------------------------------------------------------------------------
# Scenario 2: Obvious fraud — direct link to flagged account
# carol (flagged, money_laundering) → INITIATED → tx_fraud_001
# Expected verdict: FRAUD ~0.98
# ---------------------------------------------------------------------------

def seed_obvious_fraud(g):
    print("  Seeding tx_fraud_001 (FRAUD scenario)...")
    upsert_vertex(g, "acc_carol", "account", {
        "account_id": "acc_carol",
        "name": "Carol Vance",
        "is_flagged": True,
        "flag_reason": "money_laundering",
        "risk_score": 0.97,
    })
    upsert_vertex(g, "tx_fraud_001", "transaction", {
        "transaction_id": "tx_fraud_001",
        "amount": 9800.00,
        "currency": "USD",
        "timestamp": "2024-03-16T02:14:00Z",
        "merchant_category": "wire_transfer",
        "status": "completed",
    })
    upsert_vertex(g, "merchant_shell_co", "merchant", {
        "merchant_id": "merchant_shell_co",
        "merchant_name": "Global Trade Partners LLC",
        "category": "financial_services",
        "country": "XX",
    })
    upsert_edge(g, "acc_carol", "INITIATED", "tx_fraud_001")
    upsert_edge(g, "tx_fraud_001", "TRANSACTED_WITH", "merchant_shell_co")


# ---------------------------------------------------------------------------
# Scenario 3: Subtle fraud — flagged account 2 hops away via shared merchant
# dave → INITIATED → tx_subtle_001 → TRANSACTED_WITH → merchant_offshore
#                                                       ← TRANSACTED_WITH ← tx_link_001 ← INITIATED ← eve (flagged)
# Expected verdict: SUSPICIOUS ~0.70
# ---------------------------------------------------------------------------

def seed_subtle_fraud(g):
    print("  Seeding tx_subtle_001 (SUSPICIOUS/subtle scenario)...")
    upsert_vertex(g, "acc_dave", "account", {
        "account_id": "acc_dave",
        "name": "David Kim",
        "is_flagged": False,
        "risk_score": 0.12,
    })
    upsert_vertex(g, "acc_eve", "account", {
        "account_id": "acc_eve",
        "name": "Eve Marchetti",
        "is_flagged": True,
        "flag_reason": "structuring",
        "risk_score": 0.91,
    })
    upsert_vertex(g, "tx_subtle_001", "transaction", {
        "transaction_id": "tx_subtle_001",
        "amount": 750.00,
        "currency": "USD",
        "timestamp": "2024-03-17T14:55:00Z",
        "merchant_category": "financial_services",
        "status": "completed",
    })
    upsert_vertex(g, "tx_link_001", "transaction", {
        "transaction_id": "tx_link_001",
        "amount": 680.00,
        "currency": "USD",
        "timestamp": "2024-03-10T09:30:00Z",
        "merchant_category": "financial_services",
        "status": "completed",
    })
    upsert_vertex(g, "merchant_offshore", "merchant", {
        "merchant_id": "merchant_offshore",
        "merchant_name": "Cayman Settlements Ltd",
        "category": "financial_services",
        "country": "KY",
    })
    upsert_edge(g, "acc_dave", "INITIATED", "tx_subtle_001")
    upsert_edge(g, "tx_subtle_001", "TRANSACTED_WITH", "merchant_offshore")
    upsert_edge(g, "acc_eve", "INITIATED", "tx_link_001")
    upsert_edge(g, "tx_link_001", "TRANSACTED_WITH", "merchant_offshore")


# ---------------------------------------------------------------------------
# Scenario 4: False positive
# frank → INITIATED → tx_fp_001
# frank → SHARES_DEVICE → grace (flagged but flag_reason="resolved_dispute")
# Expected verdict: SUSPICIOUS, then reasoning downgrades it ~0.60
# ---------------------------------------------------------------------------

def seed_false_positive(g):
    print("  Seeding tx_fp_001 (false positive scenario)...")
    upsert_vertex(g, "acc_frank", "account", {
        "account_id": "acc_frank",
        "name": "Frank Torres",
        "is_flagged": False,
        "risk_score": 0.18,
    })
    upsert_vertex(g, "acc_grace", "account", {
        "account_id": "acc_grace",
        "name": "Grace Okonkwo",
        "is_flagged": True,
        "flag_reason": "resolved_dispute",
        "risk_score": 0.20,
    })
    upsert_vertex(g, "tx_fp_001", "transaction", {
        "transaction_id": "tx_fp_001",
        "amount": 120.00,
        "currency": "USD",
        "timestamp": "2024-03-18T11:00:00Z",
        "merchant_category": "grocery",
        "status": "completed",
    })
    upsert_vertex(g, "device_shared_01", "device", {
        "device_id": "device_shared_01",
        "device_type": "mobile",
        "ip_address": "192.168.1.55",
    })
    upsert_vertex(g, "merchant_grocery", "merchant", {
        "merchant_id": "merchant_grocery",
        "merchant_name": "Whole Foods Market",
        "category": "grocery",
        "country": "US",
    })
    upsert_edge(g, "acc_frank", "INITIATED", "tx_fp_001")
    upsert_edge(g, "tx_fp_001", "TRANSACTED_WITH", "merchant_grocery")
    upsert_edge(g, "acc_frank", "USED_DEVICE", "device_shared_01")
    upsert_edge(g, "acc_grace", "USED_DEVICE", "device_shared_01")
    # Derived shared-device relationship
    upsert_edge(g, "acc_frank", "SHARES_DEVICE", "acc_grace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Connecting to Gremlin at {GREMLIN_URL} ...")
    g, conn = get_traversal()
    try:
        print("Seeding 4 demo scenarios (idempotent — safe to run multiple times):")
        seed_clean(g)
        seed_obvious_fraud(g)
        seed_subtle_fraud(g)
        seed_false_positive(g)

        # Verify
        count = g.V().count().next()
        print(f"\nDone. Total vertices in graph: {count}")
        print("\nDemo transaction IDs:")
        print("  tx_clean_001  → expected CLEAN    (~0.95)")
        print("  tx_fraud_001  → expected FRAUD    (~0.98)")
        print("  tx_subtle_001 → expected SUSPICIOUS (~0.70)")
        print("  tx_fp_001     → expected SUSPICIOUS (~0.60)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
