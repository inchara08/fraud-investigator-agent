"""
Vector-based pattern memory for fraud-investigator-agent.

Stores fraud pattern embeddings in Aerospike (as LIST bins) and retrieves
similar past cases using cosine similarity.

In production this would use Aerospike Vector Search (AVS) for sub-millisecond
ANN search at scale. Dev mode uses numpy cosine similarity over a full
AerospikeStore namespace scan — suitable for O(10–1000) stored patterns.
"""

import numpy as np
from datetime import datetime
from typing import Optional


def build_pattern_text(graph_context: dict, transaction_id: str) -> str:
    """
    Build a text description of the fraud pattern observed in graph_context.
    This text is what gets embedded — it must surface the signals that are
    shared across similar fraud cases (merchant category, countries, flag reasons).
    """
    direct = graph_context.get("direct_context", {})
    tx = direct.get("transaction", {})
    props = tx.get("props", {})

    merchant_category = props.get("merchant_category", "unknown")
    amount = props.get("amount", 0)
    currency = props.get("currency", "USD")

    neighbors = direct.get("neighbors", [])
    merchant_names = [
        str(n["props"].get("merchant_name", ""))
        for n in neighbors if n.get("label") == "merchant"
    ]
    countries = [
        str(n["props"].get("country", ""))
        for n in neighbors if n.get("label") == "merchant"
    ]

    flagged_paths = graph_context.get("flagged_paths", [])
    flagged_details = []
    for path in flagged_paths:
        for element in path:
            if not isinstance(element, dict):
                continue
            if element.get("label") == "account":
                p = element.get("props", {})
                if p.get("is_flagged"):
                    flagged_details.append(
                        f"account {element.get('id', '?')}: "
                        f"flag_reason={p.get('flag_reason', 'unknown')}, "
                        f"risk_score={p.get('risk_score', 0)}"
                    )

    lines = [
        f"Transaction: {transaction_id}",
        f"Merchant category: {merchant_category}",
        f"Transaction amount: {amount} {currency}",
        f"Graph structure: {len(neighbors)} direct neighbors, {len(flagged_paths)} flagged path(s)",
        f"Merchant names: {', '.join(merchant_names) or 'none'}",
        f"Countries: {', '.join(countries) or 'unknown'}",
        "Flagged accounts:",
    ]
    if flagged_details:
        lines += [f"  - {d}" for d in flagged_details]
    else:
        lines.append("  none")

    return "\n".join(lines)


def embed_pattern(text: str) -> list:
    """
    Embed a fraud pattern text using Google text-embedding-004.
    Uses the same GOOGLE_API_KEY already in .env. Free tier.
    Returns a 768-dimensional vector as a plain Python list.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embedder.embed_query(text)


def store_pattern_vector(
    store,
    transaction_id: str,
    verdict: str,
    pattern_text: str,
    embedding: list,
) -> None:
    """
    Persist a fraud pattern vector to AerospikeStore.
    The embedding is stored as a native Aerospike LIST bin.
    Idempotent — re-investigating the same transaction overwrites the record.

    Namespace: ("fraud_agent", "pattern_vectors")
    Separate from flagged_accounts namespace to avoid key collisions.
    """
    store.put(
        namespace=("fraud_agent", "pattern_vectors"),
        key=transaction_id,
        value={
            "transaction_id": transaction_id,
            "verdict": verdict,
            "pattern_text": pattern_text,
            "embedding": embedding,          # Aerospike stores as LIST type
            "stored_at": datetime.utcnow().isoformat(),
        },
    )


def search_similar_patterns(store, query_embedding: list, top_k: int = 3) -> list:
    """
    Find the top-k most similar past fraud patterns using cosine similarity.

    Retrieves all stored vectors via AerospikeStore namespace scan
    (store.search() with no query= — full scan of the pattern_vectors namespace).
    NOTE: store.search(query=...) raises NotImplementedError in v0.1 — do not use it.

    Returns list of dicts: [{transaction_id, verdict, similarity, pattern_text, stored_at}]
    sorted by similarity descending. Returns [] on cold start (no stored vectors).
    """
    try:
        items = store.search(("fraud_agent", "pattern_vectors"), limit=1000)
    except Exception:
        return []

    if not items:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm_val = np.linalg.norm(q)
    if q_norm_val < 1e-10:
        return []
    q_unit = q / q_norm_val

    scored = []
    for item in items:
        stored_embedding = item.value.get("embedding") if item.value else None
        if not stored_embedding:
            continue
        try:
            v = np.array(stored_embedding, dtype=np.float32)
            v_norm_val = np.linalg.norm(v)
            if v_norm_val < 1e-10:
                continue
            v_unit = v / v_norm_val
            similarity = float(np.dot(q_unit, v_unit))
            scored.append({
                "transaction_id": item.value.get("transaction_id", item.key),
                "verdict": item.value.get("verdict", "UNKNOWN"),
                "pattern_text": item.value.get("pattern_text", ""),
                "similarity": similarity,
                "stored_at": item.value.get("stored_at", ""),
            })
        except Exception:
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]
