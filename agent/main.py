"""
Streamlit UI for fraud-investigator-agent.

Layout: 3 columns [1, 2, 1]
  Left:   Transaction ID input + quick-demo selectbox + Investigate button
  Center: Real-time agent reasoning (one expander per node as it streams)
  Right:  Graph path visualization (emoji per vertex type, 🚨 for flagged)
  Bottom: Verdict card (colored by verdict: red/orange/green)

Key design decisions:
  - Synchronous graph.stream() — avoids asyncio/Streamlit event loop conflict
  - @st.cache_resource — Aerospike client created once per process, not per rerun
  - st.session_state — guards against stream re-run on Streamlit script reruns
"""

import json
import os
import sys

# Ensure the project root is on sys.path so 'agent' is importable
# regardless of which directory Streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from agent.graph import build_fraud_graph
from agent.memory import get_aerospike_client, get_checkpointer, get_store

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Investigator",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources — one Aerospike connection per Streamlit process
# ---------------------------------------------------------------------------

@st.cache_resource
def get_cached_resources():
    client = get_aerospike_client()
    checkpointer = get_checkpointer(client)
    store = get_store(client)
    graph = build_fraud_graph(checkpointer, store)
    return graph, store, checkpointer


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "stream_log" not in st.session_state:
    st.session_state.stream_log = []  # list of (node_name, update_dict)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 Financial Fraud Investigator")
st.caption("Powered by LangGraph + Aerospike Graph + Claude")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

col_input, col_reasoning, col_graph = st.columns([1, 2, 1])

# ---------------------------------------------------------------------------
# Left column — input
# ---------------------------------------------------------------------------

with col_input:
    st.subheader("Investigation")

    quick_demo = st.selectbox(
        "Quick demo scenario",
        options=["", "tx_clean_001", "tx_fraud_001", "tx_subtle_001", "tx_fp_001"],
        format_func=lambda x: {
            "": "— select a demo —",
            "tx_clean_001": "tx_clean_001 (CLEAN)",
            "tx_fraud_001": "tx_fraud_001 (FRAUD)",
            "tx_subtle_001": "tx_subtle_001 (SUSPICIOUS)",
            "tx_fp_001": "tx_fp_001 (False Positive)",
        }.get(x, x),
    )

    manual_input = st.text_input(
        "Or enter a Transaction ID",
        placeholder="e.g. tx_fraud_001",
        value="",
    )

    # Quick demo overrides manual input
    tx_id = manual_input.strip() or quick_demo

    investigate_btn = st.button(
        "Investigate",
        type="primary",
        disabled=not tx_id,
        use_container_width=True,
    )

    st.divider()
    st.caption("Demo scenarios:")
    st.markdown("""
- `tx_clean_001` → CLEAN
- `tx_fraud_001` → FRAUD
- `tx_subtle_001` → SUSPICIOUS
- `tx_fp_001` → false positive
    """)


# ---------------------------------------------------------------------------
# Run investigation when button clicked
# ---------------------------------------------------------------------------

if investigate_btn and tx_id:
    # Clear previous results
    st.session_state.last_result = None
    st.session_state.stream_log = []

    graph, store, checkpointer = get_cached_resources()

    config = {"configurable": {"thread_id": f"fraud-{tx_id}"}}
    initial_state = {
        "transaction_id": tx_id,
        "messages": [],
        "graph_context": None,
        "memory_context": None,
        "similar_cases": None,
        "verdict": None,
        "confidence": None,
        "explanation": None,
        "graph_path": None,
        "risk_factors": None,
        "investigation_complete": False,
    }

    with col_reasoning:
        st.subheader("Agent Reasoning")
        status_placeholder = st.empty()

        try:
            for event in graph.stream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                for node_name, update in event.items():
                    # Store in session state for re-renders
                    st.session_state.stream_log.append((node_name, update))

                    node_labels = {
                        "ingest_transaction": "1️⃣ Ingest Transaction",
                        "query_graph": "2️⃣ Query Graph",
                        "check_memory": "3️⃣ Check Memory",
                        "search_similar_cases": "🔍 Vector Similarity Search",
                        "reason_about_patterns": "4️⃣ Reason About Patterns",
                        "explain_verdict": "5️⃣ Explain Verdict",
                    }
                    label = node_labels.get(node_name, f"Node: {node_name}")

                    with st.expander(label, expanded=True):
                        # Show messages
                        for msg in update.get("messages", []):
                            if hasattr(msg, "content"):
                                st.markdown(msg.content)

                        # Show graph context (collapsible JSON)
                        if update.get("graph_context"):
                            with st.expander("Raw graph data", expanded=False):
                                st.json(update["graph_context"])

                        # Show memory context
                        if update.get("memory_context"):
                            with st.expander("Memory hits", expanded=False):
                                st.json(update["memory_context"])

                        # Show similar cases from vector search
                        if update.get("similar_cases"):
                            with st.expander("Similar past cases", expanded=True):
                                for case in update["similar_cases"]:
                                    st.markdown(
                                        f"`{case['transaction_id']}` — **{case['verdict']}** — "
                                        f"{case['similarity']:.0%} similar"
                                    )
                                    with st.expander("Pattern detail", expanded=False):
                                        st.text(case.get("pattern_text", ""))

            # Fetch final state after stream completes
            final_state = graph.get_state(config).values
            st.session_state.last_result = final_state

        except Exception as e:
            st.error(f"Investigation error: {e}")
            status_placeholder.empty()


# ---------------------------------------------------------------------------
# Center column — render stream log on re-renders (no re-run)
# ---------------------------------------------------------------------------

elif st.session_state.stream_log:
    with col_reasoning:
        st.subheader("Agent Reasoning")
        node_labels = {
            "ingest_transaction": "1️⃣ Ingest Transaction",
            "query_graph": "2️⃣ Query Graph",
            "check_memory": "3️⃣ Check Memory",
            "reason_about_patterns": "4️⃣ Reason About Patterns",
            "explain_verdict": "5️⃣ Explain Verdict",
        }
        for node_name, update in st.session_state.stream_log:
            label = node_labels.get(node_name, f"Node: {node_name}")
            with st.expander(label, expanded=True):
                for msg in update.get("messages", []):
                    if hasattr(msg, "content"):
                        st.markdown(msg.content)
                if update.get("graph_context"):
                    with st.expander("Raw graph data", expanded=False):
                        st.json(update["graph_context"])
                if update.get("memory_context"):
                    with st.expander("Memory hits", expanded=False):
                        st.json(update["memory_context"])
                if update.get("similar_cases"):
                    with st.expander("Similar past cases", expanded=True):
                        for case in update["similar_cases"]:
                            st.markdown(
                                f"`{case['transaction_id']}` — **{case['verdict']}** — "
                                f"{case['similarity']:.0%} similar"
                            )
                            with st.expander("Pattern detail", expanded=False):
                                st.text(case.get("pattern_text", ""))

else:
    with col_reasoning:
        st.subheader("Agent Reasoning")
        st.info("Enter a transaction ID and click **Investigate** to begin.")


# ---------------------------------------------------------------------------
# Right column — graph path visualization
# ---------------------------------------------------------------------------

with col_graph:
    st.subheader("Risk Path")
    result = st.session_state.last_result

    if result and result.get("graph_path"):
        path = result["graph_path"]
        vertex_icons = {
            "transaction": "💳",
            "account": "👤",
            "merchant": "🏪",
            "device": "📱",
        }

        for i, element in enumerate(path):
            if element.get("type") == "vertex":
                label = element.get("label", "?")
                vid = element.get("id", "?")
                props = element.get("props", {})
                is_flagged = props.get("is_flagged", False)
                flag_reason = props.get("flag_reason", "")

                icon = vertex_icons.get(label, "●")
                flag_marker = " 🚨" if is_flagged else ""

                card = f"**{icon} {label}**\n`{vid}`{flag_marker}"
                if is_flagged and flag_reason:
                    card += f"\n_{flag_reason}_"

                if is_flagged:
                    st.error(card)
                else:
                    st.info(card)

                # Arrow between vertices (skip for last element)
                if i < len(path) - 1:
                    next_el = path[i + 1] if i + 1 < len(path) else None
                    if next_el and next_el.get("type") == "edge":
                        st.markdown(
                            f"<div style='text-align:center; color:#888; font-size:18px'>↓ {next_el.get('label','')}</div>",
                            unsafe_allow_html=True,
                        )

            # Edges are rendered as arrows above, skip standalone rendering
    else:
        st.caption("Graph path will appear after investigation.")


# ---------------------------------------------------------------------------
# Bottom — verdict card
# ---------------------------------------------------------------------------

result = st.session_state.last_result

if result and result.get("verdict") and result["verdict"] != "ERROR":
    st.divider()

    verdict = result["verdict"]
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")
    risk_factors = result.get("risk_factors", [])

    colors = {"FRAUD": "#c0392b", "SUSPICIOUS": "#e67e22", "CLEAN": "#27ae60"}
    bg_colors = {"FRAUD": "#fdecea", "SUSPICIOUS": "#fef5e7", "CLEAN": "#eafaf1"}
    color = colors.get(verdict, "#7f8c8d")
    bg = bg_colors.get(verdict, "#f5f5f5")
    icons = {"FRAUD": "🚨", "SUSPICIOUS": "⚠️", "CLEAN": "✅"}
    icon = icons.get(verdict, "❓")

    risk_html = ""
    if risk_factors:
        items = "".join(f"<li>{rf}</li>" for rf in risk_factors)
        risk_html = f"<ul style='margin:8px 0 0 0; padding-left:20px'>{items}</ul>"

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border: 2px solid {color};
            border-radius: 10px;
            padding: 20px 24px;
            margin-top: 10px;
        ">
            <h2 style="color:{color}; margin:0 0 6px 0">{icon} {verdict}</h2>
            <p style="margin:0 0 10px 0">
                <strong>Confidence:</strong> {confidence:.0%} &nbsp;|&nbsp;
                <strong>Transaction:</strong> <code>{result.get('transaction_id','')}</code>
            </p>
            <p style="margin:0 0 8px 0">{explanation}</p>
            {risk_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

elif result and result.get("verdict") == "ERROR":
    st.error(f"Investigation failed. Check that Aerospike is running and seed data has been loaded.")
