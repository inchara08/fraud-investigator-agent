"""
Streamlit UI for fraud-investigator-agent.

Layout (top → bottom):
  Header           — title + subtitle
  Try an example   — 4 scenario quick-pick buttons
  Input section    — transaction ID text input + Investigate button
  Investigation    — st.status() live stream (or static summary on re-render)
  Report           — structured investigation report [2 cols: report | graph path]

Key design decisions:
  - Synchronous graph.stream() — avoids asyncio/Streamlit event loop conflict
  - @st.cache_resource — Aerospike client created once per process, not per rerun
  - st.session_state — guards against stream re-run on Streamlit script reruns
  - selected_tx in session state — scenario buttons pre-fill the text input
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from agent.csv_loader import (
    _flag_value,
    build_graph_from_df,
    get_graph_context,
    get_template_csv,
    validate_columns,
)
from agent.graph import build_fraud_graph
from agent.memory import get_aerospike_client, get_checkpointer, get_store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("tx_clean_001",  "🟢 Low-risk Purchase",  "Standard retail payment — no flagged entities in graph"),
    ("tx_fraud_001",  "🔴 Direct Fraud",        "$9,800 wire to shell company — sender flagged for laundering"),
    ("tx_subtle_001", "🟠 Indirect Link",       "2-hop connection to structuring account via shared merchant"),
    ("tx_fp_001",     "🟡 Resolved Flag",       "Shared device with flagged account — flag context is mitigating"),
]

NODE_STEPS = {
    "ingest_transaction":    "Validating transaction",
    "query_graph":           "Mapping entity connections",
    "check_memory":          "Checking investigation history",
    "search_similar_cases":  "Searching for similar fraud patterns",
    "reason_about_patterns": "Analyzing behavioral signals",
    "explain_verdict":       "Building fraud hypothesis",
}

RISK_LEVEL = {"FRAUD": "🔴 HIGH", "SUSPICIOUS": "🟠 MEDIUM", "CLEAN": "🟢 LOW"}

VERTEX_ICONS = {"transaction": "💳", "account": "👤", "merchant": "🏪", "device": "📱"}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_cached_resources():
    try:
        client = get_aerospike_client()
        checkpointer = get_checkpointer(client)
        store = get_store(client)
        aerospike_available = True
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore
        checkpointer = MemorySaver()
        store = InMemoryStore()
        aerospike_available = False
    graph = build_fraud_graph(checkpointer, store)
    return graph, store, checkpointer, aerospike_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_summary(node_name: str, update: dict) -> str:
    """Return a one-line investigator-style summary for a completed node."""
    if node_name == "ingest_transaction":
        if update.get("verdict") == "ERROR":
            return "Transaction not found in graph — investigation halted."
        for msg in update.get("messages", []):
            content = getattr(msg, "content", "")
            if content:
                return content
        return "Transaction validated."

    if node_name == "query_graph":
        path = update.get("graph_path") or []
        flagged = sum(
            1 for el in path
            if el.get("type") == "vertex" and el.get("props", {}).get("is_flagged")
        )
        if flagged:
            return f"Entity graph mapped — **{flagged} flagged account(s)** discovered across traversal paths."
        return "Entity graph mapped — no directly flagged accounts in traversal paths."

    if node_name == "check_memory":
        mem = update.get("memory_context") or {}
        hits = [k for k, v in mem.items() if v]
        if hits:
            return f"Memory recall: **{len(hits)} account(s)** carry active flags from prior investigations."
        return "Memory recall: no prior flags found for accounts in this graph."

    if node_name == "search_similar_cases":
        cases = update.get("similar_cases") or []
        if cases:
            top = cases[0]
            return (
                f"Found **{len(cases)} similar case(s)** — "
                f"top match: `{top['transaction_id']}` ({top['verdict']}, {top['similarity']:.0%} similarity)."
            )
        return "No similar historical cases found (cold start)."

    if node_name == "reason_about_patterns":
        msgs = update.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "") or ""
            if content:
                snippet = content[:120].replace("\n", " ")
                return f"Reasoning complete — _{snippet}…_"
        return "Behavioral analysis complete."

    if node_name == "explain_verdict":
        verdict = update.get("verdict", "")
        confidence = update.get("confidence", 0.0)
        if verdict:
            return f"Verdict reached: **{verdict}** with **{confidence:.0%}** confidence."
        return "Verdict generation complete."

    return f"{NODE_STEPS.get(node_name, node_name)} complete."


def render_graph_path(graph_path: list):
    """Render the risk traversal path as a vertical chain of cards."""
    st.markdown("#### Risk Path")
    for i, element in enumerate(graph_path):
        if element.get("type") != "vertex":
            continue
        label = element.get("label", "?")
        vid = element.get("id", "?")
        props = element.get("props", {})
        is_flagged = props.get("is_flagged", False)
        flag_reason = props.get("flag_reason", "")
        icon = VERTEX_ICONS.get(label, "●")

        lines = [f"**{icon} {label}**", f"`{vid}`"]
        if is_flagged and flag_reason:
            lines.append(f"🚨 _{flag_reason}_")

        card_text = "  \n".join(lines)
        if is_flagged:
            st.error(card_text)
        else:
            st.info(card_text)

        # Arrow to next vertex
        if i + 1 < len(graph_path):
            next_el = graph_path[i + 1]
            edge_label = ""
            if next_el.get("type") == "edge":
                edge_label = next_el.get("label", "")
                # peek at the vertex after the edge
                if i + 2 < len(graph_path) and graph_path[i + 2].get("type") == "vertex":
                    st.markdown(
                        f"<div style='text-align:center;color:#888'>↓ {edge_label}</div>",
                        unsafe_allow_html=True,
                    )


def render_report(result: dict):
    """Render the structured investigation report below the process section."""
    verdict = result.get("verdict", "")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")
    risk_factors = result.get("risk_factors") or []
    graph_path = result.get("graph_path") or []
    tx_id = result.get("transaction_id", "")

    st.markdown("---")
    st.markdown("## Investigation Report")

    # Top metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Level", RISK_LEVEL.get(verdict, verdict))
    m2.metric("Confidence", f"{confidence:.0%}")
    m3.metric("Transaction ID", tx_id)

    st.markdown("---")

    report_col, graph_col = st.columns([2, 1])

    with report_col:
        # Key signals
        if risk_factors:
            st.markdown("#### Key Signals")
            for signal in risk_factors:
                st.markdown(f"- {signal}")
            st.markdown("")

        # Linked entities from graph path
        flagged_entities = [
            el for el in graph_path
            if el.get("type") == "vertex" and el.get("props", {}).get("is_flagged")
        ]
        if flagged_entities:
            st.markdown("#### Linked Flagged Entities")
            for entity in flagged_entities:
                icon = VERTEX_ICONS.get(entity.get("label", ""), "●")
                eid = entity.get("id", "")
                reason = entity.get("props", {}).get("flag_reason", "")
                label = entity.get("label", "")
                st.markdown(f"- {icon} **{eid}** ({label}) — _{reason}_")
            st.markdown("")

        # Agent reasoning
        if explanation:
            st.markdown("#### Agent Reasoning")
            st.markdown(explanation)
            st.markdown("")

        # Final decision
        st.markdown("#### Final Decision")
        if verdict == "FRAUD":
            st.error(f"🚨 **FRAUD** — Recommend immediate account freeze and SAR filing.")
        elif verdict == "SUSPICIOUS":
            st.warning(f"⚠️ **SUSPICIOUS** — Flag for manual review by a senior analyst.")
        elif verdict == "CLEAN":
            st.success(f"✅ **CLEAN** — No action required. Transaction appears legitimate.")

    with graph_col:
        if graph_path:
            render_graph_path(graph_path)
        else:
            st.caption("No graph path data available.")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Investigator Agent",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "stream_log" not in st.session_state:
    st.session_state.stream_log = []
if "selected_tx" not in st.session_state:
    st.session_state.selected_tx = ""
if "csv_graph_context" not in st.session_state:
    st.session_state.csv_graph_context = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 Fraud Investigator Agent")
st.markdown(
    "An AI-powered investigation system that traverses transaction graphs, "
    "recalls cross-session memory, and reasons over behavioral signals to "
    "produce structured fraud verdicts — simulating how a senior fraud analyst "
    "investigates suspicious activity."
)
_, _, _, _aerospike_ok = get_cached_resources()
if not _aerospike_ok:
    st.warning(
        "**Demo mode** — Aerospike is not reachable. "
        "Running with in-memory stores: investigations work normally but "
        "cross-session memory and checkpoint resume are disabled.",
        icon="⚠️",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Try an example / Upload CSV tabs
# ---------------------------------------------------------------------------

tab_demo, tab_csv = st.tabs(["Demo Scenarios", "Upload CSV"])

with tab_demo:
    st.markdown("**Try an example scenario:**")
    s_cols = st.columns(len(SCENARIOS))
    for col, (tx_val, label, description) in zip(s_cols, SCENARIOS):
        with col:
            if st.button(label, help=description, use_container_width=True):
                st.session_state.selected_tx = tx_val
                st.session_state.csv_graph_context = None  # switch back to Gremlin mode
                st.session_state.last_result = None
                st.session_state.stream_log = []

with tab_csv:
    st.markdown(
        "Upload your own transaction data. "
        "The agent builds a relationship graph from the CSV and investigates any transaction you select."
    )

    dl_col, _ = st.columns([1, 3])
    dl_col.download_button(
        "Download template CSV",
        data=get_template_csv(),
        file_name="transactions_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            df = None

        if df is not None:
            missing = validate_columns(df)
            if missing:
                st.error(
                    f"Missing required column(s): `{'`, `'.join(missing)}`. "
                    "Download the template above to see the expected format."
                )
            else:
                st.dataframe(df, use_container_width=True, height=200)
                n_senders = df["sender_id"].nunique()
                n_devices = df["device_id"].nunique() if "device_id" in df.columns else 0
                n_flagged = (
                    df["sender_is_flagged"].apply(_flag_value).sum()
                    if "sender_is_flagged" in df.columns else 0
                )
                st.caption(
                    f"{len(df)} transaction(s) · {n_senders} unique sender(s) · "
                    f"{n_devices} device(s) · {int(n_flagged)} flagged account(s)"
                )

                tx_options = df["transaction_id"].astype(str).tolist()
                selected_csv_tx = st.selectbox(
                    "Select a transaction to investigate",
                    options=tx_options,
                )

                if selected_csv_tx:
                    G = build_graph_from_df(df)
                    ctx = get_graph_context(G, selected_csv_tx)
                    if ctx is None:
                        st.error(f"Transaction `{selected_csv_tx}` could not be found after building graph.")
                    else:
                        st.session_state.selected_tx = selected_csv_tx
                        st.session_state.csv_graph_context = ctx
                        st.session_state.last_result = None
                        st.session_state.stream_log = []
                        flagged_count = len(ctx.get("flagged_paths", []))
                        neighbors_count = len(ctx.get("direct_context", {}).get("neighbors", []))
                        st.success(
                            f"Graph ready — `{selected_csv_tx}` has {neighbors_count} connected "
                            f"entity/entities and {flagged_count} flagged path(s). "
                            "Click **Investigate** below."
                        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.markdown("#### Input")
input_col, btn_col = st.columns([4, 1])

with input_col:
    tx_id = st.text_input(
        "Transaction ID",
        value=st.session_state.selected_tx,
        placeholder="e.g. tx_fraud_001",
        label_visibility="collapsed",
    )

with btn_col:
    investigate_btn = st.button(
        "🔍 Investigate",
        type="primary",
        disabled=not tx_id.strip(),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Run investigation
# ---------------------------------------------------------------------------

if investigate_btn and tx_id.strip():
    st.session_state.last_result = None
    st.session_state.stream_log = []
    st.session_state.selected_tx = tx_id.strip()

    graph, store, checkpointer, _ = get_cached_resources()
    config = {"configurable": {"thread_id": f"fraud-{tx_id.strip()}"}}
    initial_state = {
        "transaction_id": tx_id.strip(),
        "messages": [],
        # CSV mode: pre-built graph_context bypasses Gremlin in nodes 1 & 2
        "graph_context": st.session_state.csv_graph_context,
        "memory_context": None,
        "similar_cases": None,
        "verdict": None,
        "confidence": None,
        "explanation": None,
        "graph_path": None,
        "risk_factors": None,
        "investigation_complete": False,
    }

    st.markdown("---")
    st.markdown("#### Investigation Process")

    try:
        with st.status("🔍 Running investigation…", expanded=True) as inv_status:
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, update in event.items():
                    st.session_state.stream_log.append((node_name, update))
                    step_label = NODE_STEPS.get(node_name, node_name)
                    summary = _node_summary(node_name, update)
                    st.markdown(f"**{step_label}**  \n{summary}")

            inv_status.update(label="Investigation complete ✅", state="complete", expanded=False)

        final_state = graph.get_state(config).values
        st.session_state.last_result = final_state

    except Exception as e:
        st.error(f"Investigation error: {e}")

# ---------------------------------------------------------------------------
# Re-render: show static process summary (no re-stream)
# ---------------------------------------------------------------------------

elif st.session_state.stream_log and not investigate_btn:
    st.markdown("---")
    st.markdown("#### Investigation Process")
    with st.expander("Investigation complete ✅", expanded=False):
        for node_name, update in st.session_state.stream_log:
            step_label = NODE_STEPS.get(node_name, node_name)
            summary = _node_summary(node_name, update)
            st.markdown(f"**{step_label}**  \n{summary}")

# ---------------------------------------------------------------------------
# Investigation report
# ---------------------------------------------------------------------------

result = st.session_state.last_result

if result:
    if result.get("verdict") == "ERROR":
        st.markdown("---")
        st.error(
            "Transaction not found. Make sure Aerospike is running and seed data has been loaded:\n\n"
            "```bash\npython data/seed_data.py\n```"
        )
    elif result.get("verdict"):
        render_report(result)

elif not st.session_state.stream_log:
    st.markdown("---")
    st.info("Select a scenario above or enter a Transaction ID and click **Investigate** to begin.")
