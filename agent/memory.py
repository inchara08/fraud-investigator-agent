"""
Aerospike connection factory.

Provides a single aerospike.Client instance shared by both:
  - AerospikeSaver  (LangGraph checkpoint / resume state)
  - AerospikeStore  (long-term cross-session memory)

In Streamlit, wrap get_cached_resources() with @st.cache_resource so the
client is created once per process, not on every script re-run.
"""

import os
import aerospike
from langgraph.checkpoint.aerospike import AerospikeSaver
from langgraph.store.aerospike import AerospikeStore

AEROSPIKE_HOST = os.getenv("AEROSPIKE_HOST", "127.0.0.1")
AEROSPIKE_PORT = int(os.getenv("AEROSPIKE_PORT", "3000"))
AEROSPIKE_NAMESPACE = os.getenv("AEROSPIKE_NAMESPACE", "test")


def get_aerospike_client() -> aerospike.Client:
    """Connect and return a thread-safe Aerospike client."""
    config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
    return aerospike.client(config).connect()


def get_checkpointer(client: aerospike.Client) -> AerospikeSaver:
    """
    LangGraph checkpoint saver — persists graph execution state so
    investigations can resume after interruption.

    Uses AerospikeSaver's own default set name (do NOT pass set="fraud_memory"
    or it will collide with AerospikeStore).
    """
    return AerospikeSaver(client=client, namespace=AEROSPIKE_NAMESPACE)


def get_store(client: aerospike.Client) -> AerospikeStore:
    """
    Long-term memory store — remembers previously flagged accounts across
    agent sessions.  Uses set="fraud_memory" to avoid collisions with the
    checkpointer's set.
    """
    return AerospikeStore(
        client=client,
        namespace=AEROSPIKE_NAMESPACE,
        set="fraud_memory",
    )
