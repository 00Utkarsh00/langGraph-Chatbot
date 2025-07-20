# utils/graph_builder.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

import utils.nodes as N

def build_graph():
    g = StateGraph(MessagesState)

    # 1) route → decide branch
    g.add_node("router", N.route)

    # 2) for rag or both: fetch context first
    g.add_node("retrieve", N.retrieve_context)

    # 3) then call LLM (with stock-price tool still bound, if you need it)
    g.add_node("llm", N.llm_decision)

    # 4) tool executor if any stock-price calls
    g.add_node("env", N.tool_exec)

    # 5) chit-chat branch
    g.add_node("chit_chat", N.chit_chat)

    # ─── edges ─────────────────────────────────────────────────────────
    g.add_edge(START, "router")

    # router → either rag/both → retrieve → llm
    g.add_conditional_edges(
      "router",
      lambda s: s["route"],
      {
        "rag":   "retrieve",
        "both":  "retrieve",
        "price": "llm",
        "chit_chat": "chit_chat",
      }
    )

    # after retrieve → llm
    g.add_edge("retrieve", "llm")

    # agentic loop: llm → env (if tool_calls) → back to llm
    g.add_conditional_edges("llm", N.need_tools, {"Action": "env", END: END})
    g.add_edge("env", "llm")

    # chit-chat ends immediately
    g.add_edge("chit_chat", END)

    return g.compile(checkpointer=MemorySaver())
