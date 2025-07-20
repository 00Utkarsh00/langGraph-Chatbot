# utils/graph_builder.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

import utils.nodes as N

def build_graph():
    g = StateGraph(MessagesState)

    g.add_node("router", N.route)

    g.add_node("retrieve", N.retrieve_context)

    g.add_node("llm", N.llm_decision)

    g.add_node("env", N.tool_exec)

    g.add_node("chit_chat", N.chit_chat)

    g.add_edge(START, "router")

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

    g.add_edge("retrieve", "llm")

    g.add_conditional_edges("llm", N.need_tools, {"Action": "env", END: END})
    g.add_edge("env", "llm")

    g.add_edge("chit_chat", END)

    return g.compile(checkpointer=MemorySaver())
