import sys, pathlib
from utils.config import THREAD_ID, DEFAULT_PDF
from utils.loader import build_retriever
import utils.tools
from utils.graph_builder import build_graph

def main(pdf_path: pathlib.Path | None = None):
    pdf_path = pdf_path or DEFAULT_PDF         
    if not pdf_path.exists():
        sys.exit(f"PDF not found at {pdf_path}")

    utils.tools.RETRIEVER = build_retriever(str(pdf_path))
    graph = build_graph()
    cfg   = {"configurable": {"thread_id": THREAD_ID}}

    print("\n🤖  NVIDIA LangGraph Agent (ENTER to quit)\n")
    while True:
        q = input("You: ").strip()
        if not q:
            break
        for ev in graph.stream(
        {"messages": [{"role": "user", "content": q}]},
        stream_mode="values",
        config=cfg,
    ):
            node_val = next(iter(ev.values()))

            if isinstance(node_val, dict):         
                msgs = node_val.get("messages", [])
            elif isinstance(node_val, list):        
                msgs = node_val
            else:                                   
                continue

            if not msgs:
                continue

            last = msgs[-1]

            if last.type == "ai" and not getattr(last, "tool_calls", None):
                print(f"Bot: {last.content}\n")

if __name__ == "__main__":
    arg_path = pathlib.Path(sys.argv[1]) if len(sys.argv) == 2 else None
    main(arg_path)
