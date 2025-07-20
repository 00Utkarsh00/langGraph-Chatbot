from email import utils
from typing import Literal
from langgraph.graph import END, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from utils.tools import TOOLS, TOOLS_BY_NAME
from utils.router import ROUTER_CHAIN, RouteDecision
from langchain_core.messages import AIMessage
import re, datetime
LLM   = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)
from utils.tools import search_Nvidia_Doc

_TOOL_SYSTEM = SystemMessage(
    content=(
        "You have two tools:\n"
        "• search_Nvidia_Doc(query)\n"
        "• get_stock_price(ticker, date?)\n"
        "Call them whenever needed; do not fabricate data."
    )
)

def route(state: MessagesState):
    question = state["messages"][-1].content
    decision: RouteDecision = ROUTER_CHAIN.invoke({"question": question})
    return {"route": decision.route}


def retrieve_context(state: MessagesState):

    user_q = state["messages"][-1].content
    docs   = search_Nvidia_Doc(user_q)
    if not docs:
        return {"messages": [AIMessage(content="No relevant documents found.")]}
    return {
      "messages": [
        SystemMessage(content="Here are relevant excerpts from the nvidia 10-K:\n\n" + docs)
      ]
    }

def _parse_price_args(question: str):
    """Very simple extraction of ticker & date from the user text."""
    ticker = "NVDA"
    date   = None

    m = re.search(r"\b([A-Z]{2,5})\b", question.upper())
    if m:
        ticker = m.group(1)

    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    if m:
        date = m.group(1)

    if "TODAY" in question.upper():
        date = None  

    return ticker, date

def price_only(state: MessagesState):
    q = state["messages"][-1].content
    ticker, date = _parse_price_args(q)
    result = TOOLS_BY_NAME["get_stock_price"].invoke(
        {"ticker": ticker, "date": date}
    )
    return {"messages": [AIMessage(content=result)]}

def llm_decision(state):
    prompt = [_TOOL_SYSTEM] + state["messages"]

    ai_msg = LLM_WITH_TOOLS.invoke(prompt)
    print("TOOL_CALLS:", ai_msg.tool_calls)
    return {"messages": [ai_msg]}

def tool_exec(state: MessagesState):
    last = state["messages"][-1]
    tool_msgs = []

    for call in getattr(last, "tool_calls", []):
        res = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        tool_msgs.append(ToolMessage(content=res, tool_call_id=call["id"]))

    return {"messages": tool_msgs}

def chit_chat(state: MessagesState):
    history = state["messages"]
    system = SystemMessage("You are a friendly assistant. Be concise.")
    prompt = [system] + history
    return {"messages": [LLM.invoke(prompt)]}

def need_tools(state: MessagesState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls and getattr(last, "additional_kwargs", None):
        ak = last.additional_kwargs
        if "tool_calls" in ak:
            tool_calls = ak["tool_calls"]
        elif "function_call" in ak:
            fc = ak["function_call"]
            tool_calls = [{"id": "fc-0", "name": fc.get("name"), "args": fc.get("arguments")}]
    return "Action" if tool_calls else END
