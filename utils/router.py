from typing import Literal
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["rag", "price", "both", "chit_chat"] = Field(
        ...,
        description=(
            "rag        – needs info from NVIDIA 2024 10‑K\n"
            "price      – needs current/historical stock price only\n"
            "both       – needs both tools\n"
            "chit_chat  – small‑talk / no external info"
        ),
    )

_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
_llm_structured = _llm.with_structured_output(RouteDecision)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a router. Decide which tools the assistant should use:"
            "\n- 'rag'  →  if the user needs details from the NVIDIA 2024 Annual Report."
            "\n- 'price'→  if they ask only for the stock price."
            "\n- 'both' →  if they need **both** report details and price."
            "\n- 'chit_chat' → any off‑topic small‑talk.",
        ),
        ("human", "{question}"),
    ]
)

ROUTER_CHAIN = _PROMPT | _llm_structured
