"""utils/tools.py – LangChain tools the agent can call."""
import datetime, os, time, requests
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from utils.config import FINAZON_KEY

RETRIEVER = None

_FINAZON_URL = "https://api.finazon.io/latest/finazon/us_stocks_essential/price"


def _fetch_price(ticker: str, at_ts: Optional[int] = None) -> float:

    if not FINAZON_KEY:
        raise RuntimeError("FINAZON_API_KEY missing from environment")

    headers = {"Authorization": f"apikey {FINAZON_KEY}"}
    # params  = {"ticker": ticker.upper()} # SADLY THE api that I am using does not give price for all the stocks for free
    params  = {"ticker": "AAPL"} # only Apple tesla google stock is available for free
    if at_ts is not None:
        params["at"] = at_ts

    r = requests.get(_FINAZON_URL, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        if "p" in data:
            return float(data["p"])

    raise RuntimeError(f"Unexpected Finazon response: {data}")


# ── Tool 1: rag -------------------------------------------------------
@tool("search_Nvidia_Doc")
def search_Nvidia_Doc(query: str) -> str:
    """Search the NVIDIA 2024 10‑K for `query`. Returns top‑5 passages."""
    matches = RETRIEVER.invoke(query)         
    return "\n\n".join(
        f"(page {m.metadata.get('page','?')}) {m.page_content}" for m in matches
    )


# # ── Tool 2: stock price (live / historical) ----------------------------------
@tool("get_stock_price")
def get_stock_price(ticker: str = "NVDA", date: str | None = None) -> str:
    """
    Return the USD closing price for `ticker`.
    • If `date` is None → latest quote (Finazon's 'last price').
    • If `date` = 'YYYY-MM-DD' → last price **at or before** that date
      (uses midnight UTC of the given day).
    """
    try:
        at_ts = None
        if date:
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            at_ts = int(time.mktime(dt.timetuple()))
        price = _fetch_price(ticker, at_ts)
        when  = date or datetime.date.today().isoformat()
        return f"{ticker.upper()} price on {when}: USD {price:,.2f}"
    except Exception as exc:
        return f"Price lookup failed ({exc})."


# # Registry for the agent
TOOLS         = [search_Nvidia_Doc, get_stock_price]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


