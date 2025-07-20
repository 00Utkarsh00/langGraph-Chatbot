# langGraph Chatbot

A FastAPI-based AI agent powered by LangChain and LangGraph.  
It can answer questions from an NVIDIA PDF (RAG) **and** fetch live stock prices using custom tool functions.

---

## Features

- **RAG over PDF**: Perform semantic search and interact with an LLM to extract insights from your NVIDIA document.  
- **Text chat**: Communicate with the agent via REST API or a web-based UI.  
- **Audio chat**: Receive agent responses as TTS (Text-to-Speech) MP3 files.  
- **Stock prices**: Use the custom `get_price` function to fetch real-time stock quotes.  

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/langGraph-Chatbot.git
cd langGraph-Chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Update the `test.env` file with your configuration values.

### 4. Run the Server

Start the server using:

```bash
uvicorn app:app --reload
```

The server will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Testing

### Web UI (Voice + Text)

1. Open `index.html` in your browser.  
2. Type or speak your query and receive responses in text or audio format.

### Terminal Chat (No Server Required)

Run the following command to chat directly in your terminal:

```bash
python terminalTest.py
```

---

## Workflow Overview

Here’s a basic workflow design:

![Workflow Diagram](image/workflow.png)

Based on the user message, the router determines one of four options:

- **`rag`**: For extracting details from the NVIDIA 2024 Annual Report.  
- **`price`**: For fetching only the stock price.  
- **`both`**: For retrieving both report details and stock price.  
- **`chit_chat`**: For casual, off-topic conversations.

### Functionality Breakdown

1. **RAG Runner**: Adds context from the NVIDIA document to the user query before passing it to the LLM.  
2. **Stock Price Fetcher**: Provides real-time stock prices using custom functions.  used https://finazon.io/ for api

- For `rag` and `both` paths, RAG context is included in the query.  
- For `chit_chat`, no additional functions or context are used.  
- For `price`, only the stock price function is invoked without RAG context.





sample responses

Q - shared owned by gen hsun huang in 2024

AI answer - Jen-Hsun Huang, the CEO of NVIDIA, beneficially owned a total of 93,463,791 shares of NVIDIA's common stock, which represents approximately 3.79% of the company's outstanding shares. This ownership includes shares held in various trusts and entities associated with him and his family.


--------------------------------------------

Q - apple stock price today

AI - the latest stock price for Apple Inc. (AAPL) is USD 211.17.

--------------------------------------------

Q - what was Fees Billed by the Independent Registered Public Accounting Firm in 2024

A - In Fiscal 2024, the fees billed by the independent registered public accounting firm, PwC, were as follows:

- **Audit Fees:** $6,686,412
- **Audit Related Fees:** $804,258
- **Tax Fees:** $1,164,399
- **All Other Fees:** $352,000

**Total Fees:** $9,007,069

This represents an increase from the total fees of $8,308,800 billed in Fiscal 2023.

--------------------------------------------

Q: what were the Operating Expenses

AI: The operating expenses for NVIDIA for the fiscal year ended January 28, 2024, compared to the fiscal year ended January 29, 2023, are as follows:

### Operating Expenses
- **Research and Development Expenses:**
  - 2024: $8,675 million (14.2% of net revenue)
  - 2023: $7,339 million (27.2% of net revenue)
  - Change: $1,336 million (18% increase)

- **Sales, General and Administrative Expenses:**
  - 2024: $2,654 million (4.4% of net revenue)
  - 2023: $2,440 million (9.1% of net revenue)
  - Change: $214 million (9% increase)

- **Acquisition Termination Cost:**
  - 2024: $0 million
  - 2023: $1,353 million (5.0% of net revenue)
  - Change: $(1,353) million (100% decrease)

- **Total Operating Expenses:**
  - 2024: $11,329 million (18.6% of net revenue)
  - 2023: $11,132 million (41.3% of net revenue)
  - Change: $197 million (2% increase)

### Summary
The increase in operating expenses in fiscal year 2024 was primarily driven by higher research and development costs and sales, general and administrative expenses, reflecting employee growth and compensation increases. The significant decrease in acquisition termination costs in 2024 was due to the absence of costs related to the Arm acquisition, which were recorded in the previous fiscal year.
