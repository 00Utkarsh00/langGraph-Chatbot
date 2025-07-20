import os
import sys
import pathlib
from typing import List, Tuple
from utils.config import OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def ingest_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200):

    loader = PyPDFLoader(path)
    docs = loader.load() 
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs) 


def build_retriever(chunks, index_path: str = "pdf_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    idx_path = pathlib.Path(index_path)
    if idx_path.exists():
        db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_path)
    return db.as_retriever(search_kwargs={"k": 5})


def make_chain(retriever):
    """Conversational RAG chain (1 retrieval per turn)."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=False,  
    )


def repl(chain):
    history: List[Tuple[str, str]] = []
    while True:
        q = input("You: ").strip()
        if not q:
            break
        result = chain.invoke({"question": q, "chat_history": history})
        a = result["answer"]
        print(f"Bot: {a}\n")
        history.append((q, a))


def main(pdf_path: str):
    if not pathlib.Path(pdf_path).exists():
        sys.exit(f"❌ File not found: {pdf_path}")
    chunks = ingest_pdf(pdf_path)

    retriever = build_retriever(chunks)

    chain = make_chain(retriever)

    repl(chain)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip())
        sys.exit(1)
    main(sys.argv[1])