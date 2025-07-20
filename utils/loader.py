import pathlib
from typing import Final

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from utils.config import INDEX_DIR



def build_retriever(pdf_path: str, *, k: int = 5):
    emb = OpenAIEmbeddings(model="text-embedding-3-large")
    idx_path: Final = pathlib.Path(INDEX_DIR)

    if idx_path.exists():
        db = FAISS.load_local(
            str(idx_path), emb, allow_dangerous_deserialization=True
        )
    else:
        print("indexing annual report …")
        docs = PyPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)

        for _ in tqdm(range(len(chunks)), desc="Embedding chunks"):
            pass 

        db = FAISS.from_documents(chunks, emb)
        db.save_local(str(idx_path))

    return db.as_retriever(search_kwargs={"k": k})
