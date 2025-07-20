# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR       = os.getenv("PDF_INDEX_DIR", "./vector_index")
THREAD_ID       = os.getenv("THREAD_ID", "thread-1")
FINAZON_KEY = os.getenv("FINAZON_API_KEY")

DEFAULT_PDF = Path("docs/NVIDIA-2024-Annual-Report.pdf")

if not OPENAI_API_KEY:
    raise EnvironmentError("Add OPENAI_API_KEY to your .env file.")

if not FINAZON_KEY:
    raise EnvironmentError("Add FINAZON_API_KEY to your .env file.")
