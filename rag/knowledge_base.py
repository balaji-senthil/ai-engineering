from dotenv import load_dotenv

load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

folder_path = "./docs"


def load_doc(file_path: Path):
    """Loads a single ODF document and returns LangChain Document objects."""
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        print(f"✅ Loaded: {file_path.name}")
        return docs
    except Exception as e:
        print(f"❌ Failed to load {file_path.name}: {e}")
        return []


def load_all_docs(folder_path: str, max_workers: int = 8):
    """Loads all pdf files in parallel."""
    folder = Path(folder_path)
    files = list(folder.glob("*.pdf*"))

    all_docs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {executor.submit(load_doc, f): f for f in files}
        for future in as_completed(futures):
            docs = future.result()
            all_docs.extend(docs)

    print(f"\n📄 Total documents loaded: {len(all_docs)}")
    return all_docs


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vector_store = PGVector(
    embeddings=embeddings,
    collection_name="Invoices",
    connection=f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PWD']}@localhost:5432/aieng",
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.5, "fetch_k": 20, "score_threshold": "0.8"},
)


if __name__ == "main":
    all_docs = load_all_docs(folder_path)
    vector_store.add_documents(all_docs)
