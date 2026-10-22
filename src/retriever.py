import os
import re
import glob
import pandas as pd
import json
import time

from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs

# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if os.path.exists(DB_DIR):
        db = Chroma(
            collection_name="youtube_transcripts",
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        count = db._collection.count()
        print(f"Existing chunks: {count}")

        if count > 0:
            print("Reusing existing Chroma database.")
            return db

    docs = load_transcripts()

    if not docs:
        raise ValueError("No documents loaded.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    ).split_documents(docs)

    print("Total chunks:", len(chunks))
    print("Largest chunk:", max(len(c.page_content) for c in chunks))

    db = Chroma(
        collection_name="youtube_transcripts",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    batch_size = 5

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        try:
            db.add_documents(batch)
            print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

        except Exception as e:
            print(f"Failed at chunk index {i}")
            print("Error:", repr(e))
            raise

    print("Total stored:", db._collection.count())

    return db

def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("machine learning roadmap")
    # print("results ",results)
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")
        print("results ",results)