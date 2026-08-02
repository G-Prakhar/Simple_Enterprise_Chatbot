"""
ingest.py
=========
This script is run ONCE (and again any time the company's documents change).

Its whole job: take the raw text files in the `knowledge/` folder and turn
them into a searchable vector database, so that later, when a user asks a
question, we can find the most relevant paragraphs to feed to the LLM.

Think of it as building the "brain's index" before the chatbot ever talks
to a real user.

Pipeline for each file in knowledge/:
    1. Read the raw text
    2. Split it into small overlapping chunks (so search results are precise)
    3. Convert each chunk into a vector of numbers (an "embedding")
    4. Store the chunk + its embedding + its source filename in ChromaDB
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------------------------------------
# STEP 1: Load the embedding model.
#
# "all-MiniLM-L6-v2" is a small, free, open-source model that converts
# text into a 384-number vector. Similar meanings produce similar vectors,
# which is what lets us do "search by meaning" instead of "search by
# exact keyword".
#
# It downloads once (~80MB) and then runs fully offline/locally — this
# step costs no API money, unlike the LLM call itself.
# -----------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------------------------------------
# STEP 2: Connect to (or create) a local ChromaDB database.
#
# PersistentClient means the database is saved to disk at "./chroma_db"
# so it survives between script runs — you don't need to re-ingest every
# time you restart the server.
#
# get_or_create_collection is like "use this table, create it if it
# doesn't exist yet". We name it "company_kb" (company knowledge base).
# -----------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_kb")


# -----------------------------------------------------------------------
# STEP 3: Set up the text splitter.
#
# LLMs and embedding models work best on small chunks of text, not whole
# documents. RecursiveCharacterTextSplitter cuts text into pieces of
# roughly `chunk_size` characters, trying to break at natural points
# (paragraphs, then sentences, then words) rather than mid-word.
#
# `chunk_overlap` repeats a little bit of text between consecutive chunks
# so we don't lose context that happens to fall right on a chunk boundary.
# -----------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # roughly 500 characters per chunk
    chunk_overlap=50,    # last 50 characters of one chunk repeat in the next
)


def ingest_folder(folder_path: str = "knowledge") -> None:
    """
    Reads every file in `folder_path`, chunks it, embeds it, and stores
    it in the ChromaDB collection.

    This function does four things per file:
      a) read the file's raw text
      b) split it into chunks
      c) turn each chunk into an embedding vector
      d) save {chunk text, embedding, source filename} into ChromaDB
    """

    files_processed = 0
    total_chunks_stored = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip anything that isn't a regular file (e.g. subfolders)
        if not os.path.isfile(file_path):
            continue

        # (a) Read the raw text of the document
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # (b) Split the document into overlapping chunks
        chunks = text_splitter.split_text(raw_text)

        if not chunks:
            continue  # empty file, nothing to do

        # (c) Convert all chunks for this file into embeddings in one batch
        # .encode() returns a numpy array; ChromaDB wants plain lists,
        # hence .tolist()
        chunk_embeddings = embedding_model.encode(chunks).tolist()

        # Build a unique ID for every chunk so ChromaDB can store/update it.
        # Format: "filename-0", "filename-1", ... so IDs never collide
        # between different files.
        chunk_ids = [f"{filename}-{i}" for i in range(len(chunks))]

        # Metadata lets us later show "this answer came from refund_policy.md"
        chunk_metadata = [{"source": filename} for _ in chunks]

        # (d) Store everything in ChromaDB in one call
        collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=chunk_embeddings,
            metadatas=chunk_metadata,
        )

        files_processed += 1
        total_chunks_stored += len(chunks)
        print(f"  ✓ {filename}: {len(chunks)} chunks stored")

    print(
        f"\nDone. Ingested {files_processed} files "
        f"({total_chunks_stored} chunks total) into '{collection.name}'."
    )


# This block only runs when you execute `python ingest.py` directly
# (not when this file is imported by main.py for its functions).
if __name__ == "__main__":
    print("Ingesting documents from ./knowledge ...")
    ingest_folder("knowledge")