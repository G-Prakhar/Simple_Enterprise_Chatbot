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
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------------------------------------
# STEP 1: Load the embedding function.
#
# We use ChromaDB's own DefaultEmbeddingFunction here instead of loading
# sentence-transformers directly. Under the hood it's the same underlying
# model family (a small MiniLM model), but it runs on "onnxruntime"
# instead of full PyTorch.
#
# WHY THIS MATTERS: sentence-transformers pulls in the entire PyTorch
# stack as a dependency (torch, plus -- on Linux -- a pile of NVIDIA CUDA
# packages even when there's no GPU). That's several hundred MB just to
# load into memory, which is enough by itself to exceed the 512MB RAM
# limit on Render's free tier and get your app killed (exit code 137 =
# out of memory). onnxruntime has no such baggage, so this does the same
# job for a fraction of the memory footprint.
# -----------------------------------------------------------------------
embedding_function = embedding_functions.DefaultEmbeddingFunction()


# -----------------------------------------------------------------------
# STEP 2: Connect to (or create) a local ChromaDB database.
#
# PersistentClient means the database is saved to disk at "./chroma_db"
# so it survives between script runs — you don't need to re-ingest every
# time you restart the server.
#
# get_or_create_collection is like "use this table, create it if it
# doesn't exist yet". We name it "company_kb" (company knowledge base),
# and pass embedding_function so ChromaDB knows HOW to turn text into
# vectors automatically whenever we add or query documents -- we no
# longer compute embeddings ourselves.
# -----------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="company_kb",
    embedding_function=embedding_function,
)


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

        # (c) Build IDs and metadata for each chunk.
        # Unique ID format: "filename-0", "filename-1", ... so IDs never
        # collide between different files.
        chunk_ids = [f"{filename}-{i}" for i in range(len(chunks))]

        # Metadata lets us later show "this answer came from refund_policy.md"
        chunk_metadata = [{"source": filename} for _ in chunks]

        # (d) Store everything in ChromaDB in one call. We do NOT compute
        # embeddings ourselves here -- because the collection was created
        # with embedding_function above, ChromaDB automatically embeds
        # each document in `chunks` internally before storing it.
        collection.add(
            ids=chunk_ids,
            documents=chunks,
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