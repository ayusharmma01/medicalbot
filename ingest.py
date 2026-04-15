"""
==============================================================================
DATA INGESTION PIPELINE — ingest.py
==============================================================================
This script processes a medical PDF document and stores its content as vector
embeddings in Pinecone for later retrieval by the chatbot.

RAG Workflow (Ingestion Phase):
  PDF  →  Text Extraction  →  Chunking  →  Embedding  →  Pinecone Upsert
==============================================================================
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
# Reads PINECONE_API_KEY and PINECONE_INDEX_NAME from the .env file.
load_dotenv()

PINECONE_API_KEY  = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX    = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")

# ---------------------------------------------------------------------------
# 2. EXTRACT TEXT FROM PDF
# ---------------------------------------------------------------------------
# PyPDFLoader reads the PDF page-by-page and returns a list of Document
# objects, each containing the page text and metadata (page number, source).

def load_pdf(file_path: str):
    """Load a PDF file and return a list of LangChain Document objects."""
    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"   ✅ Extracted {len(documents)} pages")
    return documents

# ---------------------------------------------------------------------------
# 3. CHUNK THE DOCUMENTS
# ---------------------------------------------------------------------------
# Large pages are split into smaller, overlapping chunks so that:
#   • Each chunk fits within embedding model token limits.
#   • Overlap preserves context across chunk boundaries.
#
# chunk_size=500   — roughly one short paragraph
# chunk_overlap=50 — keeps a small window of shared context

def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into smaller text chunks for embedding."""
    print(f"✂️  Chunking documents (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")
    return chunks

# ---------------------------------------------------------------------------
# 4. CREATE EMBEDDINGS
# ---------------------------------------------------------------------------
# We use a HuggingFace Sentence Transformer model that runs locally.
# Model: all-MiniLM-L6-v2  (384-dimensional vectors, fast inference)
# This is the SAME model used at query time in app.py to ensure consistency.

def get_embedding_model():
    """Return a HuggingFace embedding model instance."""
    print("🤖 Loading embedding model: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("   ✅ Embedding model loaded")
    return embeddings

# ---------------------------------------------------------------------------
# 5. UPSERT EMBEDDINGS INTO PINECONE
# ---------------------------------------------------------------------------
# Each chunk is embedded and stored in Pinecone along with its source text
# as metadata. At query time the chatbot searches this index for relevant
# medical context.

def upsert_to_pinecone(chunks, embeddings):
    """Embed all chunks and upsert them into a Pinecone index."""

    # --- Initialise Pinecone client ---
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # --- Create the index if it does not already exist ---
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing_indexes:
        print(f"📦 Creating Pinecone index: {PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,          # Must match embedding model output dim
            metric="cosine",        # Cosine similarity for semantic search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    else:
        print(f"📦 Pinecone index '{PINECONE_INDEX}' already exists")

    index = pc.Index(PINECONE_INDEX)

    # --- Batch-embed and upsert ---
    BATCH_SIZE = 100
    print(f"⬆️  Upserting {len(chunks)} vectors in batches of {BATCH_SIZE}")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        # Generate embeddings for this batch
        texts = [chunk.page_content for chunk in batch]
        vectors = embeddings.embed_documents(texts)

        # Build upsert payload: (id, vector, metadata)
        upsert_data = []
        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            upsert_data.append({
                "id": f"chunk-{i + j}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0),
                },
            })

        index.upsert(vectors=upsert_data)
        print(f"   ✅ Upserted batch {i // BATCH_SIZE + 1}")

    print("🎉 Ingestion complete!")

# ---------------------------------------------------------------------------
# MAIN — Run the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH = "gale_encyclopedia_medicine.pdf"

    # Step 1: Load
    documents = load_pdf(PDF_PATH)

    # Step 2: Chunk
    chunks = chunk_documents(documents)

    # Step 3: Embedding model
    embedding_model = get_embedding_model()

    # Step 4: Upsert into Pinecone
    upsert_to_pinecone(chunks, embedding_model)
