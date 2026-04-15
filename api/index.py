"""
==============================================================================
BACKEND SERVER — app.py
==============================================================================
Flask application that serves the chat UI and handles RAG-based Q&A.

RAG Workflow (Query Phase):
  User Query  →  Embed (via API)  →  Pinecone Search  →  Context + Prompt  →  LLM  →  Answer
==============================================================================
"""

import os
import requests as http_requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone as PineconeClient

# ---------------------------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY", "").strip()
PINECONE_INDEX     = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot").strip()

# ---------------------------------------------------------------------------
# 2. INITIALISE PINECONE (lightweight — no heavy ML deps)
# ---------------------------------------------------------------------------

pc    = None
index = None

def _get_pinecone_index():
    """Lazily connect to Pinecone."""
    global pc, index
    if index is not None:
        return index
    pc    = PineconeClient(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    print("✅ Connected to Pinecone")
    return index

# ---------------------------------------------------------------------------
# 3. EMBEDDING VIA OPENROUTER API (lightweight — no torch/transformers!)
# ---------------------------------------------------------------------------
# Instead of running sentence-transformers locally (7 GB of PyTorch),
# we call OpenRouter's embeddings endpoint which is OpenAI-compatible.

def embed_text(text: str) -> list:
    """Embed a single text string using OpenRouter's embedding API."""
    resp = http_requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/text-embedding-3-small",
            "input": text,
            "dimensions": 384,  # Must match Pinecone index dimension (384)
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

# ---------------------------------------------------------------------------
# 4. LLM CALL VIA OPENROUTER API (lightweight — direct HTTP)
# ---------------------------------------------------------------------------

def call_llm(context: str, question: str) -> str:
    """Send the RAG prompt to OpenRouter and return the answer text."""
    system_prompt = """You are a knowledgeable, friendly, and helpful medical assistant.

You have access to medical context retrieved from a knowledge base.
Use it as your PRIMARY source of information when it is relevant.

If the context covers the patient's question, base your answer on it.
If the context is missing, incomplete, or not relevant to the question, you
should STILL provide a helpful, accurate, and general medical answer using
your own medical knowledge. In that case, add a brief note like:
"(Based on general medical knowledge.)"

Always:
• Be empathetic and supportive.
• Give practical advice (home remedies, when to see a doctor, etc.).
• Do NOT diagnose — suggest consulting a healthcare professional for
  specific diagnoses or treatment plans.
• Be concise but thorough."""

    user_message = f"""--- MEDICAL CONTEXT ---
{context}
--- END CONTEXT ---

Patient Question: {question}

Helpful Answer:"""

    resp = http_requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# 5. FLASK APPLICATION
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="../public", static_url_path="/")

@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

@app.route("/api", methods=["POST"])
def chat():
    """
    RAG endpoint — receives a user question and returns an AI-generated
    answer grounded in the medical knowledge base.

    Request JSON:  { "question": "What is diabetes?" }
    Response JSON: { "answer": "Diabetes is ..." }
    """
    data = request.get_json()
    user_question = data.get("question", "").strip()

    if not user_question:
        return jsonify({"answer": "Please enter a valid question."}), 400

    try:
        # --- Step A: Embed the user's question via API ---
        query_vector = embed_text(user_question)

        # --- Step B: Retrieve top-K relevant chunks from Pinecone ---
        idx = _get_pinecone_index()
        results = idx.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
        )

        # --- Step C: Build the context string ---
        context_chunks = []
        for match in results.get("matches", []):
            text = match["metadata"].get("text", "")
            if text:
                context_chunks.append(text)

        context = "\n\n".join(context_chunks) if context_chunks else "(No relevant context found in knowledge base.)"

        # --- Step D: Generate an answer with the LLM ---
        answer = call_llm(context, user_question)
        return jsonify({"answer": answer})

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"❌ Error in /api: {err}")
        return jsonify({"answer": f"Backend Error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# 6. RUN THE SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
