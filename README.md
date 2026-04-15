# 🩺 MedBot AI — RAG-Based Medical Assistant

MedBot is an intelligent, conversational AI assistant designed to answer health-related questions. It uses **Retrieval-Augmented Generation (RAG)** to search through a custom medical knowledge base and provide accurate, context-aware answers.

The application is fully containerised for local development and optimised for blazing-fast serverless deployment on **Vercel**.

---

## 🏗 Tech Stack

### Frontend (UI)
* **HTML/CSS/JS**: Pure, lightweight standard web technologies.
* **Tailwind CSS**: For modern, responsive styling and layout.
* **Glassmorphism & Animations**: Custom CSS for premium UI effects (sliding messages, typing indicators).
* **Vercel Edge Network**: The frontend is served statically from the `public/` directory for maximum performance.

### Backend (Serverless API)
* **Python 3.10+**: Core language.
* **Flask**: Lightweight web framework to handle the API routing.
* **Vercel Serverless Functions**: The backend runs as a serverless Python function (`api/index.py`), executing only when a request is made.

### AI & Machine Learning Pipeline
* **OpenRouter API**: Routes LLM requests to `openai/gpt-4o-mini` for fast, cost-effective, and intelligent response generation.
* **OpenAI Embeddings API**: Uses `text-embedding-3-small` (configured to 384 dimensions) to convert text into semantic vectors.
* **Pinecone**: Cloud-native Vector Database used to store and quickly search thousands of medical text chunks.
* **HuggingFace & LangChain**: (Used during local data ingestion) to process, chunk, and embed the initial medical PDF.

---

## ⚙️ How It Works (The Workflow)

The system works in two distinct phases: **1. Data Ingestion** (done once locally) and **2. Query/Chat** (happens on every user message).

### Phase 1: Data Ingestion (`ingest.py`)
*This phase builds the AI's "brain" and only needs to be run when adding new knowledge.*
1. **Load**: A large medical PDF (`gale_encyclopedia_medicine.pdf`) is loaded using LangChain's document loader.
2. **Chunk**: The massive text is split into smaller, overlapping chunks (e.g., 500 characters each) so the AI can digest them easily.
3. **Embed**: Each text chunk is passed through a sentence-transformer model to turn the text into a **Vector Embedding** (a mathematical representation of its meaning).
4. **Store**: These vectors (along with the original text) are securely uploaded and stored in the **Pinecone Vector Database**.

### Phase 2: Live Chat Workflow (`api/index.py` & `public/index.html`)
*This is what happens when a user types a question on the deployed website.*
1. **User Asks**: The user types a question (e.g., *"What is diabetes?"*) on the frontend.
2. **Fetch Request**: The browser sends an asynchronous `POST` request to the serverless backend (`/api`).
3. **Embed Query**: The Python backend calls the OpenRouter API to convert the user's question into a 384-dimensional vector.
4. **Vector Search (Retrieval)**: The backend queries Pinecone with this vector to find the **top 3 most semantically similar** medical text chunks from the database.
5. **Prompt Construction (Augmentation)**: The backend combines the user's question with the retrieved medical chunks into a single, strict "System Prompt."
6. **LLM Generation**: This combined prompt is sent to `gpt-4o-mini` via OpenRouter. The AI reads the provided medical context and generates an accurate, empathetic answer.
7. **Response**: The generated answer is sent back to the frontend and elegantly animated into a chat bubble!

---

## 📂 Project Structure (Vercel Optimised)

```text
medicalbot/
│
├── public/                 # 🌐 FRONTEND (Served statically by Vercel)
│   └── index.html          # Chat UI (contains all HTML, CSS, and JS)
│
├── api/                    # ⚙️ BACKEND (Vercel Serverless Functions)
│   └── index.py            # Main Flask app handling the /api POST endpoint
│
├── ingest.py               # 🧠 LOCAL SCRIPT: Processes PDF & populates Pinecone
├── requirements.txt        # 📦 Backend Python dependencies
├── .env                    # 🔑 Secret API keys (DO NOT COMMIT)
└── README.md               # 📖 Project documentation
```

---

## 🚀 Deployment (Vercel)

The project leverages Vercel's zero-configuration topology:
1. **Static Hosting**: Anything in the `public/` folder is automatically distributed globally on Vercel's Edge CDN.
2. **Serverless Functions**: The `api/index.py` file is automatically detected as a Python backend and deployed as an AWS Lambda function.
3. **Environment Variables**: `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` are securely injected into the serverless environment.

This architecture means the site scales infinitely, costs nothing when idle, and requires **no DevOps maintenance**.
