# Chat With PDFs — A RAG-Powered PDF Chatbot

A project I built to learn how modern AI applications actually work under the hood.

Instead of sending an entire PDF to an LLM and hoping for the best, this application uses Retrieval-Augmented Generation (RAG) to find the most relevant information from a document and provide grounded answers.

The project started as a command-line application and was later extended into a web application using FastAPI and a custom frontend, then deployed publicly on Render.

If you're reading this README, first of all, thank you. I genuinely built this project to understand the concepts behind AI engineering rather than just calling an API and getting a response.

---

## Live Demo

Deployed on Render:

**Live URL:** `https://your-render-url.onrender.com`

*(Replace with your actual Render URL)*

---

## Features

### Document Processing

* Upload and chat with PDF documents
* Automatic text extraction using PyMuPDF
* Intelligent chunking with overlap
* Vector embeddings using Gemini Embeddings
* Persistent storage with ChromaDB

### Retrieval-Augmented Generation (RAG)

* Semantic search over document chunks
* Context-aware retrieval
* Query rewriting for follow-up questions
* Source-aware answers
* Multi-turn conversations

### Web Application

* PDF upload through a browser interface
* Interactive chat UI
* Session-based conversations
* FastAPI backend
* Deployed and accessible online via Render

### Command Line Interface

* Add PDFs from local paths
* List indexed documents
* View chat history
* Clear history and database
* Ask questions directly from the terminal

---

## How It Works

### Step 1: Indexing

When a PDF is uploaded:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
```

1. Text is extracted using PyMuPDF
2. The document is split into overlapping chunks
3. Each chunk is converted into an embedding vector
4. Chunks and metadata are stored in ChromaDB

This process only happens once per document.

---

### Step 2: Question Answering

When a user asks a question:

```text
Question
 ↓
Query Rewriting
 ↓
Vector Search
 ↓
Relevant Chunks
 ↓
Prompt Construction
 ↓
Gemini
 ↓
Answer
```

1. Follow-up questions are rewritten into standalone questions when needed
2. Relevant chunks are retrieved using semantic similarity search
3. Retrieved context is injected into the prompt
4. Gemini generates an answer using only the retrieved document content

---

## Tech Stack

### LLM

* Gemini 2.5 Flash

### Embeddings

* Gemini Embedding Model

### Vector Database

* ChromaDB

### Backend

* FastAPI

### Frontend

* HTML
* CSS
* JavaScript

### PDF Processing

* PyMuPDF

### Deployment

* Render

### Language

* Python

---

## Project Structure

```text
rag_pdf_chat_sys/
│
├── app.py                 # FastAPI backend
├── chat_pdf.py            # Core RAG logic
├── requirements.txt
├── render.yaml
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── uploads/
├── chroma_db/
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd rag_pdf_chat_sys
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Running Locally

Start the web application:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## Deployment

This project is deployed using Render.

### Backend + Frontend Deployment

The FastAPI backend serves both:

* API endpoints
* Frontend files

Deployment configuration:

```yaml
services:
  - type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Environment variables are managed through Render's dashboard.

This project is deployed on Render and can be accessed through a public URL.

The application serves both the FastAPI backend and the frontend from a single deployment, making it easy to upload PDFs and chat with them directly from the browser.

One thing I learned during deployment is that building the RAG pipeline is only half the work — getting everything running reliably online is its own challenge.

---

## Example Usage

Upload a PDF and ask questions such as:

```text
What is this document about?

Summarize the key points.

Who is the author?

What methodology was used?

Explain section 3 in simple terms.
```

The system retrieves relevant chunks from the document and generates answers grounded in those sources.

---

## What I Learned

Building this project helped me understand:

- Retrieval-Augmented Generation (RAG)
- Embeddings and vector search
- ChromaDB
- Semantic retrieval
- Prompt engineering
- Query rewriting
- Conversational memory
- FastAPI
- Frontend-backend integration
- API deployment
- Building complete AI applications end-to-end

More importantly, it helped me move beyond simple chatbot tutorials and understand how real-world AI systems are structured.

This project started as:

"Let me quickly make a PDF chatbot."

Several debugging sessions later, it became a full FastAPI + RAG application deployed on Render.

---

## Future Improvements

* Hybrid search (keyword + vector search)
* Better citation support
* Multi-PDF retrieval
* Streaming responses
* User authentication
* Persistent cloud vector database
* Multi-modal document understanding
* Docker deployment
* Advanced session management

---

## Disclaimer

This project was built primarily for learning and experimentation.

Responses are generated from retrieved document chunks and may occasionally be incomplete or inaccurate. For important information, always verify answers against the original document.

And if the model gives a strange answer, there's a decent chance I'm already debugging it.
