# Chat With PDFs — A Simple RAG Chatbot

This project lets you chat with your PDF documents using Retrieval-Augmented Generation (RAG).

Instead of trying to fit an entire PDF into a model's context window, the application extracts text from PDFs, splits it into chunks, stores vector embeddings in ChromaDB, retrieves the most relevant chunks for a question, and uses Gemini to generate answers grounded in the document.

The goal of this project was to understand the core concepts behind modern AI applications such as embeddings, vector databases, retrieval, context injection, and conversational memory by building everything from scratch.

Never thought someone will actually read my README :)
---

## Features

* Chat with one or multiple PDFs
* Persistent vector database using ChromaDB
* Automatic PDF chunking and embedding
* Semantic search using vector similarity
* Conversational chat history
* Query rewriting for follow-up questions
* Source attribution showing which PDF was used
* Session persistence across restarts
* Command-line interface

---

## How It Works

### Indexing Phase

When a PDF is added:

PDF → Text Extraction → Chunking → Embeddings → ChromaDB

1. Text is extracted using PyMuPDF
2. The document is split into overlapping chunks
3. Each chunk is converted into an embedding vector
4. Embeddings and metadata are stored in ChromaDB

This only happens once per PDF.

---

### Question Answering Phase

When a question is asked:

Question → Query Rewrite → Retrieval → Context Injection → Gemini → Answer

1. Follow-up questions are rewritten into standalone questions
2. The question is embedded
3. ChromaDB retrieves the most relevant chunks
4. Retrieved chunks are injected into the prompt
5. Gemini generates an answer using only the retrieved context

---

## Tech Stack

### LLM

* Gemini 2.5 Flash

### Embeddings

* Gemini Embeddings

### Vector Database

* ChromaDB

### PDF Processing

* PyMuPDF

### Language

* Python

---

## Project Structure

```text
chat_with_pdf/
│
├── chat_pdf.py
├── chroma_db/
├── chat_history.json
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd chat_with_pdf
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

### Linux / macOS

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

## Running The Application

```bash
python chat_pdf.py
```

---

## Commands

Add a PDF:

```text
add path/to/file.pdf
```

Show indexed PDFs:

```text
list
```

Show conversation history:

```text
history
```

Clear chat history:

```text
clear history
```

Clear vector database:

```text
clear db
```

Clear everything:

```text
clear all
```

Exit:

```text
quit
```

---

## Example

```text
🥰 You: add attention_is_all_you_need.pdf

📖 Reading attention_is_all_you_need.pdf ...
✂️ Chunking text ...
🔢 Embedding chunks ...
✅ Done

🥰 You: What is multi-head attention?

🤖 Multi-head attention allows the model to attend to
different representation subspaces simultaneously...
```

---

## What I Learned

Building this project helped me understand:

* RAG architecture
* Vector embeddings
* Semantic search
* ChromaDB
* Query rewriting
* Conversational memory
* Prompt construction
* Context injection
* LLM API integration
* End-to-end AI application development

---

## Future Improvements

* Streamlit web interface
* PDF image understanding
* Hybrid search (keyword + vector search)
* Citation-level source references
* Local embedding models
* Multi-modal RAG
* Document upload through UI
* FastAPI backend

---

## Disclaimer

This project is intended for learning and experimentation. Responses are generated from retrieved document chunks and may not always be perfect, it is dumb. Always verify important information directly from the source documents.
