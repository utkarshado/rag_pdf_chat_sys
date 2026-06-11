import os
import sys
import json
import shutil
import textwrap
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

#error tracking
import traceback

#pdf
import fitz 

#vecttor database
import chromadb
from chromadb.utils import embedding_functions

#llm
import google.generativeai as genai

print(sys.executable)
print(sys.version)



DB_PATH       = "./chroma_db"        # Where ChromaDB saves vectors to disk
HISTORY_FILE  = "./chat_history.json" # Where chat turns are saved
CHUNK_SIZE    = 1000                  # Characters per chunk (~1000 words)
CHUNK_OVERLAP = 200                  # Overlap between consecutive chunks 200
TOP_K         = 5                    # How many chunks to retrieve per question
MAX_HISTORY   = 6                    # Max Q&A turns to keep in memory
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "models/gemini-2.5-flash"
COLLECTION    = "pdf_chunks"         # ChromaDB collection name

load_dotenv()

#loading and chunkiingn pdf

def load_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        full_text += f"\n[Page {page_num + 1}]\n{page.get_text()}"
    doc.close()
    return full_text


def chunk_text(text: str, source_path: str) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    base = Path(source_path).stem   # "my_doc" from "my_doc.pdf"
    name = Path(source_path).name   # "my_doc.pdf"

    while start < len(text):
        content = text[start : start + CHUNK_SIZE]
        if content.strip():  # skip blank chunks
            chunks.append({
                "id":     f"{base}__chunk_{idx}",
                "text":   content,
                "source": name,
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


#vector db stuff

def get_collection():

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["GEMINI_API_KEY"],
        model_name=EMBED_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
    )


def get_indexed_files(collection) -> set[str]:
    
    if collection.count() == 0:
        return set()
    result = collection.get(include=["metadatas"])
    return {m["source"] for m in result["metadatas"] if "source" in m}


def index_pdf(collection, pdf_path: str) -> bool:

    filename = Path(pdf_path).name

    if filename in get_indexed_files(collection):
        print(f"  😁 Already indexed: {filename}  (skipping)")
        return False

    print(f"  📖 Reading {filename} ...")
    raw_text = load_pdf(pdf_path)
    char_count = len(raw_text)
    print(f"  ✂️  Chunking {char_count:,} characters ...")
    chunks = chunk_text(raw_text, pdf_path)
    print(f"  🧨 Embedding {len(chunks)} chunks via Google API ...")

    collection.add(
        ids       = [c["id"]     for c in chunks],
        documents = [c["text"]   for c in chunks],
        metadatas = [{"source": c["source"]} for c in chunks],
    )
    print(f"  ✅ Done — '{filename}' indexed ({len(chunks)} chunks)")
    return True


#retrieval of relevant chunks

def retrieve_chunks(collection, question: str) -> list[dict]:
    
    results = collection.query(
        query_texts = [question],
        n_results   = TOP_K,
        include     = ["documents", "metadatas"],
    )
    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]


#chat history

class ChatHistory:
   

    def __init__(self):
        self.turns: list[dict] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, question: str, answer: str) -> None:
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > MAX_HISTORY:
            self.turns = self.turns[-MAX_HISTORY:]
        self._save()

    def clear(self) -> None:
        self.turns = []
        Path(HISTORY_FILE).unlink(missing_ok=True)
        print("  ✅ Chat history cleared.")

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def summary(self) -> str:
        
        return f"{len(self.turns)} turn(s) in memory"

    def format_for_prompt(self) -> str:
        
        if not self.turns:
            return "(No previous conversation)"
        lines = []
        for i, turn in enumerate(self.turns, 1):
            truncated = (turn["answer"][:400] + "…") if len(turn["answer"]) > 400 else turn["answer"]
            lines.append(f"[Turn {i}]")
            lines.append(f"  Q: {turn['question']}")
            lines.append(f"  A: {truncated}")
        return "\n".join(lines)

    def display(self) -> None:
        if self.is_empty():
            print("\n  (No conversation history yet.)")
            return
        print(f"\n{'─'*60}")
        print(f"  🛺 Chat History  ({self.summary()})")
        print(f"{'─'*60}")
        for i, turn in enumerate(self.turns, 1):
            print(f"\n  [{i}] You: {turn['question']}")
            # Wrap and indent the answer
            wrapped = textwrap.fill(turn["answer"], width=56)
            indented = textwrap.indent(wrapped, "       ")
            print(f"      Assistant: {indented.lstrip()}")
        print(f"{'─'*60}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _save(self) -> None:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"saved_at": datetime.now().isoformat(), "turns": self.turns},
                f, indent=2, ensure_ascii=False,
            )

    def _load(self) -> None:
        if not Path(HISTORY_FILE).exists():
            return
        try:
            data = json.loads(Path(HISTORY_FILE).read_text(encoding="utf-8"))
            loaded = data.get("turns", [])[-MAX_HISTORY:]
            self.turns = loaded
            if loaded:
                print(f"  🏍️ Resumed {len(loaded)} turn(s) from previous session.")
        except Exception:
            self.turns = []


#rewruting the query if required obviously

def rewrite_query_if_needed(question: str, history: ChatHistory, llm) -> str:
    
    if history.is_empty():
        return question  # First question — nothing to resolve

    prompt = f"""You are a query rewriter for a document search system.

Given a conversation history and a new question, rewrite the question
to be completely self-contained — so it can be understood with NO prior context.

Rules:
- Resolve pronouns (he/she/it/they/this/that) using the history.
- Expand vague references ("the method", "the author", "the result") to their full names.
- If the question is already self-contained, return it UNCHANGED.
- Return ONLY the rewritten question. No explanation, no quotes, no preamble.

CONVERSATION HISTORY:
{history.format_for_prompt()}

NEW QUESTION: {question}

REWRITTEN QUESTION:"""

    response = llm.generate_content(prompt)
    rewritten = response.text.strip().strip('"').strip("'")
    return rewritten if rewritten else question


#answering the query

def generate_answer(
    original_question: str,
    chunks: list[dict],
    history: ChatHistory,
    llm,
) -> tuple[str, list[str]]:
    
    # Build context block with source labels
    context_parts = [
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)
    sources = sorted({c["source"] for c in chunks})

    prompt = f"""You are a precise, helpful assistant that answers questions about documents.

STRICT RULES:
- Use ONLY the document context below to answer. Do not use outside knowledge.
- If the answer isn't in the context, say: "I couldn't find that in the indexed documents."
- When possible, mention which source file(s) the information comes from.
- Keep answers clear and concise.

══ DOCUMENT CONTEXT ══════════════════════════════════════
{context}

══ CONVERSATION HISTORY ══════════════════════════════════
{history.format_for_prompt()}

══ CURRENT QUESTION ══════════════════════════════════════
{original_question}

══ YOUR ANSWER ═══════════════════════════════════════════"""

    response = llm.generate_content(prompt)
    return response.text.strip(), sources


#cli - command lie interface

HELP_TEXT = """
┌─────────────────────────────────────────────────────────┐
│  COMMANDS                                               │
├─────────────────────────────────────────────────────────┤
│  add <path/to/file.pdf>  →  Index a PDF file            │
│  list                    →  Show all indexed PDFs       │
│  history                 →  Show this session's chat    │
│  clear history           →  Delete chat history         │
│  clear db                →  Delete all indexed PDFs     │
│  clear all               →  Delete everything           │
│  help                    →  Show this menu              │
│  quit  /  exit  /  q     →  Exit                        │
├─────────────────────────────────────────────────────────┤
│  <anything else>         →  Ask a question!             │
└─────────────────────────────────────────────────────────┘
"""


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                   PDF CHAT  — RAG Assistant              ║
║     Multi-PDF • Persistent DB • Chat History             ║
╚══════════════════════════════════════════════════════════╝""")


def handle_add(user_input: str, collection) -> None:
    
    parts = user_input.split(None, 1)
    if len(parts) < 2:
        print("  Usage: add <path/to/file.pdf>")
        return
    pdf_path = parts[1].strip().strip('"').strip("'")
    if not Path(pdf_path).exists():
        print(f"  ❌ File not found: {pdf_path}")
        return
    if not pdf_path.lower().endswith(".pdf"):
        print(f"  ⚠️  '{pdf_path}' doesn't look like a PDF (no .pdf extension).")
        confirm = input("  Continue anyway? (y/n): ").strip().lower()
        if confirm != "y":
            return
    index_pdf(collection, pdf_path)


def handle_list(collection) -> None:
    
    indexed = get_indexed_files(collection)
    if not indexed:
        print("\n  ⚠️  No PDFs indexed yet.  Use: add <path.pdf>")
        return
    print(f"\n  📚 Indexed PDFs ({len(indexed)}):")
    for name in sorted(indexed):
        print(f"     • {name}")


def handle_clear(user_input: str, collection, history: ChatHistory) -> bool:
    
    cmd = user_input.lower().strip()

    if cmd == "clear history":
        history.clear()
        return False

    if cmd in ("clear db", "clear database"):
        confirm = input("  ⚠️  Delete ALL indexed PDFs? (yes/no): ").strip().lower()
        if confirm == "yes":
            shutil.rmtree(DB_PATH, ignore_errors=True)
            print("  ✅ Vector database cleared. Restart to re-index.")
            return True
        print("  Cancelled.")
        return False

    if cmd == "clear all":
        confirm = input("  ⚠️  Delete ALL data (DB + history)? (yes/no): ").strip().lower()
        if confirm == "yes":
            history.clear()
            shutil.rmtree(DB_PATH, ignore_errors=True)
            print("  ✅ Everything cleared. Restart to re-index.")
            return True
        print("  Cancelled.")
        return False

    # Plain "clear" — ask what to clear
    print("  What do you want to clear?")
    print("    clear history  →  chat log only")
    print("    clear db       →  indexed PDFs only")
    print("    clear all      →  everything")
    return False


def run_rag_pipeline(
    user_input: str,
    collection,
    history: ChatHistory,
    llm,
) -> None:
    

    # Guard: need at least one PDF
    if collection.count() == 0:
        print("  ⚠️  No PDFs indexed yet.  Use: add <path.pdf>")
        return

    # ── Step A: rewrite vague follow-ups ──────────────────────────────────────
    if not history.is_empty():
        print("  🔄 Checking for follow-up references ...")
        standalone = rewrite_query_if_needed(user_input, history, llm)
        if standalone.lower() != user_input.lower():
            print(f"  📝 Searching as: \"{standalone}\"")
    else:
        standalone = user_input

    # ── Step B: retrieve relevant chunks ──────────────────────────────────────
    print(f"  🔍 Searching {collection.count()} chunks ...")
    chunks = retrieve_chunks(collection, standalone)
    sources_found = sorted({c["source"] for c in chunks})

    # ── Step C: generate answer ───────────────────────────────────────────────
    print(f"  🤖 Generating answer (from {len(chunks)} chunks) ...")
    answer, sources = generate_answer(user_input, chunks, history, llm)

    # ── Step D: store turn ────────────────────────────────────────────────────
    history.add(user_input, answer)

    # ── Display ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    for line in answer.split("\n"):
        if line.strip():
            print(textwrap.fill(line, width=60))
        else:
            print()
    print()
    print(f"  📎 Sources: {', '.join(sources)}")
    print(f"  📜 History: {history.summary()}")
    print("─" * 60)


#mainloop

def main():
    print_banner()

    # ── Check API key ──────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\n  ❌ GEMINI_API_KEY not set.")
        print("    Windows:    set GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    # ── Init Gemini ────────────────────────────────────────────────────────────
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)

    # ── Init ChromaDB ──────────────────────────────────────────────────────────
    print(f"\n  🤓  Opening vector store at ./{DB_PATH} ...")
    try:
        collection = get_collection()
    except Exception as e:
        print(f"  ❌ Could not open ChromaDB: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Init history ───────────────────────────────────────────────────────────
    history = ChatHistory()

    # ── Index any PDFs passed on the command line ──────────────────────────────
    cli_pdfs = [a for a in sys.argv[1:] if not a.startswith("-")]
    if cli_pdfs:
        print()
        for pdf_arg in cli_pdfs:
            p = Path(pdf_arg)
            if p.exists() and p.suffix.lower() == ".pdf":
                index_pdf(collection, str(p))
            else:
                print(f"  ⚠️  Skipped: {pdf_arg} (not found or not a .pdf)")

    # ── Show state ─────────────────────────────────────────────────────────────
    indexed = get_indexed_files(collection)
    print()
    if indexed:
        print(f"  😶‍🌫️ Ready — {len(indexed)} PDF(s) indexed:")
        for name in sorted(indexed):
            print(f"     • {name}")
    else:
        print("  ⚠️  No PDFs indexed yet.")
        print("  Add one with: add <path/to/file.pdf>")

    print(HELP_TEXT)
    print("  Type your question, or a command above.")
    print("─" * 60)

    # ── Main loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("\n  🥰 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        # ── Exit ───────────────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            break

        # ── Help ───────────────────────────────────────────────────────────────
        elif cmd == "help":
            print(HELP_TEXT)

        # ── List ───────────────────────────────────────────────────────────────
        elif cmd == "list":
            handle_list(collection)

        # ── History ────────────────────────────────────────────────────────────
        elif cmd == "history":
            history.display()

        # ── Add PDF ────────────────────────────────────────────────────────────
        elif cmd.startswith("add "):
            handle_add(user_input, collection)

        # ── Clear ──────────────────────────────────────────────────────────────
        elif cmd.startswith("clear"):
            should_exit = handle_clear(user_input, collection, history)
            if should_exit:
                break

        # ── RAG pipeline ───────────────────────────────────────────────────────
        else:
            run_rag_pipeline(user_input, collection, history, llm)


if __name__ == "__main__":
    main()
