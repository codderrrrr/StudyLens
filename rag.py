import os
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
# ⚠️ SECURITY NOTE: Never hardcode API keys in production!
# Use environment variables: export GOOGLE_API_KEY="your_key"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NOTES_FILE     = "output.txt"
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 150
EMBED_MODEL    = "BAAI/bge-base-en-v1.5"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MMR_K          = 8
MMR_FETCH_K    = 20
RERANK_TOP_N   = 4


# =========================
# 1. LOAD NOTES
# =========================
def load_notes(path: str) -> str:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] Notes file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit("[ERROR] Notes file is empty.")
    print(f"[INFO] Loaded {len(text):,} characters from '{path}'")
    return text


# =========================
# 2. CHUNKING
# =========================
def build_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        length_function=len,
        add_start_index=True,
    )
    docs = splitter.create_documents([text])
    print(f"[INFO] Created {len(docs)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return docs


# =========================
# 3. EMBEDDINGS
# =========================
def build_embeddings():
    print("[INFO] Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =========================
# 4. VECTOR STORE
# =========================
def build_vectorstore(docs, embeddings):
    print("[INFO] Building FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    return db


# =========================
# 5. MMR RETRIEVER
# =========================
def build_retriever(db):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": 0.6,
        },
    )


# =========================
# 6. RERANKER
# =========================
def build_reranker():
    print("[INFO] Loading reranker model...")
    return CrossEncoder(RERANK_MODEL)

def rerank_docs(reranker, query: str, docs, top_n: int = RERANK_TOP_N):
    if not docs:
        return docs
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


# =========================
# 7. LLM
# =========================
def build_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )


# =========================
# 8. PROMPTS
# =========================

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite the follow-up "
     "as a clear standalone search query. Keep it concise — one sentence max. "
     "Output ONLY the rewritten query, nothing else."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an academic assistant helping a student with their course notes.

CONTEXT FROM NOTES:
{context}

ANSWERING RULES:
1. If the answer is clearly present in the notes above: answer from the notes, \
   and end with a brief "📌 Source: notes" marker.
2. If the answer is only partially in the notes: use both the notes and your \
   knowledge, and flag which parts come from where.
3. If the answer is NOT in the notes at all: answer from your general academic \
   knowledge — but start your reply with: \
   "⚠️ Not in your notes — general answer:" so the student knows.
4. Never make up facts. If genuinely unsure, say so.
5. Be concise and exam-focused. Avoid padding."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_GENERATION_PROMPT = """You are a university professor creating exam and quiz questions.

NOTES CONTENT:
{context}

TASK: Generate 6 exam-style questions that test UNDERSTANDING, not memorization.
Apply the "teaching A, testing B" principle:
- If notes explain a Turing machine that adds → ask about one that multiplies
- If notes define a DFA for even binary numbers → ask them to design one for divisibility by 3
- Abstract the concept and apply it to a new but related scenario

FORMAT each question exactly like this:
Q[n]. [The question — can be design, proof-sketch, comparison, or application]
Type: [Conceptual / Design / Application / Tricky MCQ]
Difficulty: [Easy / Medium / Hard]
Hint: [Which topic/concept from notes this tests]
---

Mix: 2 Easy, 3 Medium, 1 Hard. Include at least one MCQ.
"""


# =========================
# 9. CHAIN BUILDERS
# =========================

def build_rag_chain(llm, retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONDENSE_PROMPT
    )
    answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_chain)
    return rag_chain


# =========================
# 10. ASK WITH RERANKING + HISTORY [FIXED]
# =========================

def ask_question(query: str, rag_chain, retriever, reranker, chat_history: list, llm):  # ← Added llm param
    print("\n🔍 Retrieving relevant chunks...")

    # Get the condensed query for reranking
    if chat_history:
        condense_prompt_filled = CONDENSE_PROMPT.format_messages(
            chat_history=chat_history, input=query
        )
        # ✅ FIX: Use llm directly, not rag_chain.first.llm
        condensed = llm.invoke(condense_prompt_filled)
        search_query = condensed.content.strip()
    else:
        search_query = query

    print(f"   Search query: {search_query}")

    # Retrieve via MMR
    raw_docs = retriever.invoke(search_query)
    print(f"   MMR returned {len(raw_docs)} docs")

    # Rerank
    reranked_docs = rerank_docs(reranker, search_query, raw_docs)
    print(f"   Reranker kept top {len(reranked_docs)}")

    # Build context from reranked docs
    context_text = "\n\n---\n\n".join(d.page_content for d in reranked_docs)

    # ✅ FIX: Use llm.invoke() directly instead of rag_chain.first.llm.invoke()
    filled = ANSWER_PROMPT.format_messages(
        context=context_text,
        chat_history=chat_history,
        input=query,
    )
    response = llm.invoke(filled)  # ← Fixed line
    answer   = response.content.strip()

    print("\n================ ANSWER ================\n")
    print(answer)
    print("\n============= SOURCES (top chunks) =====\n")
    for i, doc in enumerate(reranked_docs):
        offset = doc.metadata.get("start_index", "?")
        end_pos = int(offset) + len(doc.page_content) if isinstance(offset, int) else "?"
        print(f"  [{i+1}] chars {offset}–{end_pos}")
        print(f"      {doc.page_content[:200].strip()}...")
        print()

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    return answer, chat_history

# =========================
# 11. QA GENERATION
# =========================

def generate_qa(db, llm, reranker):
    print("\n📝 Sampling notes for question generation...")

    seed_queries = [
        "key definitions and formal proofs",
        "algorithms and construction procedures",
        "theorems and their conditions",
        "examples and edge cases",
    ]

    seen  = set()
    docs  = []
    temp_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.8},
    )

    for seed in seed_queries:
        for doc in temp_retriever.invoke(seed):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)

    docs = rerank_docs(reranker, "important exam topics and concepts", docs, top_n=8)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt  = QA_GENERATION_PROMPT.format(context=context)
    result  = llm.invoke(prompt)

    print("\n============= GENERATED QUESTIONS =============\n")
    print(result.content)
    return result.content


# =========================
# 12. MAIN LOOP [FIXED]
# =========================

def main():
    print("\n🔥 ADVANCED RAG SYSTEM — IMPROVED PIPELINE 🚀\n")

    text        = load_notes(NOTES_FILE)
    docs        = build_chunks(text)
    embeddings  = build_embeddings()
    db          = build_vectorstore(docs, embeddings)
    retriever   = build_retriever(db)
    reranker    = build_reranker()
    llm         = build_llm()
    rag_chain   = build_rag_chain(llm, retriever)

    print("\n✅ Pipeline ready.\n")

    chat_history = []

    while True:
        print("─" * 45)
        choice = input("1. Ask a question\n2. Generate exam questions\n3. Clear chat history\n4. Exit\nChoice: ").strip()

        if choice == "1":
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            # ✅ FIX: Pass llm to ask_question
            _, chat_history = ask_question(query, rag_chain, retriever, reranker, chat_history, llm)
            print(f"\n[History length: {len(chat_history)//2} turns]")

        elif choice == "2":
            generate_qa(db, llm, reranker)

        elif choice == "3":
            chat_history = []
            print("[INFO] Chat history cleared.")

        elif choice == "4":
            print("Bye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()