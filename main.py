import os
os.environ["USER_AGENT"] = "EnterpriseRAG/1.0"

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from unstructured.partition.pdf import partition_pdf
import google.generativeai as genai
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
load_dotenv()

print("All imports done!")
# ----------------------------------------------------------------------

PDF_PATH = "Company.pdf"

print("PDF path set!") 

# ----------------------------------------------------------------------
def load_document(file_path_or_url):
    if file_path_or_url.startswith("http"):
        loader = WebBaseLoader(file_path_or_url)
    elif file_path_or_url.endswith(".pdf"):
        loader = PyPDFLoader(file_path_or_url)
    elif file_path_or_url.endswith(".docx"):
        loader = Docx2txtLoader(file_path_or_url)
    elif file_path_or_url.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path_or_url, mode="elements")
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

print("Document loader ready!")

elements = partition_pdf(PDF_PATH, strategy="fast")

sections = []
current_title = "General"
current_text = ""

for el in elements:
    if el.category == "Title":
        if current_text.strip():
            sections.append({"title": current_title, "text": current_text.strip()})
        current_title = el.text
        current_text = ""
    else:
        current_text += " " + el.text

if current_text.strip():
    sections.append({"title": current_title, "text": current_text.strip()})

print(f"Total sections created: {len(sections)}")

# ----------------------------------------------------------------------


structured_docs = [
    Document(page_content=s["text"], metadata={"section_title": s["title"]})
    for s in sections
]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
final_chunks = splitter.split_documents(structured_docs)

print(f"Total final chunks: {len(final_chunks)}")

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(final_chunks, embedder)

print("Vector store created!")


# ----------------------------------------------------------------------

bm25_retriever = BM25Retriever.from_documents(final_chunks)
bm25_retriever.k = 3

semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]
)

print("Hybrid retriever ready!")

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Reranker ready!")

def rerank_chunks(query, chunks, top_n=3):
    pairs = [[query, chunk.page_content] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in scored[:top_n]]
    return top_chunks

print("Reranking function ready!")
# ----------------------------------------------------------------------
BLOCKED_KEYWORDS = ["hack", "bypass", "ignore previous instructions", "system prompt", "jailbreak"]

def check_input_guardrail(user_query):
    query_lower = user_query.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_lower:
            return False, "This query contains restricted content and cannot be processed."
    return True, None

print("Guardrail ready!")


# ----------------------------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel('gemini-flash-latest')
print("Gemini ready!")

def rewrite_query(user_query):
    prompt = f"""Rewrite the following question into a clear search query for finding information in a company policy document. 
Keep it CONCISE — do not add extra words or formal synonyms. If the question is already clear and specific, return it EXACTLY as is.
Only output the rewritten query, nothing else.

Original question: {user_query}

Rewritten query:"""
    response = model_gemini.generate_content(prompt)
    return response.text.strip()

chat_history = []

def add_to_history(question, answer):
    chat_history.append({"question": question, "answer": answer})

def contextualize_query(current_question, history):
    if not history:
        return current_question
    
    history_text = ""
    for turn in history:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
    
    prompt = f"""Given the conversation history and a new question, determine if the new question is a follow-up related to the history, or a completely new/unrelated topic.

- If it is a follow-up (uses words like "it", "that", "its", or clearly continues the previous topic), rewrite it into a complete standalone question using the history for context.
- If it is a new, unrelated question, just return the new question EXACTLY AS IS, ignoring the history completely.

Only output the final question, nothing else.

Conversation History:
{history_text}

New Question: {current_question}

Final Question:"""
    
    response = model_gemini.generate_content(prompt)
    return response.text.strip()

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    prompt = f"""Answer the question based only on the context below. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    response = model_gemini.generate_content(prompt)
    answer = response.text.strip()
    sources = [chunk.metadata.get('section_title', 'Unknown') for chunk in retrieved_chunks]
    return answer, sources

def ask_question(user_query):
    is_safe, block_message = check_input_guardrail(user_query)
    if not is_safe:
        return block_message, []
    
    standalone_query = contextualize_query(user_query, chat_history)
    better_query = rewrite_query(standalone_query)
    retrieved = hybrid_retriever.invoke(better_query)
    top_chunks = rerank_chunks(better_query, retrieved, top_n=3)
    answer, sources = generate_answer(better_query, top_chunks)
    add_to_history(user_query, answer)
    return answer, sources
print("Master pipeline ready!")


# ----------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel

# ----------------------------------------------------------------------
app = FastAPI()

API_KEY = os.getenv("AUTH_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return key

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QuestionRequest, key: str = Depends(verify_api_key)):
    answer, sources = ask_question(request.question)
    return {"answer": answer, "sources": sources}

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static_index.html", "r", encoding="utf-8") as f:
        return f.read()

# ----------------------------------------------------------------------
# ---------------- Evaluation ----------------

test_questions = [
    {
        "question": "Who is the CEO?",
        "ground_truth": "Charu Raheja, PhD is the Chair and CEO of TriageLogic."
    },
    {
        "question": "What is the HIPAA compliance policy?",
        "ground_truth": "HIPAA is a set of standards to protect patient health information privacy and security, requiring employee training and confidentiality."
    },
    {
        "question": "What is the company's data security policy?",
        "ground_truth": "Access to data is limited to authorized staff or approved vendors, with reasonable efforts made to safeguard information."
    }
]

def run_evaluation():
    results = []
    for t in test_questions:
        system_answer, sources = ask_question(t["question"])
        
        eval_prompt = f"""You are evaluating an AI system's answer against the correct answer.

Question: {t['question']}
Correct Answer: {t['ground_truth']}
System's Answer: {system_answer}

Rate the system's answer on a scale of 1-10 for accuracy (does it contain the correct information?).
Only output a single number, nothing else."""
        
        score_response = model_gemini.generate_content(eval_prompt)
        score = score_response.text.strip()
        
        results.append({
            "question": t["question"],
            "system_answer": system_answer,
            "ground_truth": t["ground_truth"],
            "score": score
        })
    
    return results

if __name__ == "__main__":
    eval_results = run_evaluation()
    for r in eval_results:
        print(f"\nQ: {r['question']}")
        print(f"System Answer: {r['system_answer'][:150]}...")
        print(f"Score: {r['score']}/10")


