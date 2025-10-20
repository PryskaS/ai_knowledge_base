import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

# --- RAG Components ---
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Load Environment & Configure Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Load heavy models ONCE at startup ---
# This is a critical performance optimization.
logger.info("Loading RAG models and connecting to vector database...")

try:
    # Path to our persistent vector database
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
    COLLECTION_NAME = "ai_knowledge"

    # Embedding model to convert queries into vectors
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Connection to our existing ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # LLM for generating the final answer
    llm_client = OpenAI()
    
    logger.info("✅ Models and database loaded successfully!")

except Exception as e:
    logger.error(f"❌ Failed to load models or connect to DB: {e}", exc_info=True)
    # Set components to None to handle failure gracefully
    embedding_model = collection = llm_client = None

# --- 2. API Contract (Pydantic Models) ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question about the knowledge base.")

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[str]

# --- 3. FastAPI Application ---
app = FastAPI(
    title="AI Knowledge Base (RAG Service)",
    description="An API to ask questions about a specific knowledge base using RAG.",
    version="1.0.0"
)

@app.get("/health", tags=["Monitoring"])
def health_check():
    status = "ok" if all([embedding_model, collection, llm_client]) else "error_loading_models"
    return {"status": status, "items_in_db": collection.count() if collection else 0}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def answer_query(request: QueryRequest):
    """Answers a user's query using the RAG pipeline."""
    if not all([embedding_model, collection, llm_client]):
        raise HTTPException(status_code=503, detail="Service is unavailable due to model loading failure.")

    try:
        # --- The RAG Pipeline in Action ---
        
        # 1. RETRIEVAL: Embed the user's query and retrieve relevant context
        logger.info(f"Embedding user query: '{request.query[:50]}...'")
        query_embedding = embedding_model.encode(request.query).tolist()
        
        retrieved_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3 # Retrieve the top 3 most relevant chunks
        )
        retrieved_chunks = retrieved_results['documents'][0]
        
        # 2. AUGMENTATION: Create the augmented prompt
        context_str = "\n\n---\n\n".join(retrieved_chunks)
        augmented_prompt = f"""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Do not make up an answer.
        Answer in the same language as the question.

        Context:
        {context_str}

        Question: {request.query}

        Answer:
        """
        
        # 3. GENERATION: Call the LLM to generate the final answer
        logger.info("Generating final answer with LLM...")
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0.0 # Low temperature for factual, grounded answers
        )
        final_answer = response.choices[0].message.content
        
        return QueryResponse(answer=final_answer, retrieved_context=retrieved_chunks)

    except Exception as e:
        logger.exception("An error occurred during the RAG pipeline execution.")
        raise HTTPException(status_code=500, detail=str(e))