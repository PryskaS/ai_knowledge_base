import chromadb
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Constants ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge.txt"

CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "ai_knowledge"

def main():
    """
    Main function to build our AI Knowledge Base.
    This is our offline "indexing" pipeline.f
    """
    print("--- Starting AI Knowledge Base Indexing Pipeline ---")

    # --- 1. Load the Document ---
    print(f"Loading knowledge base from: {KNOWLEDGE_BASE_PATH}")
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # --- 2. Split the Document into Chunks ---
    # Chunking is critical. It breaks down large texts into
    # smaller, semantically meaningful pieces that are easier for the model to retrieve and reason about.
    text_splitter = CharacterTextSplitter(
        separator="\n\n", # Split on double newlines (paragraphs)
        chunk_size=300,   # Max characters per chunk
        chunk_overlap=50, # Overlap between chunks to maintain context
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"Split document into {len(chunks)} chunks.")

    # --- 3. Initialize Embedding Model and Vector Database ---
    print("Initializing embedding model and vector database...")
    # Using a small, fast, but effective embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # ChromaDB setup. PersistentClient saves the DB to disk.
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create the collection (like a table in a traditional DB)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # --- 4. Embed and Store the Chunks ---
    print("Embedding chunks and storing them in the vector database...")
    for i, chunk in enumerate(chunks):
        # Create a numerical vector (embedding) for the chunk
        embedding = embedding_model.encode(chunk).tolist()
        
        # Store the chunk, its embedding, and a unique ID in the collection
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )
    
    print("\n✅ --- Indexing Pipeline Complete ---")
    print(f"Vector database created at: {CHROMA_DB_PATH}")
    print(f"Number of items in collection '{COLLECTION_NAME}': {collection.count()}")


if __name__ == "__main__":
    main()