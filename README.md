# AI Knowledge Base (RAG Service) 🧠📚🔍

[![CI/CD Pipeline](https://github.com/PRYSKAS/ai_knowledge_base/actions/workflows/ci.yml/badge.svg)](https://github.com/PRYSKAS/ai_knowledge_base/actions)

An AI microservice that implements the **Retrieval-Augmented Generation (RAG)** architecture. This system enables a Large Language Model (LLM) to answer questions based on a specific, private knowledge base, eliminating hallucinations and providing factual, context-aware responses.

## 🧠 Core Concept: Giving an LLM a Long-Term Memory

The RAG architecture solves a fundamental problem with LLMs: their knowledge is static and generic. RAG connects the LLM to an external source of truth through two main phases:

1.  **Indexing Pipeline (Offline):** An ETL process where documents (text, PDFs, etc.) are loaded, split into smaller "chunks," and each chunk is converted into a numerical vector (an "embedding") using a specialized model. These vectors, representing the semantic meaning of the text, are stored in a **Vector Database**.
2.  **RAG Service (Online):** When a user asks a question, the system:
    * **Retrieves:** Converts the question into a vector and uses it to search the database for the most relevant text chunks (similarity search).
    * **Augments:** Inserts these retrieved chunks into a prompt along with the original question, instructing the LLM to use *only* this information to answer.
    * **Generates:** Sends the augmented prompt to the LLM, which generates a factual answer grounded in the provided context.

## 🚀 Engineering & MLOps Highlights

This project demonstrates the end-to-end construction of a production-ready Question & Answering system.

* **Offline Indexing Pipeline:** A modular and reusable script (`indexer.py`) that processes and vectorizes knowledge using `LangChain` (for text splitting), `SentenceTransformers` (for embeddings), and `ChromaDB` (for vector storage).
* **Optimized RAG Inference Service:** A **FastAPI** service that loads the models and database once on startup to ensure low-latency responses.
* **Hallucination Reduction:** The augmented prompt is designed to instruct the LLM to be factual and strictly base its answers on the provided context, a critical technique for business applications.
* **Standardized Infrastructure:** The application is fully containerized with **Docker** (including the vector database) and validated by a **GitHub Actions** pipeline, proving the robustness of our "AI service factory" pattern.

## 🏗️ System Architecture

This diagram illustrates the two phases of the RAG system:

```mermaid
graph TD
    subgraph "Phase 1: Indexing Pipeline (Offline)"
        A[Documents .txt] --> B{Text Splitter};
        B --> |Chunks| C{Embedding Model};
        C --> |Vectors| D[(Chroma Vector DB)];
    end

    subgraph "Phase 2: RAG Service (Online)"
        E[User] -->|Question| F(FastAPI Service);
        F -->|Embed Question| G{Embedding Model};
        G -->|Similarity Search| D;
        D -->|Relevant Context| F;
        F -->|Build Augmented Prompt| H[LLM];
        H -->|Factual Answer| F;
        F -->|JSON Response| E;
    end
    ```

## 🏁 Getting Started

### Prerequisites

* Git
* Python 3.9+
* Docker Desktop (running)
* An OpenAI API Key

### 1. Preparing the Knowledge Base (Indexing)

This step needs to be run only once, or whenever the `knowledge.txt` file is updated.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PRYSKAS/ai_knowledge_base.git
    cd ai_knowledge_base
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optional) Customize Knowledge:** Edit the `knowledge.txt` file in the project root with your own text content.
4.  **Run the Indexing Pipeline:**
    ```bash
    python -m indexing_pipeline.indexer
    ```
    This will create (or update) the `chroma_db` folder containing the vector embeddings of your knowledge base.

### 2. Running the Q&A Service

1.  **Set up environment variables:**
    * Create a `.env` file from the example: `copy .env.example .env` (on Windows) or `cp .env.example .env` (on Unix/macOS).
    * Add your `OPENAI_API_KEY` to the new `.env` file.

2.  **Run locally using Uvicorn (for development):**
    ```bash
    uvicorn serving.main:app --reload --port 8001
    ```
    Access the API documentation and interact with the service at `http://127.0.0.1:8001/docs`.

3.  **Run using Docker (Recommended for stable execution):**
    * **Build the Docker image:**
        ```bash
        docker build -t ai-knowledge-base-service .
        ```
        *(This copies the `chroma_db` folder into the image)*
    * **Run the container:**
        ```bash
        docker run -d -p 8001:8001 --env-file .env --name ai-kb-rag ai-knowledge-base-service
        ```
    Access the API at `http://127.0.0.1:8001/docs`.

---