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
... (You can include the Mermaid diagram from the Portuguese version here) ...

## 🏁 Getting Started
... (The "Getting Started" section can be adapted from the Portuguese version) ...
---