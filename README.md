# 📚 RAG-LlamaIndex -- Retrieval Augmented Generation with LlamaIndex

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-orange)
![LLM](https://img.shields.io/badge/LLM-Llama3.1-purple)

# 📌 Project Overview

This project implements a Retrieval Augmented Generation (RAG) pipeline
using LlamaIndex, ChromaDB, HuggingFace embeddings, and Groq LLM.

The system loads documents from a directory, converts them into vector
embeddings, stores them in a Chroma vector database, and retrieves
relevant information to answer user queries using a large language
model.

------------------------------------------------------------------------

## Tech Stack

-   Python
-   LlamaIndex
-   ChromaDB
-   HuggingFace Embeddings
-   NLTK
-   Groq LLM (Llama 3.1)

------------------------------------------------------------------------

# 📂 Project Structure

RAG-LlamaIndex │ ├── documents/ ├── vector-db/ ├── document_ingestion.py
├── retrieval.py ├── constants.py ├── requirements.txt └── README.md

------------------------------------------------------------------------

## How It Works

1.  Load documents using SimpleDirectoryReader
2.  Split documents into nodes using SimpleNodeParser
3.  Generate embeddings using HuggingFaceEmbedding
4.  Store vectors in ChromaDB
5.  Retrieve relevant nodes based on query
6.  Send context to Groq LLM to generate response

------------------------------------------------------------------------

# 🚀 Installation

Clone the repository

git clone https://github.com/CoderVindra/RAG-LlamaIndex.git cd
RAG-LlamaIndex

Create virtual environment

python -m venv venv

Activate environment

Windows venv`\Scripts`{=tex}`\activate`{=tex}

Mac/Linux source venv/bin/activate

Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

# 🔑 Environment Variables

Create a .env file

GROQ_API_KEY=your_api_key

------------------------------------------------------------------------

# ▶️ Running the Project

Step 1: Add documents inside documents/

Step 2: Create vector database python document_ingestion.py

Step 3: Query the system python retrieval.py

Example query: What does document say about deductive reasoning?

------------------------------------------------------------------------

# ✨ Features

-   Document based question answering
-   Vector similarity search
-   Persistent vector database
-   HuggingFace embeddings
-   Groq powered LLM inference

------------------------------------------------------------------------

# 👨‍💻 Author

**Ravindra Pawar**

Backend & AI Developer\
Python \| LangChain \| RAG \| LLM Applications
