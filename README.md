# AskPDF - Basic RAG

This Streamlit application allows you to upload a PDF document and interactively ask questions about its content using a Large Language Model (LLM) hosted on Hugging Face. The app processes the PDF text, creates vector embeddings for semantic search, and leverages the Hugging Face LLaMA model to generate answers based on user queries.

---

## Features

- Upload PDF documents and extract their text content.
- Split large documents into manageable chunks for efficient embedding and search.
- Create or load a vector store index (using FAISS) to perform similarity search on the document chunks.
- **Basic Retrieval-Augmented Generation (RAG):** retrieves relevant document chunks from the PDF via similarity search and uses a language model to generate answers.  
- Query the document by asking questions in natural language.
- Generate answers using the Hugging Face Inference API for the `decapoda-research/llama-7b-hf` model.
- Simple, interactive UI built with Streamlit.

---

## Requirements

- Python 3.8+
- Streamlit
- PyPDF2
- langchain
- langchain-community (for FAISS vector store support)
- faiss-cpu
- requests
- python-dotenv

Install dependencies with:

```bash
pip install streamlit PyPDF2 langchain langchain-community faiss-cpu requests python-dotenv
