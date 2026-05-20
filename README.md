# Farmer Advisory System using RAG

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based Farmer Advisory System using agriculture-related documents.

The system allows users to ask natural language agriculture questions and receive context-aware answers generated using Large Language Models (LLMs).

The project explores and compares multiple semantic retrieval approaches:

- BM25 Keyword Retrieval
- Embedding-Based Semantic Retrieval
- Hybrid Retrieval (BM25 + Embeddings)

---

## Objectives

- Build a domain-specific RAG system for agriculture advisory
- Compare semantic retrieval techniques
- Improve farmer query understanding using embeddings
- Generate grounded answers using OpenAI LLMs
- Deploy an interactive Streamlit application

---

## Technologies Used

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| LLM | OpenAI GPT-4o-mini |
| Semantic Search | Hybrid Retrieval |

---

## System Architecture

User Query  
↓  
Hybrid Retrieval (BM25 + Embeddings)  
↓  
Top Relevant Chunks  
↓  
OpenAI LLM  
↓  
Generated Farmer Advisory Answer  
↓  
Streamlit Web Interface

---

## Running the Application

### Install dependencies

```bash
pip install -r requirements.txt

Run Streamlit app
python -m streamlit run app.py


