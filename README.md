# Hybrid Fashion Search Engine 🛍️

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB)](https://www.python.org/)
[![Dockerized](https://img.shields.io/badge/Docker-Container-2496ED)](https://www.docker.com/)
[![Search](https://img.shields.io/badge/Search-FAISS%20%2B%20BM25-00599C)](https://github.com/facebookresearch/faiss)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20App-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/stephenhoang/hm-semantic-search)

> **PROJECT STATUS:** 🚀 Completed & Deployed  
> **LIVE DEMO:** https://huggingface.co/spaces/stephenhoang/hm-semantic-search

## 📖 Introduction
This project implements a **Hybrid Information Retrieval System** for fashion e-commerce, solving the classic **"Vocabulary Mismatch Problem"** (e.g., searching for *"running shoes"* but products are labeled *"sneakers"*). 

By fusing **Lexical Search (BM25)** with **Semantic Search (Sentence-BERT)**, the system balances keyword precision with contextual understanding.

## 💡 Key Engineering Features
Based on the analysis of 100k+ H&M products, the system features:

* [cite_start]**Hybrid Architecture:** Uses a Weighted Score Fusion algorithm ($\alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{SBERT}$) to rank results[cite: 234].
* [cite_start]**Smart Text Enrichment:** A custom NLP pipeline (spaCy + Regex) that expands meaningful vocabulary from **6,721** to **14,370** tokens, validated via **Zipf's Law**[cite: 397, 401].
* [cite_start]**High-Performance Indexing:** Utilizes **FAISS (IndexFlatIP)** for millisecond-latency vector retrieval[cite: 218].
* [cite_start]**Recommendation Engine:** Solves the "Cold Start" problem using Item-to-Item Content-Based Filtering[cite: 229].

## 🛠 Tech Stack
* **Core:** Python, NumPy, Pandas.
* **Search & AI:** PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`), Rank-BM25, FAISS.
* **Deployment:** Docker, Streamlit, Hugging Face Spaces (CPU Basic).

## 🚀 How to Run Locally (Docker)
This project is fully containerized. To run it on your machine:

```bash
# 1. Build the image
docker build -t hybrid-fashion-search .

# 2. Run the container
docker run -p 8501:8501 hybrid-fashion-search
```


## Performance Showcase
The system successfully retrieves products even when keywords don't match:
- Query: "Protection from cold" $\rightarrow$ Result: "Puffer Jacket" (Score: 0.90)
- Query: "Running shoes" $\rightarrow$ Result: "Trainers/Sneakers" (Score: 0.92) 
