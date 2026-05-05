# NVIDIA AI System Advisor

A compact Retrieval-Augmented Generation (RAG) assistant for answering questions about NVIDIA AI tools: NeMo, Triton Inference Server, and TensorRT. It retrieves relevant documentation chunks with FAISS, generates grounded answers with Gemini, and presents them in a Streamlit chat UI with a light Dustin Henderson-inspired tone.

## What It Does

- Answers NVIDIA AI ecosystem questions using local documentation chunks
- Retrieves relevant context before generation to reduce hallucinations
- Cites source sections and documentation links
- Redirects out-of-scope NVIDIA questions to official docs
- Supports both Streamlit UI and terminal mode

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| LLM | Gemini via `google-genai` |
| Config | `.env` with `GEMINI_API_KEY` |

## Project Files

```text
app.py                Streamlit chat UI
rag_pipeline.py       RAG pipeline, Gemini calls, formatting, tone layer
chunker.py            Markdown chunking logic
embedder.py           Embedding + FAISS index creation
01_nemo.md            NeMo knowledge source
02_triton.md          Triton knowledge source
03_tensorrt.md        TensorRT knowledge source
chunks.json           Processed chunks
chunk_store.json      Chunk metadata and text
faiss_index.index     FAISS vector index
requirements.txt      Python dependencies
```

## How It Works

```text
User question
  -> embed query with MiniLM
  -> search FAISS for top matching chunks
  -> apply retrieval quality check
  -> send grounded prompt + context to Gemini
  -> format answer and sources
  -> apply Dustin-style tone
  -> render in Streamlit or terminal
```

The retrieval quality gate prevents weak matches from being sent to the LLM, which helps avoid confident but unsupported answers.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Get a Gemini API key from:

```text
https://aistudio.google.com/apikey
```

For Streamlit Cloud, add the same key under app **Secrets**:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

## Run The App

Recommended Streamlit UI:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

Terminal mode:

```powershell
.venv\Scripts\python.exe rag_pipeline.py
```

## Rebuild The Index

Only needed if you edit the markdown knowledge files:

```powershell
.venv\Scripts\python.exe chunker.py
.venv\Scripts\python.exe embedder.py
```

This regenerates:

```text
chunks.json
chunk_store.json
faiss_index.index
```

## Limitations

- Covers only NeMo, Triton, and TensorRT
- Other NVIDIA topics are redirected to official docs
- Knowledge freshness depends on the local markdown files
- Gemini free tier may rate-limit frequent usage
- Each query is handled independently, without long-term conversation memory

## Acknowledgements

Built for the NVIDIA NCA GenL Certification project using NVIDIA documentation, FAISS, Sentence Transformers, Streamlit, and Gemini.
