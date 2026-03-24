---
title: RAG Debugger
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# RAG Debugger: Making LLM Systems Reliable

Most RAG systems focus on getting *an answer*.  
This project focuses on something more important:

**What happens when the system fails - and how to fix it.**

---

## Overview

This is a production-style Retrieval-Augmented Generation (RAG) system with a built-in debugging and self-improvement layer.

Instead of treating the LLM as a black box, the system:

- evaluates answer quality
- detects failure modes
- explains what went wrong
- automatically improves the result when possible

The UI makes this behavior visible in real time using a **before vs after comparison**.

---

## Key Capabilities

### 1. Hybrid Retrieval
Combines:
- Vector search (FAISS)
- Keyword search (BM25)

This improves recall and makes retrieval more robust to different query styles.

### 2. Answer Generation (LLM)
- Uses Groq (LLaMA 3.1)
- Deterministic decoding for reproducibility
- Prompting optimized for grounded responses

### 3. Evaluation Layer

Each response is evaluated using:

- **Retrieval score** - how relevant the documents are
- **Grounding score** - how well the answer uses retrieved context
- **Confidence score** - combined signal for reliability

### 4. Failure Detection

The system explicitly classifies failures into:

- `query_issue` -> poorly formed or repetitive queries
- `missing_context` -> knowledge base lacks relevant information
- `grounding_failure` -> answer is weak or not supported by context

This makes the system explainable and debuggable.

### 5. Auto-Fix Engine

When a failure is detected, the system attempts to fix it:

- rewrites queries (for example, removes repetition)
- expands queries for better retrieval
- increases retrieval depth when needed

Only applied when necessary - good queries are left untouched.

### 6. Answer Quality Validation

Prevents misleading outputs by detecting:

- trivial responses (for example, one-word answers)
- weak grounding

These are marked as failures instead of being treated as valid answers.

### 7. Deterministic Behavior

Same query -> same result.

- LLM randomness removed
- retrieval stabilized
- thresholds tuned for consistency

This makes the system more reliable and debuggable.

### 8. Debug UI

A simple interface shows:

- Before vs After results
- Confidence changes
- Failure type
- Fixes applied
- Diagnosis and suggestions

This makes the system behavior transparent and easy to reason about.

---

## Example

**Input:**
```text
refund refund refund refund
```

**Before:**
- Answer: I don't know
- Confidence: 19%
- Failure: query_issue

**After Auto-Fix:**
- Query: `refund information`
- Answer: Refunds can be created via the API using `/v1/refunds`...
- Confidence: 78%
- Improvement: +59%

---

## Architecture

```text
User Query
  ->
Hybrid Retriever (FAISS + BM25)
  ->
LLM Generator (Groq)
  ->
Evaluation Layer
  ->
Failure Detection
  ->
Auto-Fix Engine (if needed)
  ->
Final Response + Debug Signals
```

---

## Tech Stack

- **Backend:** FastAPI
- **Vector Search:** FAISS
- **Keyword Search:** BM25
- **LLM:** Groq (LLaMA 3.1)
- **Frontend:** HTML + Tailwind

---

## Running Locally

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the environment variable:

```bash
set GROQ_API_KEY=your_key_here
```

PowerShell:

```powershell
$env:GROQ_API_KEY="your_key_here"
$env:FRONTEND_ORIGINS="http://127.0.0.1:8000"
```

4. Start the backend:

```bash
python -m uvicorn app.main:app
```

5. Open the UI:

```text
http://127.0.0.1:8000/ui
```

---

## Deploying to Hugging Face Spaces

This repo can be deployed as a Docker Space with the full system intact.

1. Create a new Space on Hugging Face.
2. Select:
   - **SDK:** Docker
   - **Hardware:** CPU Basic (free)
3. Push this repository to the Space.
4. In the Space settings, add these secrets:

```text
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

5. Wait for the Docker build to finish.

The Space will serve the UI at the root URL, and the API will still be available from:

```text
/query
/search
/health
/api
```

Notes:

- Hugging Face Spaces free CPU hardware provides significantly more RAM than Render free, which makes it a better fit for the full embedding-based system.
- The first query can still be slower than later ones because the embedding model may need to warm up.

---

## Deploying to Render and Vercel

This repo is set up for a split deployment:

- **Render** hosts the FastAPI backend
- **Vercel** hosts the static frontend

### Render backend

1. Create a new **Web Service** on Render and connect this repo.
2. Render will pick up [render.yaml](render.yaml).
3. Set these environment variables in Render:

```text
GROQ_API_KEY=your_groq_api_key
FRONTEND_ORIGINS=https://your-vercel-project.vercel.app
```

4. Deploy the service and copy the public Render URL.

### Vercel frontend

1. Import the same repo into Vercel.
2. The frontend is served from [frontend/index.html](frontend/index.html).
3. Update [vercel.json](vercel.json) and replace:

```text
https://your-render-service.onrender.com
```

with your real Render backend URL.

4. Deploy on Vercel.

After that:

- the frontend root URL loads the UI
- `/query` on Vercel is proxied to the Render backend
- the backend accepts requests only from the configured `FRONTEND_ORIGINS`

If you want to keep everything on Render only, you can also skip Vercel and use:

```text
https://your-render-service.onrender.com/ui
```

---

## Why This Project

Most RAG examples stop at "it works".

In practice, the harder problem is:

- detecting when it *doesn't* work
- understanding why
- improving it without manual intervention

This project focuses on that gap.

---

## Key Takeaways

- Reliability > raw generation
- Observability is critical in LLM systems
- Failure modes should be explicit, not hidden
- Small improvements (query rewrite, retrieval tuning) can significantly improve results

---

## Future Improvements

- multi-document ranking optimization
- better grounding metrics (semantic overlap, attribution)
- feedback loop with user corrections
- safer caching and latency optimization

---

## Feedback

Happy to hear thoughts, improvements, or critiques.  
This project is meant to explore how we can make LLM systems more reliable in practice.
