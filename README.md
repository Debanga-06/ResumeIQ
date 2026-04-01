# ResumeIQ — AI-Powered Resume Screening Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.7+-09A3D5?style=flat-square&logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-e8b84b?style=flat-square)
![Version](https://img.shields.io/badge/Version-2.0.0-success?style=flat-square)

**A real NLP/ML resume screening pipeline — no LLM shortcuts, actual models.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Setup](#-setup) • [API Reference](#-api-reference) • [How It Works](#-how-it-works) • [Project Structure](#-project-structure)

</div>

---

## Overview

ResumeIQ is a full-stack resume screening platform that uses a **real NLP/ML pipeline** to match candidates to job descriptions. Unlike tools that outsource everything to a language model, ResumeIQ uses spaCy, TF-IDF, BM25, Sentence-BERT, and Jaccard similarity as proper signal sources — each chosen for a specific reason and combined in a weighted ensemble.

Built as both a functional product and a portfolio-grade demonstration of applied NLP.

---

## Features

| Feature | Description |
|---|---|
| **ML Match Score** | Weighted ensemble of 5 NLP signals (0–100) |
| **BM25 Ranking** | Probabilistic term ranking — the algorithm behind Elasticsearch |
| **ATS Simulation** | Predicts whether a resume would pass automated keyword filters |
| **Skill Gap Analysis** | Matched, missing, and partial skills with fuzzy matching |
| **spaCy NER** | Extracts organizations, technologies, dates, and locations |
| **TF-IDF Interpretability** | Shows which terms drove the score |
| **AI Coaching** | Gemini generates 3 targeted resume improvement suggestions |
| **Batch Screening** | Upload 20 resumes at once, ranked by score, exportable to CSV |
| **PDF & Image Upload** | PDF.js for text PDFs, Claude Vision OCR for scanned images |
| **Saved Sessions** | localStorage-based history — compare analyses over time |

---

## Tech Stack

### Backend

| Library | Purpose |
|---|---|
| **FastAPI** | Async REST API framework |
| **spaCy `en_core_web_md`** | NER, POS tagging, lemmatization, word vectors |
| **Sentence-Transformers** | SBERT semantic embeddings (`all-MiniLM-L6-v2`) |
| **scikit-learn** | TF-IDF vectorization, cosine similarity |
| **rank-bm25** | BM25Okapi probabilistic ranking |
| **rapidfuzz** | Fuzzy skill matching |
| **httpx** | Async HTTP client for Claude API calls |

### Frontend

| Technology | Purpose |
|---|---|
| **Vanilla HTML/CSS/JS** | Zero-dependency frontend, no build step required |
| **PDF.js** | Client-side PDF text extraction |
| **Gemini Vision API** | OCR for scanned resumes and images |
| **localStorage** | Persistent session storage |

---

## Setup

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ResumeAI.git
cd ResumeAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy model

```bash
python -m spacy download en_core_web_md
```

### 4. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

You should see:

```
Loading spaCy model...
Loading SBERT model...
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 5. Open the frontend

```bash
open index.html      # macOS
start index.html     # Windows
xdg-open index.html  # Linux
```

Or drag `index.html` into any modern browser. The frontend connects to `localhost:8000` automatically.

### 6. Verify everything works

```bash
curl http://localhost:8000/health
# → {"status":"ok","models_loaded":true,"version":"2.0.0"}
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Reference

### `GET /health`

Returns server status and model load confirmation.

```json
{
  "status": "ok",
  "models_loaded": true,
  "version": "2.0.0"
}
```

---

### `POST /analyze`

Run the full ML pipeline on a single resume + job description pair.

**Request body:**
```json
{
  "resume": "John Doe — Senior Engineer...",
  "job_description": "We are looking for a Python engineer..."
}
```

**Response (abbreviated):**
```json
{
  "score": 74,
  "scoring_breakdown": {
    "component_scores": {
      "semantic_similarity": 81.2,
      "skill_match_rate": 66.7,
      "bm25_similarity": 73.4,
      "tfidf_cosine": 61.0,
      "jaccard_overlap": 52.3
    }
  },
  "ats_analysis": {
    "ats_score": 68,
    "ats_verdict": "Likely to pass ATS",
    "keywords_found": ["python", "fastapi", "docker"],
    "keywords_missing": ["graphql", "kafka"]
  },
  "skill_analysis": {
    "matched_skills": ["python", "react", "postgresql"],
    "missing_skills": ["graphql", "kubernetes"],
    "partial_match_skills": ["postgres"]
  },
  "processing_time_ms": 1842.3
}
```

---

### `POST /suggest`

Generate 3 AI-powered resume improvement suggestions grounded in the ML analysis.

**Request body:**
```json
{
  "resume": "...",
  "job_description": "...",
  "missing_skills": ["graphql", "kubernetes"],
  "matched_skills": ["python", "react"],
  "score": 74,
  "ats_score": 68,
  "ats_verdict": "Likely to pass ATS"
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "priority": "high",
      "section": "Skills",
      "title": "Add GraphQL to skills section",
      "problem": "GraphQL is required but absent from the resume entirely",
      "fix": "Add 'GraphQL (familiar, learning)' under web technologies in the Skills section",
      "impact": "ATS systems scan for exact keyword matches — absence causes immediate filtering"
    }
  ],
  "overall_advice": "The biggest gap is GraphQL — even one line mentioning exposure would significantly improve ATS pass rate."
}
```

---

### `POST /batch`

Screen up to 20 resumes simultaneously, returned ranked by score.

**Request body:**
```json
[
  {
    "candidate_name": "Jane Smith",
    "resume": "Jane Smith — Full Stack Engineer...",
    "job_description": "We are looking for..."
  }
]
```

**Response:**
```json
{
  "total": 2,
  "summary": {
    "highest_score": 82,
    "avg_score": 66.5,
    "avg_ats": 59.0
  },
  "candidates": [
    { "rank": 1, "candidate_name": "Jane Smith", "score": 82 },
    { "rank": 2, "candidate_name": "Bob Lee", "score": 51 }
  ]
}
```

---

## How It Works

### The ML Pipeline

```
Raw text (resume + JD)
  │
  ├─ 1. Section segmentation     regex header detection
  │                               → splits into experience / skills / education
  │
  ├─ 2. Text preprocessing       spaCy pipeline
  │                               → tokenize → lemmatize → remove stopwords
  │
  ├─ 3. Skill extraction         two strategies in parallel
  │                               → fuzzy taxonomy matching (rapidfuzz, threshold 85)
  │                               → spaCy NER for PRODUCT / ORG entities
  │
  ├─ 4. TF-IDF cosine similarity  sklearn TfidfVectorizer
  │                               → ngram_range=(1,2), sublinear_tf=True
  │
  ├─ 5. BM25 similarity           rank_bm25 BM25Okapi
  │                               → JD tokens as query, resume as document
  │                               → normalized against self-score upper bound
  │
  ├─ 6. SBERT semantic similarity sentence-transformers all-MiniLM-L6-v2
  │                               → 384-dim embeddings, L2-normalized
  │                               → cosine = dot product (fast)
  │
  ├─ 7. Jaccard + skill overlap   |A ∩ B| / |A ∪ B|
  │                               → fuzzy partial match extension (threshold 80)
  │
  ├─ 8. ATS score                 keyword rate (60%)
  │                               + section presence (25%)
  │                               + formatting signals (15%)
  │
  └─ 9. Weighted ensemble score
         0.30 × SBERT
       + 0.28 × skill_match_rate
       + 0.20 × BM25
       + 0.12 × TF-IDF
       + 0.10 × Jaccard
```

### Scoring Weights Explained

| Signal | Weight | Rationale |
|---|---|---|
| SBERT semantic | 30% | Captures meaning across paraphrases — "built APIs" ≈ "backend service development" even with zero shared words |
| Skill match rate | 28% | Direct coverage of JD requirements — the most job-relevant signal |
| BM25 | 20% | Better than TF-IDF for short docs — length-normalized and term-saturating |
| TF-IDF cosine | 12% | Keyword and terminology alignment; kept for score interpretability |
| Jaccard overlap | 10% | Raw vocabulary overlap sanity check; least nuanced, lowest weight |

> Weights are expert-informed heuristics. Ideal next step: tune with logistic regression on labeled hire/reject data.

### Why BM25 Instead of Pure TF-IDF?

TF-IDF linearly rewards term repetition — mentioning "Python" 10 times scores 10× higher than once. BM25 applies a **saturation curve** (parameter `k1=1.5`) so the 10th mention adds almost nothing. It also applies **document length normalization** (parameter `b=0.75`) preventing long resumes from scoring higher purely from word count. BM25 is the ranking algorithm behind Elasticsearch, Solr, and early Google.

### Why the ATS Score Matters

Most Fortune 500 companies use Applicant Tracking Systems (Workday, Greenhouse, Taleo) that do **exact keyword matching** before any human reads a resume. `"ML engineer"` ≠ `"machine learning engineer"` in a naive ATS. The ATS score answers: *"Would this resume even reach a recruiter's inbox?"*

---

## Project Structure

```
ResumeAI/
│
├── Nlp_pipeline.py          # Core ML/NLP logic
│   ├── preprocess_text()        spaCy lemmatization + stopword removal
│   ├── tokenize_for_bm25()      lightweight tokenization for BM25
│   ├── parse_sections()         regex-based resume section segmentation
│   ├── extract_skills()         fuzzy taxonomy matching + spaCy NER
│   ├── tfidf_similarity()       sklearn TF-IDF cosine similarity
│   ├── get_tfidf_top_terms()    most important terms for interpretability
│   ├── bm25_similarity()        BM25Okapi normalized similarity
│   ├── semantic_similarity()    SBERT 384-dim embedding cosine
│   ├── skill_overlap_analysis() Jaccard + fuzzy partial matching
│   ├── compute_ats_score()      ATS keyword + structure + formatting check
│   ├── extract_named_entities() spaCy NER for ORG, DATE, GPE, PRODUCT
│   ├── composite_score()        weighted ensemble scoring
│   └── analyze_resume()         master pipeline orchestrator
│
├── main.py                  # FastAPI REST API
│   ├── GET  /health             status + model load check
│   ├── POST /analyze            single resume ML analysis
│   ├── POST /suggest            Claude AI improvement suggestions
│   └── POST /batch              multi-resume screening + ranking
│
├── index.html               # Frontend (no build step needed)
│   ├── Single Screen tab        upload or paste → analyze one resume
│   ├── Batch tab                multi-file drag-and-drop → ranked table → CSV export
│   ├── Results tab              score ring, ATS panel, skills, TF-IDF, NER, AI coaching
│   └── Saved Sessions tab       localStorage history with score comparison
│
└── requirements.txt         # Python dependencies
```

---

## NLP/ML Concepts Demonstrated

This project is designed to show — and explain — real techniques:

| Concept | Where Used |
|---|---|
| Lemmatization | `preprocess_text()` — "running" → "run" |
| Stopword removal | `preprocess_text()` — removes "the", "and", etc. |
| TF-IDF | `tfidf_similarity()` — term weight = frequency × rarity |
| N-gram features | `TfidfVectorizer(ngram_range=(1,2))` — "machine learning" as one token |
| BM25 | `bm25_similarity()` — probabilistic ranking with saturation |
| Dense embeddings | `semantic_similarity()` — 384-dim SBERT vectors |
| Cosine similarity | Both TF-IDF and SBERT — angle between vectors, length-invariant |
| Jaccard similarity | `skill_overlap_analysis()` — `\|A ∩ B\| / \|A ∪ B\|` |
| Fuzzy matching | `extract_skills()` — Levenshtein-based approximate matching |
| Named entity recognition | `extract_named_entities()` — spaCy ORG, DATE, GPE, PRODUCT |
| Ensemble scoring | `composite_score()` — weighted combination of signals |

---

## Extending the Project

| Extension | Approach |
|---|---|
| Fine-tune SBERT on resume data | `SentenceTransformer` + `CosineSimilarityLoss` on labeled pairs |
| Tune ensemble weights with data | Collect labeled hire/reject data → logistic regression on scores |
| Redis caching | Hash inputs with `hashlib.sha256`, store results — ~5ms vs ~2s |
| Rate limiting | `pip install slowapi` — 10 requests/minute per IP |
| Deploy to Railway | `railway up`, set `PORT=8000` environment variable |
| Containerize | `docker build -t resumeiq . && docker run -p 8000:8000 resumeiq` |

---

## License
This project is licensed under the GNU GPL v3.0 License.
See the LICENSE file for details.

---

<div align="center">

Built with Python · spaCy · Sentence-Transformers · FastAPI · Gemini API

</div>
