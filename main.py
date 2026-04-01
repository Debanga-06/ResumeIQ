"""
main.py — FastAPI backend for ResumeIQ

Endpoints:
  GET  /          → root (Render health check fallback)
  GET  /health    → model status + memory
  POST /analyze   → single resume ML analysis
  POST /batch     → rank up to 30 resumes
  POST /improve   → Gemini-powered resume coaching

Local:   uvicorn main:app --reload --port 8000
Render:  uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
Docs:    http://localhost:8000/docs

ARCHITECTURE DECISIONS:
  - Models are NOT loaded at import time (lazy via model_loader.py)
  - Single worker enforced on free tier to prevent RAM multiplication
  - google-genai (new unified SDK) replaces deprecated google-generativeai
  - $PORT is read from environment — Render injects it automatically
"""

import gc
import json
import os
import time
import sys
import traceback
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional

warnings.filterwarnings("ignore", message=".*position_ids.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# model_loader is imported here but models are NOT loaded yet.
# They load on first request that calls get_nlp() / get_sbert().
from model_loader import get_nlp, get_sbert, memory_usage_mb

# Nlp_pipeline imports the functions — it must NOT call get_nlp/get_sbert
# at module level either. If it does, patch it to call get_nlp() lazily.
from Nlp_pipeline import analyze_resume
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Gemini client ──────────────────────────────────────────────────────────────
# FIX: Use google-genai (new unified SDK, import as google.genai).
# The old google-generativeai package is deprecated and conflicts with
# the new google-genai package when both are installed.
#
# Install:  pip install google-genai
# DO NOT install google-generativeai alongside it.
_gemini_client = None

def _get_gemini_client():
    """Return a cached Gemini client, raising 500 if key is missing."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(
                500,
                "GEMINI_API_KEY not set. Add it to your .env file (local) "
                "or Render environment variables (production)."
            )
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ── Lifespan ───────────────────────────────────────────────────────────────────
# On FREE TIER: do NOT warm up here — let first request trigger lazy loading.
# On PAID TIER: uncomment warmup() to pre-load models on startup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] ResumeIQ API starting. RAM before models: {memory_usage_mb()} MB")

    # ── Uncomment for paid tier / local dev only ──
    # from model_loader import warmup
    # warmup()
    # print(f"[startup] Models loaded. RAM: {memory_usage_mb()} MB")

    yield  # app is running

    # Shutdown: release model memory
    from model_loader import _nlp, _sbert
    import model_loader
    model_loader._nlp   = None
    model_loader._sbert = None
    gc.collect()
    print("[shutdown] Models released.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ResumeIQ API",
    description="Real NLP/ML resume screening: spaCy + TF-IDF + BM25 + SBERT + Jaccard + ATS + Gemini coaching",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://resumeiq-lite.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────
class ScreenRequest(BaseModel):
    resume:          str           = Field(..., min_length=30)
    job_description: str           = Field(..., min_length=30)
    candidate_name:  Optional[str] = None


class ImproveRequest(BaseModel):
    resume:          str  = Field(..., min_length=30)
    job_description: str  = Field(..., min_length=30)
    analysis:        dict = Field(..., description="Full /analyze response — avoids re-running pipeline")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Root route. Render's health check may hit / before /health.
    Always return 200 so the service is marked healthy.
    """
    return {
        "service": "ResumeIQ API",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    """
    Health check with memory telemetry.
    Render requires this to return 200 within 30s of startup.
    """
    return {
        "status":       "ok",
        "version":      "2.0.0",
        "ram_usage_mb": memory_usage_mb(),

        # Report which models are currently loaded (lazy — may be None initially)
        "models": {
            "spacy_loaded": _is_loaded("_nlp"),
            "sbert_loaded": _is_loaded("_sbert"),
        },
    }


def _is_loaded(attr: str) -> bool:
    import model_loader
    return getattr(model_loader, attr) is not None


@app.post("/analyze")
async def analyze(req: ScreenRequest):
    """
    Run the full 5-signal NLP pipeline on one resume + JD pair.

    Signal weights:
      SBERT semantic      30%
      Skill match rate    28%
      BM25                20%
      TF-IDF cosine       12%
      Jaccard overlap     10%

    NOTE: First call triggers lazy model loading (~5-10s). Subsequent calls
    are fast (~1-2s). This is expected behaviour on the free tier.
    """
    if len(req.resume.strip()) < 30:
        raise HTTPException(400, "Resume too short — provide more text")
    if len(req.job_description.strip()) < 30:
        raise HTTPException(400, "Job description too short")

    t0 = time.perf_counter()
    try:
        result = analyze_resume(req.resume, req.job_description)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if req.candidate_name:
            result["candidate_name"] = req.candidate_name
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/batch")
async def batch_analyze(resumes: List[ScreenRequest]):
    """
    Screen up to 30 resumes. Returns ranked list (highest score first).
    All resumes share the same lazy-loaded model instances — no extra RAM per candidate.
    """
    if len(resumes) > 30:
        raise HTTPException(400, "Batch limit: 30 resumes per request")

    results = []
    for i, req in enumerate(resumes):
        t0 = time.perf_counter()
        try:
            r = analyze_resume(req.resume, req.job_description)
            r["index"]              = i
            r["candidate_name"]     = req.candidate_name or f"Candidate {i + 1}"
            r["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            results.append(r)
        except Exception as e:
            results.append({
                "index":          i,
                "candidate_name": req.candidate_name or f"Candidate {i + 1}",
                "error":          str(e),
                "score":          0,
            })

    ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    for pos, r in enumerate(ranked):
        r["rank"] = pos + 1

    return {
        "total":      len(ranked),
        "candidates": ranked,
        "summary": {
            "highest_score": ranked[0]["score"] if ranked else 0,
            "lowest_score":  ranked[-1]["score"] if ranked else 0,
            "avg_score": (
                round(sum(r.get("score", 0) for r in ranked) / len(ranked), 1)
                if ranked else 0
            ),
        },
    }


@app.post("/improve")
async def improve(req: ImproveRequest):
    """
    Generate 5 targeted resume improvements using Gemini 2.5 Flash.

    Accepts the full /analyze result to avoid re-running the ML pipeline.
    Requires GEMINI_API_KEY in environment.

    Uses google-genai (new unified SDK):
      pip install google-genai
      import from google.genai — NOT google.generativeai
    """
    client = _get_gemini_client()

    # Pull the signals we need from the analysis dict
    analysis       = req.analysis
    score          = analysis.get("score", 0)
    missing_skills = analysis.get("skill_analysis", {}).get("missing_skills", [])
    partial_skills = analysis.get("skill_analysis", {}).get("partial_match_skills", [])
    ats_missing    = analysis.get("ats_analysis", {}).get("keywords_missing", [])
    ats_score      = analysis.get("ats_analysis", {}).get("ats_score", 0)
    tfidf_jd_terms = analysis.get("tfidf_analysis", {}).get("jd_top_terms", [])

    prompt = f"""You are an expert resume coach and technical recruiter with 15 years FAANG experience.

A candidate's resume was analysed against a job description using a real ML pipeline
(SBERT semantic similarity + BM25 + TF-IDF + skill matching + ATS simulation).

ML ANALYSIS RESULTS:
- Overall match score:          {score}/100
- Missing skills (JD required, absent in resume): {', '.join(missing_skills[:15]) or 'None'}
- Partially matched skills:     {', '.join(partial_skills[:10]) or 'None'}
- ATS score:                    {ats_score}/100
- ATS keywords missing:         {', '.join(ats_missing[:12]) or 'None'}
- High-weight JD terms (TF-IDF):{', '.join(tfidf_jd_terms[:10]) or 'None'}

RESUME (first 1500 chars):
{req.resume[:1500]}

JOB DESCRIPTION (first 1000 chars):
{req.job_description[:1000]}

Generate EXACTLY 5 actionable resume improvements grounded in the ML analysis above.
Each suggestion must:
1. Target a SPECIFIC weakness from the analysis (not generic advice)
2. Include exact text to ADD or REWRITE (real bullet points where relevant)
3. Explain the impact on ATS filtering or recruiter assessment

Return ONLY a raw JSON array — no markdown fences, no preamble, nothing else:
[
  {{
    "priority": 1,
    "category": "ATS Keywords | Skills Gap | Experience Framing | Quantification | Missing Section",
    "title": "max 8-word action title",
    "problem": "specific problem found by ML analysis",
    "fix": "exact text to add or rewrite",
    "impact": "why this matters for this specific role"
  }}
]"""

    try:
        # Run blocking Gemini SDK call in a thread pool — keeps FastAPI event
        # loop free to handle other requests while waiting for the API.
        import asyncio
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
        )

        # Extract text safely — SDK response shape can vary by version
        if hasattr(response, "text") and response.text:
            raw = response.text.strip()
        else:
            raw = response.candidates[0].content.parts[0].text.strip()

        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Find the JSON array bounds defensively
        json_start = raw.find("[")
        json_end   = raw.rfind("]") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError(
                f"Gemini did not return a JSON array. "
                f"First 300 chars of response:\n{raw[:300]}"
            )

        suggestions = json.loads(raw[json_start:json_end])

        return {
            "suggestions": suggestions,
            "model":       "gemini-2.5-flash",
            "count":       len(suggestions),
        }

    except json.JSONDecodeError as e:
        # Return HTTP 500 — not a silent 200 with an error body
        raise HTTPException(500, f"Gemini returned invalid JSON: {e}")
    except HTTPException:
        raise  # re-raise our own 500s (missing API key, etc.)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Improvement generation failed: {e}")


# ── Dev entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,       
        workers=1,         
    )
