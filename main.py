"""
main.py — FastAPI backend for AI Resume Screener

Endpoints:
  GET  /health    → confirm models loaded
  POST /analyze   → single resume analysis
  POST /batch     → rank multiple resumes against one JD
  POST /improve   → AI-generated resume improvement suggestions (Gemini)

Start:  uvicorn main:app --reload --port 8000
Docs:   http://localhost:8000/docs
"""

import os
import time
import logging
import traceback
import json
import asyncio
import warnings

from dotenv import load_dotenv
load_dotenv()

import google.genai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from Nlp_pipeline import analyze_resume

# ── Logging ──────────────────────────────────────────────────────────────────
# Configured here (the app entrypoint) so Nlp_pipeline's logger.info/.exception
# calls also show up in Render's logs, with stage timings and tracebacks.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("resumeiq.api")

# ── Suppress noisy warnings ────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*position_ids.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ResumeIQ API",
    description="Real NLP/ML resume screening: spaCy + TF-IDF + BM25 + SBERT + Jaccard + ATS",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://resumeiq-1-c8ry.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini client (initialised once at startup) ────────────────────────────────
# BUG FIX: Previously the client was re-created on every /improve request.
# Creating it once at module level is faster and the correct pattern.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_gemini_client: genai.Client | None = None

if _GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=_GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — /improve endpoint will return 500.")


# ── Request / Response models ──────────────────────────────────────────────────
class ScreenRequest(BaseModel):
    resume: str          = Field(..., min_length=30, description="Full resume text")
    job_description: str = Field(..., min_length=30, description="Job description text")
    candidate_name: Optional[str] = Field(None, description="Optional label for batch export")


class ImproveRequest(BaseModel):
    resume: str          = Field(..., min_length=30)
    job_description: str = Field(..., min_length=30)
    analysis: dict       = Field(..., description="The full /analyze result — avoids re-running the pipeline")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


# ── /health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "models_loaded": True, "version": "2.0.0"}


# ── /analyze ───────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: ScreenRequest):
    """
    Run the full 5-signal NLP pipeline on one resume + JD pair.

    Signals:
      SBERT semantic similarity  (30%)
      Skill match rate           (25%)
      BM25 probabilistic         (20%)
      TF-IDF cosine              (15%)
      Jaccard overlap            (10%)

    Also returns: ATS score simulation, spaCy NER extraction, TF-IDF term analysis.

    BUG FIX (root cause of production 502s): this route is `async def`, but
    `analyze_resume()` is a fully synchronous, CPU-bound function (spaCy +
    SBERT + fuzzy matching). Calling it directly blocked the single asyncio
    event loop for the entire duration of the request — on Render's shared
    CPU that could run long enough to trip the platform's upstream timeout,
    which surfaces to the browser as a 502 rather than a normal FastAPI
    error response. Running it via asyncio.to_thread() keeps the event loop
    free to serve /health and other requests while the analysis runs.
    """
    if len(req.resume.strip()) < 30:
        raise HTTPException(400, "Resume too short")
    if len(req.job_description.strip()) < 30:
        raise HTTPException(400, "Job description too short")

    req_start = time.time()
    logger.info(
        "/analyze: request received (resume=%d chars, jd=%d chars, candidate=%s)",
        len(req.resume), len(req.job_description), req.candidate_name or "-",
    )
    try:
        result = await asyncio.to_thread(analyze_resume, req.resume, req.job_description)
        result["processing_time_ms"] = round((time.time() - req_start) * 1000, 1)
        if req.candidate_name:
            result["candidate_name"] = req.candidate_name
        logger.info(
            "/analyze: request completed in %.1fms (score=%s)",
            result["processing_time_ms"], result.get("score"),
        )
        return result
    except Exception as e:
        logger.exception("/analyze: analysis failed after %.1fms", (time.time() - req_start) * 1000)
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ── /batch ─────────────────────────────────────────────────────────────────────
@app.post("/batch")
async def batch_analyze(resumes: List[ScreenRequest]):
    """
    Screen multiple resumes. Returns all results sorted by score descending.
    Max 30 resumes per request.

    Each resume is analyzed via asyncio.to_thread() for the same reason as
    /analyze — analyze_resume() is CPU-bound and must not block the event
    loop, especially across up to 30 sequential calls in one request.
    """
    if len(resumes) > 30:
        raise HTTPException(400, "Batch limit is 30 resumes per request")

    logger.info("/batch: request received (%d resumes)", len(resumes))
    results = []
    for i, req in enumerate(resumes):
        req_start = time.time()
        try:
            r = await asyncio.to_thread(analyze_resume, req.resume, req.job_description)
            r["index"]              = i
            r["candidate_name"]     = req.candidate_name or f"Candidate {i + 1}"
            r["processing_time_ms"] = round((time.time() - req_start) * 1000, 1)
            results.append(r)
        except Exception as e:
            logger.exception("/batch: candidate %d failed", i)
            results.append({
                "index":          i,
                "candidate_name": req.candidate_name or f"Candidate {i + 1}",
                "error":          str(e),
                "score":          0,
            })

    logger.info("/batch: request completed (%d results)", len(results))
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)


# ── /improve ───────────────────────────────────────────────────────────────────
@app.post("/improve")
async def improve(req: ImproveRequest):
    """
    Generate 5 targeted resume improvement suggestions using Gemini.

    Requires GEMINI_API_KEY in environment / .env file.
    Accepts the full /analyze result so the ML pipeline is NOT re-run.
    """
    if _gemini_client is None:
        raise HTTPException(500, "GEMINI_API_KEY not set — add it to your .env file")

    analysis       = req.analysis
    score          = analysis.get("score", 0)
    missing_skills = analysis.get("skill_analysis", {}).get("missing_skills", [])
    partial_skills = analysis.get("skill_analysis", {}).get("partial_match_skills", [])
    # BUG FIX: ats_analysis's actual field is "missing_keywords" (see
    # Nlp_pipeline.py's compute_ats_score()), not "keywords_missing" — the
    # old key name here meant ats_missing was silently always [] and never
    # reached the Gemini prompt below.
    ats_missing    = analysis.get("ats_analysis", {}).get("missing_keywords", [])
    ats_score      = analysis.get("ats_analysis", {}).get("ats_score", 0)
    tfidf_jd_terms = analysis.get("tfidf_analysis", {}).get("jd_top_terms", [])

    prompt = f"""You are an expert resume coach and technical recruiter with 15 years experience at FAANG companies.

A candidate's resume was analysed against a job description using ML (SBERT + BM25 + TF-IDF + skill matching).

ANALYSIS RESULTS:
- Overall match score: {score}/100
- Missing skills (required by JD, absent in resume): {', '.join(missing_skills[:15]) or 'None identified'}
- Partially matched skills (close but wrong terminology): {', '.join(partial_skills[:10]) or 'None'}
- ATS keyword score: {ats_score}/100
- ATS keywords missing: {', '.join(ats_missing[:12]) or 'None'}
- High-importance JD terms (TF-IDF weighted): {', '.join(tfidf_jd_terms[:10]) or 'None'}

RESUME (first 1500 chars):
{req.resume[:1500]}

JOB DESCRIPTION (first 1000 chars):
{req.job_description[:1000]}

Generate exactly 5 specific, actionable resume improvements. Each must:
1. Target a SPECIFIC weakness identified in the ML analysis above
2. Include a concrete example of what to ADD or CHANGE (actual bullet point text where relevant)
3. Explain WHY it matters (ATS filter? Recruiter expectation? Skill gap?)

Return ONLY a JSON array — no markdown fences, no explanation, nothing else:
[
  {{
    "priority": 1,
    "category": "ATS Keywords | Skills Gap | Experience Framing | Quantification | Missing Section",
    "title": "short action title (max 8 words)",
    "problem": "what is currently wrong or missing",
    "fix": "exact text to add or change",
    "impact": "why this matters for this specific role"
  }}
]"""

    try:
        # asyncio.to_thread() runs the blocking Gemini SDK call in a thread pool
        # so FastAPI's async event loop is not blocked.
        response = await asyncio.to_thread(
            _gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if hasattr(response, "text") and response.text:
            raw = response.text.strip()
        else:
            raw = response.candidates[0].content.parts[0].text.strip()

        json_start = raw.find("[")
        json_end   = raw.rfind("]") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError(f"Gemini did not return a JSON array. Raw output:\n{raw[:300]}")

        suggestions = json.loads(raw[json_start:json_end])

        return {
            "suggestions": suggestions,
            "model": "gemini-2.5-flash",
        }

    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Gemini returned invalid JSON: {str(e)}")
    except Exception as e:
        logger.exception("/improve: generation failed")
        traceback.print_exc()
        raise HTTPException(500, f"Improvement generation failed: {str(e)}")


# ── Local dev entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
