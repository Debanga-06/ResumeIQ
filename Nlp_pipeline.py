"""
Nlp_pipeline.py — Real ML/NLP Resume Screening Pipeline

Models & algorithms used:
  - spaCy (en_core_web_sm)         → NER, POS tagging, lemmatization, noun chunks
  - scikit-learn TfidfVectorizer   → TF-IDF document vectors
  - sentence-transformers (SBERT)  → Dense semantic embeddings
  - rank_bm25 (BM25Okapi)          → Probabilistic term weighting
  - rapidfuzz                      → Fuzzy skill matching
  - sklearn cosine_similarity      → Vector distance

WHY BM25 over pure TF-IDF?
  TF-IDF has two problems with short docs like resumes:
    1. Term saturation: a word appearing 10x is not 10x more relevant than appearing 2x
    2. Document length bias: longer resumes get artificially boosted scores
  BM25 fixes both with:
    - A saturation parameter k1 (default 1.5) that dampens extreme term frequencies
    - A length normalization parameter b (default 0.75) that penalizes long docs fairly
  BM25 is the algorithm behind Elasticsearch and most production search engines.

WHY ATS score separately?
  Applicant Tracking Systems used by 99% of Fortune 500 companies do EXACT keyword
  matching — no semantics, no fuzzy. A candidate with 90% SBERT semantic similarity
  can fail ATS if they write "built APIs" instead of "REST API development".
  The ATS score tells candidates: "fix THIS specific wording to pass the filter."

PERFORMANCE NOTES (read this if /analyze is slow or 502s on Render)
  This file previously called `get_nlp()(text)` up to 7 times per request
  (preprocessing, skill extraction, ATS keywords, NER — twice each for
  resume + JD), with NO upper bound on input length. On a long resume/JD,
  that meant several full spaCy pipeline runs over uncapped text, plus a
  skill-fuzzy-match loop that rebuilt its candidate list from scratch for
  every one of ~100 taxonomy skills. On Render's shared CPU this can easily
  take longer than the platform's upstream timeout, and because the FastAPI
  route that calls this module is `async def` but was calling this fully
  synchronous, CPU-bound pipeline directly (not via a thread), it also
  blocked the whole event loop for the duration — see main.py's /analyze
  handler for the corresponding fix (asyncio.to_thread).

  Fixes in this version:
    1. MAX_INPUT_CHARS caps every input once, up front, before anything
       else runs — every downstream stage inherits the cap.
    2. Exactly ONE spaCy Doc is created per document (resume, JD) and
       reused for preprocessing, skill NER, ATS keyword extraction, and
       full NER extraction — down from up to 4 parses per document.
    3. extract_skills() builds its fuzzy-match candidate list (words +
       bigrams) ONCE per document instead of once per taxonomy skill —
       this was the single biggest CPU cost on longer resumes.
    4. SBERT inference runs inside torch.inference_mode() to avoid
       retaining any autograd state.
    5. Stage-by-stage logging so Render logs show exactly where time is
       spent (or where a failure happens) instead of a bare 502/500.
"""

import re
import time
import logging
import spacy
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch ships transitively via sentence-transformers
    _TORCH_AVAILABLE = False

logger = logging.getLogger("resumeiq.pipeline")

# ─── Perf / safety limits ─────────────────────────────────────────────────────
# Resumes and JDs are a few hundred to a couple thousand words in practice.
# Capping at ~6000 chars (roughly 1000-1200 words) is generous for real
# documents but bounds spaCy parse time, fuzzy-match candidate count, and
# memory for pathological inputs (e.g. a badly-parsed PDF dumping 50k chars).
MAX_INPUT_CHARS = 6000

# ─── Lazy model loading ────────────────────────────────────────────────────────
# Models are NOT loaded when FastAPI imports this module.
# This keeps /health lightweight and prevents Render startup OOM.

nlp = None
sbert = None


def get_nlp():
    global nlp
    if nlp is None:
        logger.info("Loading spaCy model (en_core_web_sm)...")
        t0 = time.time()
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded in %.2fs", time.time() - t0)
    return nlp


def get_sbert():
    global sbert
    if sbert is None:
        logger.info("Loading SBERT model (all-MiniLM-L6-v2)...")
        t0 = time.time()
        sbert = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )
        logger.info("SBERT model loaded in %.2fs", time.time() - t0)
    return sbert


# ─── Skill taxonomy ───────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
        "rust", "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "bash",
        "shell", "sql", "html", "css", "dart", "perl"
    ],
    "web_frameworks": [
        "react", "vue", "angular", "next.js", "nuxt", "svelte", "django", "flask",
        "fastapi", "spring", "express", "node.js", "rails", "laravel", "asp.net",
        "fastify", "nestjs", "gatsby", "remix"
    ],
    "ml_ai": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
        "numpy", "spacy", "nltk", "huggingface", "transformers", "bert", "gpt",
        "llm", "langchain", "xgboost", "lightgbm", "catboost", "opencv",
        "machine learning", "deep learning", "nlp", "computer vision", "mlops",
        "feature engineering", "hyperparameter tuning", "model deployment"
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "dynamodb", "sqlite", "oracle", "sql server", "bigquery", "snowflake",
        "databricks", "neo4j", "influxdb"
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "github actions", "ci/cd", "helm", "istio",
        "prometheus", "grafana", "datadog", "cloudformation", "lambda", "s3", "ec2"
    ],
    "soft_skills": [
        "leadership", "communication", "collaboration", "mentoring", "agile",
        "scrum", "project management", "problem solving", "cross-functional",
        "stakeholder management"
    ],
}

ALL_SKILLS = sorted(set(s for skills in SKILL_TAXONOMY.values() for s in skills))

SECTION_PATTERNS = {
    "experience": re.compile(r"(experience|employment|work history|professional background)", re.I),
    "education":  re.compile(r"(education|academic|degree|university|college)", re.I),
    "skills":     re.compile(r"(skills|technologies|tech stack|competencies|expertise)", re.I),
    "projects":   re.compile(r"(projects|portfolio|open.?source|contributions)", re.I),
    "summary":    re.compile(r"(summary|objective|about|profile|overview)", re.I),
}


# ─── 0. INPUT SAFETY ──────────────────────────────────────────────────────────
def truncate_text(text: str, label: str) -> str:
    """Cap input length once, up front. Every downstream stage (spaCy,
    fuzzy skill matching, BM25, SBERT) inherits this cap, which is what
    keeps a single oversized request from ballooning into multi-second
    processing time / memory spikes on Render."""
    if len(text) > MAX_INPUT_CHARS:
        logger.warning(
            "%s truncated from %d to %d chars", label, len(text), MAX_INPUT_CHARS
        )
        return text[:MAX_INPUT_CHARS]
    return text


def clean_text(text: str) -> str:
    """Strip URLs, emails, and phone numbers; collapse whitespace.
    Run ONCE per document, before the single spaCy parse, so junk tokens
    never enter spaCy's NER, TF-IDF, or BM25 vocab downstream."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-\(\)]{8,}\d", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── 1. TEXT PREPROCESSING ────────────────────────────────────────────────────
def preprocess_text(doc) -> str:
    """
    Lemmatize → remove stopwords/punctuation from an ALREADY-PARSED spaCy Doc.
    Returns clean string suitable for TF-IDF and BM25.
    (No longer calls spaCy itself — the Doc is parsed once in analyze_resume()
    and reused across every stage that needs it.)
    """
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct
        and not token.is_space and len(token.text) > 1
    ]
    return " ".join(tokens)


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Light tokenization for BM25 — preserves technical terms better than
    aggressive lemmatization. Splits on whitespace/punctuation, lowercases.
    BM25 works on token lists, not strings.
    """
    text = re.sub(r"[^\w\s\.\+\#]", " ", text.lower())
    tokens = re.findall(r"[\w\.\+\#]+", text)
    return [t for t in tokens if len(t) > 1]


# ─── 2. SECTION SEGMENTATION ──────────────────────────────────────────────────
def parse_sections(text: str) -> Dict[str, str]:
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in lines:
        stripped = line.strip()
        matched = None
        for name, pattern in SECTION_PATTERNS.items():
            if pattern.search(stripped) and len(stripped) < 60:
                matched = name
                break
        if matched:
            current = matched
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(stripped)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


# ─── 3. SKILL EXTRACTION ──────────────────────────────────────────────────────
def _build_ngrams(text_lower: str) -> List[str]:
    """Word + bigram candidate list for fuzzy matching, built ONCE per
    document. Previously this was rebuilt from scratch inside the loop for
    every one of ~100 taxonomy skills — O(skills × words) redone work that
    was the single biggest CPU cost in the pipeline on longer resumes."""
    words = re.findall(r"[\w\.\+\#]+", text_lower)
    bigrams = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
    return list(dict.fromkeys(words + bigrams))  # dedupe, preserve order


def extract_skills(text_lower: str, doc, threshold: int = 85) -> Dict[str, List[str]]:
    """
    Two-strategy extraction:
    A) Fuzzy match against skill taxonomy (handles "Postgres" → "postgresql")
    B) spaCy NER for PRODUCT/ORG entities (catches things taxonomy misses)

    `text_lower` and `doc` should both come from the same already-cleaned,
    already-truncated document text (see analyze_resume()).
    """
    found: Dict[str, List[str]] = {cat: [] for cat in SKILL_TAXONOMY}
    ngrams = None  # built lazily — only if we actually need a fuzzy fallback

    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            if skill in text_lower:
                found[category].append(skill)
                continue
            if ngrams is None:
                ngrams = _build_ngrams(text_lower)
            # score_cutoff lets rapidfuzz's C implementation bail out early
            # on poor candidates instead of scoring every ngram fully.
            match = process.extractOne(
                skill, ngrams, scorer=fuzz.ratio, score_cutoff=threshold
            )
            if match:
                found[category].append(skill)

    ner_skills = [
        ent.text.lower() for ent in doc.ents
        if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART")
    ]
    found["ner_extracted"] = list(set(ner_skills))
    found["all"] = sorted(set(s for cat, skills in found.items() if cat != "all" for s in skills))
    return found


# ─── 4. TF-IDF SIMILARITY ─────────────────────────────────────────────────────
def tfidf_similarity(resume_clean: str, jd_clean: str) -> float:
    """
    Cosine similarity on TF-IDF vectors.
    ngram_range=(1,2) captures "machine learning", "react native" as single features.
    sublinear_tf=True applies log(1+tf) to dampen very frequent terms.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        return float(round(cosine_similarity(matrix[0:1], matrix[1:2])[0][0], 4))
    except Exception:
        logger.exception("tfidf_similarity failed, returning 0.0")
        return 0.0


def get_tfidf_top_terms(resume_clean: str, jd_clean: str, top_n: int = 15) -> Dict:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        names = vectorizer.get_feature_names_out()
        resume_top = [names[i] for i in np.argsort(matrix[0].toarray()[0])[::-1][:top_n]]
        jd_top     = [names[i] for i in np.argsort(matrix[1].toarray()[0])[::-1][:top_n]]
        return {
            "resume_top_terms": resume_top,
            "jd_top_terms":     jd_top,
            "overlap_terms":    list(set(resume_top) & set(jd_top)),
        }
    except Exception:
        logger.exception("get_tfidf_top_terms failed, returning {}")
        return {}


# ─── 5. BM25 SIMILARITY ───────────────────────────────────────────────────────
def bm25_similarity(resume_tokens: List[str], jd_tokens: List[str]) -> float:
    """
    BM25Okapi: probabilistic relevance model.

    How it works:
      corpus = [resume_tokens]   ← treat resume as the "document"
      query  = jd_tokens         ← treat JD as the "query"

    BM25 asks: "How relevant is the resume document to the JD query?"

    Score formula per term t:
      IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))

    Where:
      tf(t,d)  = term frequency of t in document d
      k1=1.5   = saturation: diminishing returns for repeated terms
      b=0.75   = length norm: penalizes very long documents
      avgdl    = average document length in corpus

    We normalize to [0,1] by dividing by the self-score (max possible).
    """
    if not resume_tokens or not jd_tokens:
        return 0.0

    try:
        bm25 = BM25Okapi([resume_tokens], k1=1.5, b=0.75)
        raw_score = bm25.get_scores(jd_tokens)[0]

        bm25_self = BM25Okapi([jd_tokens], k1=1.5, b=0.75)
        max_score = bm25_self.get_scores(jd_tokens)[0]

        if max_score <= 0:
            return 0.0

        normalized = float(raw_score / max_score)
        return round(max(0.0, min(1.0, normalized)), 4)
    except Exception:
        logger.exception("bm25_similarity failed, returning 0.0")
        return 0.0


# ─── 6. SBERT SEMANTIC SIMILARITY ─────────────────────────────────────────────
def semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Dense embedding cosine similarity via Sentence-BERT.
    Captures MEANING — 'built REST APIs' ≈ 'backend service development'.
    normalize_embeddings=True → dot product == cosine similarity (faster).
    Runs inside torch.inference_mode() when torch is importable, so no
    autograd graph / gradient buffers are retained for inference-only calls.
    """
    model = get_sbert()
    texts = [resume_text[:2000], jd_text[:2000]]

    if _TORCH_AVAILABLE:
        with torch.inference_mode():
            embeddings = model.encode(texts, normalize_embeddings=True)
    else:
        embeddings = model.encode(texts, normalize_embeddings=True)

    return round(max(0.0, float(np.dot(embeddings[0], embeddings[1]))), 4)


# ─── 7. SKILL OVERLAP (JACCARD) ───────────────────────────────────────────────
def skill_overlap_analysis(resume_skills: List[str], jd_skills: List[str]) -> Dict:
    """
    Jaccard = |A ∩ B| / |A ∪ B|
    Extended with fuzzy matching so near-matches become "partial" not "missing".
    """
    resume_set, jd_set = set(resume_skills), set(jd_skills)
    matched, missing, partial = [], [], []

    for jd_skill in jd_set:
        if jd_skill in resume_set:
            matched.append(jd_skill)
        else:
            result = process.extractOne(jd_skill, resume_set, scorer=fuzz.token_sort_ratio)
            if result and result[1] >= 80:
                partial.append({"required": jd_skill, "found": result[0], "confidence": result[1]})
            else:
                missing.append(jd_skill)

    union = jd_set | resume_set
    return {
        "jaccard_similarity": round(len(matched) / len(union), 4) if union else 0.0,
        "matched":            sorted(matched),
        "missing":            sorted(missing),
        "partial":            partial,
        "match_rate":         round(len(matched) / len(jd_set), 4) if jd_set else 0.0,
    }


# ─── 8. ATS SCORE ──────────────────────────────────────────────────────────────
def compute_ats_score(jd_doc, resume_lower: str) -> Dict:
    """
    ATS (Applicant Tracking System) Simulation.

    Most ATS systems (Taleo, Workday, Greenhouse) do EXACT or near-exact
    keyword matching. They DO NOT use embeddings or semantic understanding.
    A candidate scoring 90% on SBERT can still fail ATS with 30% keyword match
    because they used different phrasing.

    This function simulates that:
      1. Extract significant keywords from the JD (nouns, proper nouns, noun chunks)
      2. Check each keyword for EXACT substring presence in the resume
      3. ATS score = % of JD keywords found verbatim in resume

    `jd_doc` is the already-parsed spaCy Doc for the JD text (reused, not
    re-parsed here); `resume_lower` is the already-cleaned resume text,
    lowercased.
    """
    jd_keywords = []
    for token in jd_doc:
        if (
            token.pos_ in ("NOUN", "PROPN")
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ):
            jd_keywords.append(token.lemma_.lower())

    noun_chunks = [chunk.text.lower() for chunk in jd_doc.noun_chunks if len(chunk.text) > 3]
    jd_keywords.extend(noun_chunks)
    jd_keywords = list(set(jd_keywords))

    found_keywords = [kw for kw in jd_keywords if kw in resume_lower]
    missing_keywords = [kw for kw in jd_keywords if kw not in resume_lower]

    ats_score = round(len(found_keywords) / len(jd_keywords) * 100, 1) if jd_keywords else 0.0

    if ats_score >= 70:
        risk = "Low"
        risk_note = "Good keyword coverage — likely to pass most ATS filters."
    elif ats_score >= 45:
        risk = "Medium"
        risk_note = "Some important keywords missing — may be filtered by strict ATS."
    else:
        risk = "High"
        risk_note = "Many JD keywords absent — likely to fail ATS before human review."

    return {
        "ats_score":          ats_score,
        "risk_level":         risk,
        "risk_note":          risk_note,
        "total_jd_keywords":  len(jd_keywords),
        "found_keywords":     sorted(found_keywords)[:20],
        "missing_keywords":   sorted(missing_keywords)[:20],
        "pass_probability":   f"{min(95, max(5, int(ats_score * 0.9 + 5)))}%",
    }


# ─── 9. NER EXTRACTION ────────────────────────────────────────────────────────
def extract_named_entities(doc, text: str) -> Dict:
    """`doc` is the already-parsed spaCy Doc; `text` is the same
    already-cleaned document text, used only for the years-of-experience
    regex (which doesn't need spaCy)."""
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text.strip() not in entities[ent.label_]:
            entities[ent.label_].append(ent.text.strip())

    years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)", text, re.I)
    return {
        "entities":                    entities,
        "years_experience_mentioned":  sorted(set(int(y) for y in years), reverse=True),
        "organizations":               entities.get("ORG", []),
        "locations":                   entities.get("GPE", []),
        "dates":                       entities.get("DATE", []),
        "technologies":                entities.get("PRODUCT", []),
    }


# ─── 10. COMPOSITE SCORING ENGINE ─────────────────────────────────────────────
def composite_score(
    tfidf_sim: float,
    semantic_sim: float,
    bm25_sim: float,
    skill_match_rate: float,
    jaccard: float,
) -> Dict:
    """
    Weighted ensemble across 5 signals.

    Component           Weight  Rationale
    ──────────────────  ──────  ─────────────────────────────────────────────
    Semantic (SBERT)     0.30   Meaning-level match, most sophisticated signal
    Skill match rate     0.25   Direct requirement coverage, most job-relevant
    BM25                 0.20   Better than TF-IDF for short docs
    TF-IDF cosine        0.15   Keyword/terminology alignment
    Jaccard overlap      0.10   Raw vocabulary overlap sanity check

    Weights sum to 1.0 exactly.
    """
    weights = {
        "semantic":    0.30,
        "skill_match": 0.25,
        "bm25":        0.20,
        "tfidf":       0.15,
        "jaccard":     0.10,
    }

    raw = (
        semantic_sim        * weights["semantic"]
        + skill_match_rate  * weights["skill_match"]
        + bm25_sim          * weights["bm25"]
        + tfidf_sim         * weights["tfidf"]
        + jaccard           * weights["jaccard"]
    )

    return {
        "final_score": max(0, min(100, int(round(raw * 100)))),
        "component_scores": {
            "semantic_sbert":   round(semantic_sim * 100, 1),
            "skill_match_rate": round(skill_match_rate * 100, 1),
            "bm25":             round(bm25_sim * 100, 1),
            "tfidf_cosine":     round(tfidf_sim * 100, 1),
            "jaccard_overlap":  round(jaccard * 100, 1),
        },
        "weights": weights,
    }


# ─── 11. MASTER ANALYSIS FUNCTION ─────────────────────────────────────────────
def analyze_resume(resume_text: str, jd_text: str) -> Dict:
    """
    Full pipeline entry point. Same signature and output shape as before —
    only the internals changed (see PERFORMANCE NOTES at the top of this file).
    """
    t_start = time.time()
    logger.info(
        "analyze_resume: start (resume=%d chars, jd=%d chars)",
        len(resume_text), len(jd_text)
    )

    try:
        # 0. Cap input length once, up front.
        resume_text = truncate_text(resume_text, "resume")
        jd_text = truncate_text(jd_text, "job_description")

        # 1. Sections — cheap, run on the truncated (not yet URL/email-stripped)
        #    text so section headers aren't disturbed by whitespace collapsing.
        resume_sections = parse_sections(resume_text)
        jd_sections = parse_sections(jd_text)

        # Clean once; every downstream stage (spaCy, TF-IDF, BM25) shares this.
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)

        # 2. ONE spaCy parse per document, reused everywhere below.
        nlp_model = get_nlp()
        t0 = time.time()
        resume_doc = nlp_model(resume_clean)
        jd_doc = nlp_model(jd_clean)
        logger.info("stage: spaCy parse done in %.2fs", time.time() - t0)

        # 3. Lemmatized text for TF-IDF, + BM25 tokens
        resume_lemmas = preprocess_text(resume_doc)
        jd_lemmas = preprocess_text(jd_doc)
        resume_tokens = tokenize_for_bm25(resume_clean)
        jd_tokens = tokenize_for_bm25(jd_clean)
        logger.info("stage: preprocessing/tokenization done")

        # 4. Skills
        resume_lower = resume_clean.lower()
        jd_lower = jd_clean.lower()
        t0 = time.time()
        resume_skills_data = extract_skills(resume_lower, resume_doc)
        jd_skills_data = extract_skills(jd_lower, jd_doc)
        logger.info("stage: skill extraction done in %.2fs", time.time() - t0)

        # 5. TF-IDF
        tfidf_sim = tfidf_similarity(resume_lemmas, jd_lemmas)
        tfidf_terms = get_tfidf_top_terms(resume_lemmas, jd_lemmas)

        # 6. BM25
        bm25_sim = bm25_similarity(resume_tokens, jd_tokens)

        # 7. SBERT
        t0 = time.time()
        sem_sim = semantic_similarity(resume_clean, jd_clean)
        logger.info("stage: SBERT similarity done in %.2fs", time.time() - t0)

        # 8. Skill overlap
        skill_analysis = skill_overlap_analysis(
            resume_skills_data["all"],
            jd_skills_data["all"],
        )

        # 9. ATS score (reuses jd_doc — no re-parse)
        ats = compute_ats_score(jd_doc, resume_lower)

        # 10. NER (reuses resume_doc / jd_doc — no re-parse)
        resume_ner = extract_named_entities(resume_doc, resume_clean)
        jd_ner = extract_named_entities(jd_doc, jd_clean)

        # 11. Composite score
        scoring = composite_score(
            tfidf_sim=tfidf_sim,
            semantic_sim=sem_sim,
            bm25_sim=bm25_sim,
            skill_match_rate=skill_analysis["match_rate"],
            jaccard=skill_analysis["jaccard_similarity"],
        )

        # 12. Assemble
        result = {
            "score":            scoring["final_score"],
            "scoring_breakdown": scoring,
            "skill_analysis": {
                "matched_skills":       skill_analysis["matched"],
                "missing_skills":       skill_analysis["missing"],
                "partial_match_skills": [p["required"] for p in skill_analysis["partial"]],
                "partial_details":      skill_analysis["partial"],
                "resume_skills":        resume_skills_data["all"],
                "jd_skills":            jd_skills_data["all"],
                "skills_by_category": {
                    "resume": {k: v for k, v in resume_skills_data.items() if k not in ("all", "ner_extracted") and v},
                    "jd":     {k: v for k, v in jd_skills_data.items()     if k not in ("all", "ner_extracted") and v},
                },
            },
            "similarity_scores": {
                "tfidf_cosine":     round(tfidf_sim * 100, 1),
                "bm25":             round(bm25_sim * 100, 1),
                "semantic_sbert":   round(sem_sim * 100, 1),
                "skill_jaccard":    round(skill_analysis["jaccard_similarity"] * 100, 1),
                "skill_match_rate": round(skill_analysis["match_rate"] * 100, 1),
            },
            "ats_analysis":  ats,
            "tfidf_analysis": tfidf_terms,
            "named_entities": {
                "resume": resume_ner,
                "jd":     jd_ner,
            },
            "sections_found": {
                "resume": list(resume_sections.keys()),
                "jd":     list(jd_sections.keys()),
            },
            "text_stats": {
                "resume_word_count": len(resume_text.split()),
                "jd_word_count":     len(jd_text.split()),
                "resume_vocab_size": len(set(resume_lemmas.split())),
                "jd_vocab_size":     len(set(jd_lemmas.split())),
            },
        }

        logger.info("analyze_resume: done in %.2fs", time.time() - t_start)
        return result

    except Exception:
        logger.exception("analyze_resume: failed after %.2fs", time.time() - t_start)
        raise


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    r = analyze_resume(
        "Python developer, 5 years. React, FastAPI, PostgreSQL, Docker, Kubernetes, AWS.",
        "Need senior engineer: Python, TypeScript, React, GraphQL, PostgreSQL, Kubernetes, Redis."
    )
    print(json.dumps(r, indent=2))
