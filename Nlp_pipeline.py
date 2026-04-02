"""
Nlp_pipeline.py — Real ML/NLP Resume Screening Pipeline

Models & algorithms used:
  - spaCy (en_core_web_md)         → NER, POS tagging, lemmatization
  - scikit-learn TfidfVectorizer   → TF-IDF document vectors
  - sentence-transformers (SBERT)  → Dense semantic embeddings
  - rank_bm25 (BM25Okapi)          → Probabilistic term weighting (NEW)
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
"""

import re
import json
import spacy
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# BM25 — NEW
from rank_bm25 import BM25Okapi

# ─── Load models once at module level ────────────────────────────────────────
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")

print("Loading SBERT model...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")


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


# ─── 1. TEXT PREPROCESSING ───────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Tokenize → lowercase → lemmatize → remove stopwords/punctuation.
    Returns clean string suitable for TF-IDF and BM25.
    """
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-\(\)]{8,}\d", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
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


# ─── 2. SECTION SEGMENTATION ─────────────────────────────────────────────────
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


# ─── 3. SKILL EXTRACTION ─────────────────────────────────────────────────────
def extract_skills(text: str, threshold: int = 85) -> Dict[str, List[str]]:
    """
    Two-strategy extraction:
    A) Fuzzy match against skill taxonomy (handles "Postgres" → "postgresql")
    B) spaCy NER for PRODUCT/ORG entities (catches things taxonomy misses)
    """
    text_lower = text.lower()
    found: Dict[str, List[str]] = {cat: [] for cat in SKILL_TAXONOMY}

    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            if skill in text_lower:
                found[category].append(skill)
            else:
                words = re.findall(r"[\w\.\+\#]+", text_lower)
                ngrams = words + [" ".join(words[i:i+2]) for i in range(len(words)-1)]
                match = process.extractOne(skill, ngrams, scorer=fuzz.ratio)
                if match and match[1] >= threshold:
                    found[category].append(skill)

    doc = nlp(text[:5000])
    ner_skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART")]
    found["ner_extracted"] = list(set(ner_skills))
    found["all"] = sorted(set(s for cat, skills in found.items() if cat != "all" for s in skills))
    return found


# ─── 4. TF-IDF SIMILARITY ────────────────────────────────────────────────────
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
        return {}


# ─── 5. BM25 SIMILARITY (NEW) ─────────────────────────────────────────────────
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
        # BM25 treats resume as the searchable corpus
        bm25 = BM25Okapi([resume_tokens], k1=1.5, b=0.75)

        # Score: how well does the JD query match the resume?
        raw_score = bm25.get_scores(jd_tokens)[0]

        # Normalize: score the doc against itself to get the theoretical max
        bm25_self = BM25Okapi([jd_tokens], k1=1.5, b=0.75)
        max_score = bm25_self.get_scores(jd_tokens)[0]

        if max_score <= 0:
            return 0.0

        normalized = float(raw_score / max_score)
        return round(max(0.0, min(1.0, normalized)), 4)
    except Exception:
        return 0.0


# ─── 6. SBERT SEMANTIC SIMILARITY ────────────────────────────────────────────
def semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Dense embedding cosine similarity via Sentence-BERT.
    Captures MEANING — 'built REST APIs' ≈ 'backend service development'.
    normalize_embeddings=True → dot product == cosine similarity (faster).
    """
    embeddings = sbert.encode(
        [resume_text[:2000], jd_text[:2000]],
        normalize_embeddings=True
    )
    return round(max(0.0, float(np.dot(embeddings[0], embeddings[1]))), 4)


# ─── 7. SKILL OVERLAP (JACCARD) ──────────────────────────────────────────────
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


# ─── 8. ATS SCORE (NEW) ───────────────────────────────────────────────────────
def compute_ats_score(resume_text: str, jd_text: str) -> Dict:
    """
    ATS (Applicant Tracking System) Simulation.

    Most ATS systems (Taleo, Workday, Greenhouse) do EXACT or near-exact
    keyword matching. They DO NOT use embeddings or semantic understanding.
    A candidate scoring 90% on SBERT can still fail ATS with 30% keyword match
    because they used different phrasing.

    This function simulates that:
      1. Extract significant keywords from JD (nouns, proper nouns, tech terms)
      2. Check each keyword for EXACT substring presence in the resume
      3. ATS score = % of JD keywords found verbatim in resume

    We also identify "shadow keywords" — terms the JD emphasizes that the
    candidate never uses (e.g., JD says "REST API", resume says "web services").
    """
    resume_lower = resume_text.lower()

    # Extract JD keywords using spaCy POS filtering
    doc = nlp(jd_text[:4000])
    jd_keywords = []
    for token in doc:
        # Keep: nouns, proper nouns, adjectives that are likely technical
        if (
            token.pos_ in ("NOUN", "PROPN")
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ):
            jd_keywords.append(token.lemma_.lower())

    # Also extract multi-word technical phrases (noun chunks)
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
    jd_keywords.extend(noun_chunks)
    jd_keywords = list(set(jd_keywords))

    # Check verbatim presence in resume
    found_keywords = [kw for kw in jd_keywords if kw in resume_lower]
    missing_keywords = [kw for kw in jd_keywords if kw not in resume_lower]

    ats_score = round(len(found_keywords) / len(jd_keywords) * 100, 1) if jd_keywords else 0.0

    # ATS risk level
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


# ─── 9. NER EXTRACTION ───────────────────────────────────────────────────────
def extract_named_entities(text: str) -> Dict:
    doc = nlp(text[:8000])
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


# ─── 10. COMPOSITE SCORING ENGINE (UPDATED) ───────────────────────────────────
def composite_score(
    tfidf_sim: float,
    semantic_sim: float,
    bm25_sim: float,
    skill_match_rate: float,
    jaccard: float,
) -> Dict:
    """
    Updated weighted ensemble — now 5 signals including BM25.

    Component           Weight  Rationale
    ──────────────────  ──────  ─────────────────────────────────────────────
    Semantic (SBERT)     0.30   Meaning-level match, most sophisticated signal
    Skill match rate     0.25   Direct requirement coverage, most job-relevant
    BM25                 0.20   Better than TF-IDF for short docs (NEW)
    TF-IDF cosine        0.15   Keyword/terminology alignment
    Jaccard overlap      0.10   Raw vocabulary overlap sanity check

    BM25 replaces some TF-IDF weight because it handles term saturation better.
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
        semantic_sim      * weights["semantic"]
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


# ─── 11. MASTER ANALYSIS FUNCTION ────────────────────────────────────────────
def analyze_resume(resume_text: str, jd_text: str) -> Dict:
    """
    Full pipeline entry point.

    Steps:
      1.  Section parsing
      2.  Text preprocessing (lemmatize, clean)
      3.  BM25 tokenization (light tokenize, preserves tech terms)
      4.  Skill extraction (taxonomy + NER)
      5.  TF-IDF cosine similarity
      6.  BM25 similarity  ← NEW
      7.  SBERT semantic similarity
      8.  Skill overlap + Jaccard
      9.  ATS score simulation  ← NEW
      10. NER entity extraction
      11. Composite weighted score (5 signals)
      12. Assemble final report
    """
    # 1. Sections
    resume_sections = parse_sections(resume_text)
    jd_sections     = parse_sections(jd_text)

    # 2. Preprocess
    resume_clean = preprocess_text(resume_text)
    jd_clean     = preprocess_text(jd_text)

    # 3. BM25 tokenization
    resume_tokens = tokenize_for_bm25(resume_text)
    jd_tokens     = tokenize_for_bm25(jd_text)

    # 4. Skills
    resume_skills_data = extract_skills(resume_text)
    jd_skills_data     = extract_skills(jd_text)

    # 5. TF-IDF
    tfidf_sim   = tfidf_similarity(resume_clean, jd_clean)
    tfidf_terms = get_tfidf_top_terms(resume_clean, jd_clean)

    # 6. BM25
    bm25_sim = bm25_similarity(resume_tokens, jd_tokens)

    # 7. SBERT
    sem_sim = semantic_similarity(resume_text, jd_text)

    # 8. Skill overlap
    skill_analysis = skill_overlap_analysis(
        resume_skills_data["all"],
        jd_skills_data["all"],
    )

    # 9. ATS score
    ats = compute_ats_score(resume_text, jd_text)

    # 10. NER
    resume_ner = extract_named_entities(resume_text)
    jd_ner     = extract_named_entities(jd_text)

    # 11. Composite score
    scoring = composite_score(
        tfidf_sim=tfidf_sim,
        semantic_sim=sem_sim,
        bm25_sim=bm25_sim,
        skill_match_rate=skill_analysis["match_rate"],
        jaccard=skill_analysis["jaccard_similarity"],
    )

    # 12. Assemble
    return {
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
            "resume_vocab_size": len(set(resume_clean.split())),
            "jd_vocab_size":     len(set(jd_clean.split())),
        },
    }


if __name__ == "__main__":
    import json
    r = analyze_resume(
        "Python developer, 5 years. React, FastAPI, PostgreSQL, Docker, Kubernetes, AWS.",
        "Need senior engineer: Python, TypeScript, React, GraphQL, PostgreSQL, Kubernetes, Redis."
    )
    print(json.dumps(r, indent=2))
