"""
Nlp_pipeline.py — Real ML/NLP Resume Screening Pipeline

CRITICAL: This file must NOT import spaCy or SentenceTransformer at module level.
All model access goes through model_loader.get_nlp() / model_loader.get_sbert()
which implement lazy singleton loading.
"""

import os
import re
import warnings
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*position_ids.*")

# Lazy accessors — models load on first call, cached forever after
from model_loader import get_nlp, get_sbert


# ── Skill taxonomy ─────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
        "rust", "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "bash",
        "shell", "sql", "html", "css", "dart", "perl",
    ],
    "web_frameworks": [
        "react", "vue", "angular", "next.js", "nuxt", "svelte", "django", "flask",
        "fastapi", "spring", "express", "node.js", "rails", "laravel", "asp.net",
        "fastify", "nestjs", "gatsby", "remix",
    ],
    "ml_ai": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
        "numpy", "spacy", "nltk", "huggingface", "transformers", "bert", "gpt",
        "llm", "langchain", "xgboost", "lightgbm", "catboost", "opencv",
        "machine learning", "deep learning", "nlp", "computer vision", "mlops",
        "feature engineering", "hyperparameter tuning",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "dynamodb", "sqlite", "oracle", "sql server", "bigquery", "snowflake",
        "databricks", "neo4j", "influxdb",
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "github actions", "ci/cd", "helm", "istio",
        "prometheus", "grafana", "datadog", "cloudformation",
        "lambda", "s3", "ec2", "ecs", "fargate", "cloud run",
    ],
    "soft_skills": [
        "leadership", "communication", "collaboration", "mentoring", "agile",
        "scrum", "project management", "problem solving", "cross-functional",
        "stakeholder management", "presentation",
    ],
}

SECTION_PATTERNS = {
    "experience": re.compile(r"(experience|employment|work history|professional background)", re.I),
    "education":  re.compile(r"(education|academic|degree|university|college|qualification)", re.I),
    "skills":     re.compile(r"(skills|technologies|tech stack|competencies|expertise)", re.I),
    "projects":   re.compile(r"(projects|portfolio|open.?source|contributions)", re.I),
    "summary":    re.compile(r"(summary|objective|about|profile|overview)", re.I),
}


# ── 1. Text preprocessing ──────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-\(\)]{8,}\d", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    nlp  = get_nlp()
    doc  = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
        and not token.is_space and len(token.text) > 1
    ]
    return " ".join(tokens)


def tokenize_for_bm25(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


# ── 2. Section segmentation ────────────────────────────────────────────────────
def parse_sections(text: str) -> Dict[str, str]:
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in lines:
        stripped = line.strip()
        matched = None
        for name, pat in SECTION_PATTERNS.items():
            if pat.search(stripped) and len(stripped) < 60:
                matched = name
                break
        if matched:
            current = matched
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(stripped)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


# ── 3. Skill extraction ────────────────────────────────────────────────────────
def extract_skills(text: str, threshold: int = 85) -> Dict[str, List[str]]:
    text_lower = text.lower()
    found: Dict[str, List[str]] = {cat: [] for cat in SKILL_TAXONOMY}
    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            if skill in text_lower:
                found[category].append(skill)
            else:
                words  = re.findall(r"[\w\.\+\#]+", text_lower)
                ngrams = words + [" ".join(words[i:i+2]) for i in range(len(words)-1)]
                match  = process.extractOne(skill, ngrams, scorer=fuzz.ratio)
                if match and match[1] >= threshold:
                    found[category].append(skill)
    nlp = get_nlp()
    doc = nlp(text[:5000])
    ner_skills = [
        ent.text.lower() for ent in doc.ents
        if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART")
    ]
    found["ner_extracted"] = list(set(ner_skills))
    found["all"] = sorted(set(
        s for cat, skills in found.items() if cat != "all" for s in skills
    ))
    return found


# ── 4. TF-IDF ──────────────────────────────────────────────────────────────────
def tfidf_similarity(resume_clean: str, jd_clean: str) -> float:
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
        names  = vectorizer.get_feature_names_out()
        r_top  = [names[i] for i in np.argsort(matrix[0].toarray()[0])[::-1][:top_n]]
        j_top  = [names[i] for i in np.argsort(matrix[1].toarray()[0])[::-1][:top_n]]
        return {"resume_top_terms": r_top, "jd_top_terms": j_top, "overlap_terms": list(set(r_top) & set(j_top))}
    except Exception:
        return {}


# ── 5. BM25 ────────────────────────────────────────────────────────────────────
def bm25_similarity(resume_text: str, jd_text: str) -> float:
    r_tokens  = tokenize_for_bm25(resume_text)
    j_tokens  = tokenize_for_bm25(jd_text)
    if not r_tokens or not j_tokens:
        return 0.0
    bm25      = BM25Okapi([r_tokens])
    raw_score = bm25.get_scores(j_tokens)[0]
    max_score = BM25Okapi([r_tokens]).get_scores(r_tokens)[0]
    if max_score <= 0:
        return 0.0
    return round(min(float(raw_score / max_score), 1.0), 4)


# ── 6. SBERT semantic similarity ───────────────────────────────────────────────
def semantic_similarity(resume_text: str, jd_text: str) -> float:
    sbert      = get_sbert()
    embeddings = sbert.encode(
        [resume_text[:2000], jd_text[:2000]],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    score = float(np.dot(embeddings[0], embeddings[1]))
    del embeddings   # release numpy array immediately
    return round(max(0.0, score), 4)


# ── 7. Skill overlap + Jaccard ─────────────────────────────────────────────────
def skill_overlap_analysis(resume_skills: List[str], jd_skills: List[str]) -> Dict:
    resume_set, jd_set = set(resume_skills), set(jd_skills)
    matched, missing, partial = [], [], []
    for skill in jd_set:
        if skill in resume_set:
            matched.append(skill)
        else:
            result = process.extractOne(skill, resume_set, scorer=fuzz.token_sort_ratio)
            if result and result[1] >= 80:
                partial.append({"required": skill, "found": result[0], "confidence": result[1]})
            else:
                missing.append(skill)
    union = jd_set | resume_set
    return {
        "jaccard_similarity": round(len(matched) / len(union), 4) if union else 0.0,
        "matched":            sorted(matched),
        "missing":            sorted(missing),
        "partial":            partial,
        "match_rate":         round(len(matched) / len(jd_set), 4) if jd_set else 0.0,
    }


# ── 8. ATS score ───────────────────────────────────────────────────────────────
def compute_ats_score(resume_text: str, jd_text: str) -> Dict:
    resume_lower = resume_text.lower()
    nlp          = get_nlp()
    jd_doc       = nlp(jd_text.lower()[:3000])
    jd_keywords  = set(
        token.lemma_ for token in jd_doc
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 2
    )
    jd_keywords.update(extract_skills(jd_text)["all"])
    found_keywords   = [kw for kw in jd_keywords if kw in resume_lower]
    missing_keywords = sorted(set(jd_keywords) - set(found_keywords))
    keyword_rate     = len(found_keywords) / len(jd_keywords) if jd_keywords else 0
    sections_found   = parse_sections(resume_text)
    section_score    = sum(1 for s in ["experience", "education", "skills"] if s in sections_found) / 3
    formatting_checks = {
        "has_email":         bool(re.search(r"\S+@\S+\.\S+", resume_text)),
        "has_dates":         bool(re.search(r"\b(20\d{2}|19\d{2})\b", resume_text)),
        "has_phone":         bool(re.search(r"\+?\d[\d\s\-]{8,}\d", resume_text)),
        "reasonable_length": 200 <= len(resume_text.split()) <= 1200,
        "no_tables":         not bool(re.search(r"\|.*\|", resume_text)),
    }
    formatting_score = sum(formatting_checks.values()) / len(formatting_checks)
    ats_score = max(0, min(100, int(round(
        (keyword_rate * 0.60 + section_score * 0.25 + formatting_score * 0.15) * 100
    ))))
    return {
        "ats_score":          ats_score,
        "keyword_match_rate": round(keyword_rate * 100, 1),
        "keywords_found":     sorted(found_keywords)[:20],
        "keywords_missing":   missing_keywords[:20],
        "sections_found":     list(sections_found.keys()),
        "formatting_checks":  formatting_checks,
        "ats_verdict": (
            "Likely to pass ATS"       if ats_score >= 65 else
            "May be filtered"          if ats_score >= 45 else
            "High risk of ATS rejection"
        ),
    }


# ── 9. NER extraction ──────────────────────────────────────────────────────────
def extract_named_entities(text: str) -> Dict:
    nlp = get_nlp()
    doc = nlp(text[:8000])
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text.strip() not in entities[ent.label_]:
            entities[ent.label_].append(ent.text.strip())
    years = []
    for pat in [r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)",
                r"(\d+)\+?\s*yrs?\s*(?:of\s+)?(?:experience|exp)"]:
        years.extend(int(m) for m in re.findall(pat, text, re.I))
    return {
        "entities":                   entities,
        "years_experience_mentioned": sorted(set(years), reverse=True),
        "organizations":              entities.get("ORG", []),
        "locations":                  entities.get("GPE", []),
        "dates":                      entities.get("DATE", []),
        "technologies":               entities.get("PRODUCT", []),
    }


# ── 10. Composite score ────────────────────────────────────────────────────────
def composite_score(tfidf_sim, semantic_sim, bm25_sim, skill_match_rate, jaccard) -> Dict:
    weights = {"semantic": 0.30, "skill_match": 0.28, "bm25": 0.20, "tfidf": 0.12, "jaccard": 0.10}
    raw = (
        semantic_sim     * weights["semantic"]
        + skill_match_rate * weights["skill_match"]
        + bm25_sim         * weights["bm25"]
        + tfidf_sim        * weights["tfidf"]
        + jaccard          * weights["jaccard"]
    )
    return {
        "final_score": max(0, min(100, int(round(raw * 100)))),
        "component_scores": {
            "semantic_similarity": round(semantic_sim * 100, 1),
            "skill_match_rate":    round(skill_match_rate * 100, 1),
            "bm25_similarity":     round(bm25_sim * 100, 1),
            "tfidf_cosine":        round(tfidf_sim * 100, 1),
            "jaccard_overlap":     round(jaccard * 100, 1),
        },
        "weights": weights,
    }


# ── 11. Master pipeline ────────────────────────────────────────────────────────
def analyze_resume(resume_text: str, jd_text: str) -> Dict:
    resume_sections = parse_sections(resume_text)
    resume_clean    = preprocess_text(resume_text)
    jd_clean        = preprocess_text(jd_text)
    resume_skills   = extract_skills(resume_text)
    jd_skills       = extract_skills(jd_text)
    tfidf_sim       = tfidf_similarity(resume_clean, jd_clean)
    tfidf_terms     = get_tfidf_top_terms(resume_clean, jd_clean)
    bm25_sim        = bm25_similarity(resume_text, jd_text)
    sem_sim         = semantic_similarity(resume_text, jd_text)
    skill_analysis  = skill_overlap_analysis(resume_skills["all"], jd_skills["all"])
    resume_ner      = extract_named_entities(resume_text)
    ats             = compute_ats_score(resume_text, jd_text)
    scoring = composite_score(
        tfidf_sim=tfidf_sim, semantic_sim=sem_sim, bm25_sim=bm25_sim,
        skill_match_rate=skill_analysis["match_rate"],
        jaccard=skill_analysis["jaccard_similarity"],
    )
    return {
        "score":             scoring["final_score"],
        "scoring_breakdown": scoring,
        "ats_analysis":      ats,
        "skill_analysis": {
            "matched_skills":       skill_analysis["matched"],
            "missing_skills":       skill_analysis["missing"],
            "partial_match_skills": [p["required"] for p in skill_analysis["partial"]],
            "partial_details":      skill_analysis["partial"],
            "resume_skills":        resume_skills["all"],
            "jd_skills":            jd_skills["all"],
            "skills_by_category": {
                "resume": {k: v for k, v in resume_skills.items() if k not in ("all","ner_extracted") and v},
                "jd":     {k: v for k, v in jd_skills.items()     if k not in ("all","ner_extracted") and v},
            },
        },
        "similarity_scores": {
            "tfidf_cosine":     round(tfidf_sim * 100, 1),
            "bm25":             round(bm25_sim * 100, 1),
            "semantic_sbert":   round(sem_sim * 100, 1),
            "skill_jaccard":    round(skill_analysis["jaccard_similarity"] * 100, 1),
            "skill_match_rate": round(skill_analysis["match_rate"] * 100, 1),
        },
        "tfidf_analysis": tfidf_terms,
        "named_entities": {"resume": resume_ner},
        "sections_found": {"resume": list(resume_sections.keys())},
        "text_stats": {
            "resume_word_count": len(resume_text.split()),
            "jd_word_count":     len(jd_text.split()),
            "resume_vocab_size": len(set(resume_clean.split())),
            "jd_vocab_size":     len(set(jd_clean.split())),
        },
    }
