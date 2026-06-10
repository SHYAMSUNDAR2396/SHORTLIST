"""Constant tables and lookup helpers for the Candidate Ranking System.

This module centralizes every static table the scorers rely on: skill-group
synonym mappings, proficiency factors, consulting-firm and keyword sets,
domain category sets, product company-size set, education tier/degree/field
mappings, and the composite scoring weights.

All string tables are stored in lowercase so that lookups can be performed
case-insensitively. Helper functions normalize their inputs to lowercase
before matching.

References (design "Data Models" section and requirements):
- SKILL_GROUPS / MUST_HAVE_GROUPS / NICE_TO_HAVE_GROUPS / skill_to_group (Req 3.1, 3.4)
- PROFICIENCY_FACTOR (Req 3.2)
- CONSULTING_FIRMS (Req 4.2, 6.4)
- PRODUCT_COMPANY_SIZES (Req 4.1)
- PRODUCTION_KEYWORDS (Req 4.4)
- RESEARCH_KEYWORDS (Req 4.5)
- CV_SPEECH_ROBOTICS_CATEGORIES / NLP_IR_CATEGORIES (Req 6.2)
- EDUCATION_TIER_VALUES / DEGREE_LEVEL_VALUES / AI_ML_FIELDS (Req 8.6)
- WEIGHTS (Req 8.1)
"""

from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Skill groups (Req 3.1, 3.4)
# ---------------------------------------------------------------------------
# Each canonical group maps to a list of lowercase variant strings. Related
# technologies belong to the same canonical group so that, e.g., Pinecone and
# Milvus both resolve to "vector_databases".
SKILL_GROUPS: Dict[str, List[str]] = {
    # Must-have (weight=2)
    "embeddings_retrieval": [
        "embeddings", "retrieval", "retrieval systems", "semantic search",
        "sentence transformers", "dense retrieval", "bm25", "hybrid search",
    ],
    "vector_databases": [
        "pinecone", "milvus", "weaviate", "chromadb", "faiss",
        "qdrant", "vector database", "vector db", "vector search",
    ],
    "python": ["python", "pytorch", "tensorflow", "numpy", "pandas"],
    "evaluation_frameworks": [
        "ndcg", "mrr", "map", "evaluation", "ranking metrics",
        "precision", "recall", "f1",
    ],
    # Nice-to-have (weight=1)
    "llm_finetuning": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning",
        "finetuning", "llm fine-tuning", "adapter tuning",
    ],
    "learning_to_rank": [
        "learning to rank", "l2r", "lambdamart", "listnet", "ranknet",
    ],
    "hr_tech": ["hr tech", "hr-tech", "recruitment", "talent", "ats"],
    "distributed_systems": [
        "distributed systems", "kafka", "spark", "ray", "dask",
        "distributed computing", "microservices",
    ],
}

# Must-have groups carry a category weight of 2; nice-to-have groups carry 1.
MUST_HAVE_GROUPS: Set[str] = {
    "embeddings_retrieval", "vector_databases", "python", "evaluation_frameworks",
}
NICE_TO_HAVE_GROUPS: Set[str] = {
    "llm_finetuning", "learning_to_rank", "hr_tech", "distributed_systems",
}

# Category weights derived from the must-have / nice-to-have classification.
MUST_HAVE_WEIGHT: int = 2
NICE_TO_HAVE_WEIGHT: int = 1


def _build_reverse_lookup() -> Dict[str, str]:
    """Precompute a flat variant -> canonical group lookup (all lowercase)."""
    lookup: Dict[str, str] = {}
    for group, variants in SKILL_GROUPS.items():
        for variant in variants:
            lookup[variant.lower()] = group
    return lookup


# Precomputed reverse lookup so skill_to_group runs in (near) constant time.
_SKILL_VARIANT_TO_GROUP: Dict[str, str] = _build_reverse_lookup()


def group_weight(group: Optional[str]) -> int:
    """Return the category weight (2 must-have, 1 nice-to-have, 0 otherwise)."""
    if group in MUST_HAVE_GROUPS:
        return MUST_HAVE_WEIGHT
    if group in NICE_TO_HAVE_GROUPS:
        return NICE_TO_HAVE_WEIGHT
    return 0


def skill_to_group(skill_name: str) -> Optional[str]:
    """Map a skill name to its canonical skill group.

    The match is case-insensitive. An exact match against a known variant is
    preferred; failing that, a substring match is attempted in both directions
    (variant contained in the skill name, or skill name contained in a
    variant) so that names like "Pinecone Vector DB" still resolve.

    Returns the canonical group name, or ``None`` when no group matches.
    """
    if not skill_name:
        return None

    normalized = skill_name.strip().lower()
    if not normalized:
        return None

    # 1. Exact match (fast path).
    group = _SKILL_VARIANT_TO_GROUP.get(normalized)
    if group is not None:
        return group

    # 2. Substring-aware match. Iterate longest variants first so that more
    #    specific variants win (e.g., "vector database" before "vector db").
    best_group: Optional[str] = None
    best_len = 0
    for variant, variant_group in _SKILL_VARIANT_TO_GROUP.items():
        if variant in normalized or normalized in variant:
            if len(variant) > best_len:
                best_len = len(variant)
                best_group = variant_group
    return best_group


# ---------------------------------------------------------------------------
# Proficiency factor (Req 3.2)
# ---------------------------------------------------------------------------
PROFICIENCY_FACTOR: Dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.5,
    "beginner": 0.25,
}

# ---------------------------------------------------------------------------
# Consulting firms (Req 4.2, 6.4)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS: Set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
}

# ---------------------------------------------------------------------------
# Product company sizes (Req 4.1)
# ---------------------------------------------------------------------------
PRODUCT_COMPANY_SIZES: Set[str] = {"11-50", "51-200", "201-500", "501-1000"}

# Company size that, in isolation with no production keywords, caps the
# production_experience_score at 0.2 (Req 4.7).
LARGE_ENTERPRISE_SIZE: str = "10001+"

# ---------------------------------------------------------------------------
# Career-history keyword sets (Req 4.4, 4.5)
# ---------------------------------------------------------------------------
PRODUCTION_KEYWORDS: List[str] = [
    "embeddings", "vector search", "retrieval", "ranking",
    "recommendation", "inference pipeline", "model serving",
]

RESEARCH_KEYWORDS: List[str] = [
    "published", "paper", "conference", "journal",
    "thesis", "theoretical", "research lab",
]

# Title-relevance terms (Req 4.6): titles containing any of these map to 1.0.
AI_ML_TITLE_TERMS: List[str] = [
    "ai", "ml", "machine learning", "data science", "nlp",
]

# Terms that mark a title as having at least some technical relevance (Req 4.6):
# a title with none of these is treated as non-technical (weight 0.0).
TECHNICAL_TITLE_TERMS: List[str] = [
    "technology", "tech", "engineering", "engineer",
    "data", "science", "scientist", "analytics", "analyst",
]

# ---------------------------------------------------------------------------
# Domain category sets for disqualification (Req 6.2)
# ---------------------------------------------------------------------------
CV_SPEECH_ROBOTICS_CATEGORIES: Set[str] = {
    "computer vision", "cv", "image classification", "object detection",
    "image processing", "image segmentation", "ocr", "gans", "gan",
    "speech recognition", "speech", "tts", "text to speech",
    "asr", "robotics", "slam", "lidar", "pose estimation",
}

NLP_IR_CATEGORIES: Set[str] = {
    "nlp", "natural language processing", "information retrieval",
    "retrieval", "search", "text mining", "named entity recognition",
    "ner", "question answering", "semantic search",
}

# ---------------------------------------------------------------------------
# Education mappings (Req 8.6)
# ---------------------------------------------------------------------------
EDUCATION_TIER_VALUES: Dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
    "unknown": 0.25,
}

# Degree level values. Keys are normalized (lowercase, no dots) variant forms.
DEGREE_LEVEL_VALUES: Dict[str, float] = {
    # Doctorate
    "phd": 1.0,
    "ph d": 1.0,
    "doctorate": 1.0,
    "doctoral": 1.0,
    # Masters
    "me": 0.8,
    "mtech": 0.8,
    "msc": 0.8,
    "ms": 0.8,
    "masters": 0.8,
    "master": 0.8,
    "mca": 0.8,
    "mba": 0.8,
    # Bachelors
    "btech": 0.6,
    "be": 0.6,
    "bsc": 0.6,
    "bs": 0.6,
    "bachelors": 0.6,
    "bachelor": 0.6,
    "bca": 0.6,
}

DEGREE_LEVEL_DEFAULT: float = 0.6

# Field-of-study relevance to AI/ML/CS (Req 8.6).
AI_ML_FIELDS: Set[str] = {
    "computer science", "machine learning", "artificial intelligence",
    "data science", "ai",
}
ADJACENT_FIELDS: Set[str] = {
    "mathematics", "statistics", "electronics", "information technology",
    "physics",
}
FIELD_RELEVANCE_RELEVANT: float = 1.0
FIELD_RELEVANCE_ADJACENT: float = 0.5
FIELD_RELEVANCE_UNRELATED: float = 0.2

# Education composite sub-weights (Req 8.6): 0.4*tier + 0.3*degree + 0.3*field.
EDUCATION_TIER_WEIGHT: float = 0.4
EDUCATION_DEGREE_WEIGHT: float = 0.3
EDUCATION_FIELD_WEIGHT: float = 0.3

# ---------------------------------------------------------------------------
# Composite scoring weights (Req 8.1)
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "skill": 0.35,
    "career": 0.25,
    "experience": 0.15,
    "behavioral": 0.10,
    "education": 0.10,
    "location_work_mode": 0.05,
}


# ---------------------------------------------------------------------------
# Education helper functions (Req 8.6)
# ---------------------------------------------------------------------------
def tier_value(tier: str) -> float:
    """Map an institution tier string to its value (default unknown=0.25)."""
    if not tier:
        return EDUCATION_TIER_VALUES["unknown"]
    return EDUCATION_TIER_VALUES.get(tier.strip().lower(), EDUCATION_TIER_VALUES["unknown"])


def _normalize_degree_token(degree: str) -> str:
    """Lowercase a degree string and strip dots/punctuation for matching.

    Converts e.g. "M.Tech" -> "mtech", "Ph.D." -> "phd", "B.E." -> "be".
    """
    cleaned = []
    for ch in degree.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append(" ")
        # drop dots, slashes, and other punctuation
    return "".join(cleaned).strip()


def degree_to_level(degree: str) -> float:
    """Map a degree string to a level value (Ph.D=1.0, masters=0.8, bachelors=0.6).

    Matching is tolerant of punctuation and spacing ("M.Tech", "M Tech",
    "MTech" all resolve to the masters value). Falls back to the bachelors
    default when the degree is unrecognized but present.
    """
    if not degree:
        return DEGREE_LEVEL_DEFAULT

    normalized = _normalize_degree_token(degree)
    if not normalized:
        return DEGREE_LEVEL_DEFAULT

    # 1. Direct match against the collapsed (spaceless) token.
    collapsed = normalized.replace(" ", "")
    if collapsed in DEGREE_LEVEL_VALUES:
        return DEGREE_LEVEL_VALUES[collapsed]

    # 2. Token-wise match (handles "master of science", "bachelor of tech").
    tokens = normalized.split()
    for token in tokens:
        if token in DEGREE_LEVEL_VALUES:
            return DEGREE_LEVEL_VALUES[token]

    # 3. Keyword heuristics on the full string.
    if "phd" in collapsed or "doctor" in normalized:
        return 1.0
    if "master" in normalized:
        return 0.8
    if "bachelor" in normalized:
        return 0.6

    return DEGREE_LEVEL_DEFAULT


def field_relevance(field_of_study: str) -> float:
    """Map a field_of_study to its relevance value.

    relevant (AI/ML/CS/data science) = 1.0, adjacent (math, stats,
    electronics, IT, physics) = 0.5, unrelated = 0.2.
    """
    if not field_of_study:
        return FIELD_RELEVANCE_UNRELATED

    normalized = field_of_study.strip().lower()
    if not normalized:
        return FIELD_RELEVANCE_UNRELATED

    # Exact membership first.
    if normalized in AI_ML_FIELDS:
        return FIELD_RELEVANCE_RELEVANT
    if normalized in ADJACENT_FIELDS:
        return FIELD_RELEVANCE_ADJACENT

    # Substring-aware match (e.g., "M.Sc Computer Science and Engineering").
    for relevant in AI_ML_FIELDS:
        if relevant in normalized:
            return FIELD_RELEVANCE_RELEVANT
    for adjacent in ADJACENT_FIELDS:
        if adjacent in normalized:
            return FIELD_RELEVANCE_ADJACENT

    return FIELD_RELEVANCE_UNRELATED
