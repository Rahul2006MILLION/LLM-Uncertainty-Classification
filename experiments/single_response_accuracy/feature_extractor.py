"""Single-Response Feature Extractor for Single-Response Accuracy Benchmark.

Defines:
1. Canonical 6 single-response linguistic features (computed identically via FeatureExtractor.extract_single_response_features).
2. Extended 11 single-response features (epistemic markers, certainty density, conditionals, alternatives, abstention indicators, directness, sentence metrics, question-response embedding similarity).
3. Embedding generator wrapper using local Ollama all-minilm engine.
"""

import os
import re
import math
import logging
from typing import Dict, List, Any, Tuple
import numpy as np

from src.features.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Canonical 6 features from TRUE_SINGLE_RESPONSE
CANONICAL_6_FEATURES = [
    "response_len",
    "distinct_1gram_ratio",
    "distinct_2gram_ratio",
    "question_overlap",
    "hedging_score",
    "negation_count",
]

# Extended 11 single-response features
EXTENDED_11_FEATURES = [
    "hedging_density",
    "epistemic_marker_density",
    "certainty_marker_density",
    "conditional_phrase_count",
    "alternative_answer_count",
    "abstention_indicator",
    "explicit_uncertainty_indicator",
    "sentence_count",
    "avg_sentence_len",
    "question_response_sim",
    "answer_directness",
]

ALL_17_LINGUISTIC_FEATURES = CANONICAL_6_FEATURES + EXTENDED_11_FEATURES

# Lexicons for extended feature extraction
HEDGING_TERMS = {
    "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "might", "could", "may", "suggests", "seems", "appears",
    "roughly", "approximately", "around", "generally", "typically",
    "often", "sometimes", "presumably", "potentially",
}

EPISTEMIC_MARKERS = {
    "probably", "possibly", "might", "could", "likely", "unclear",
    "uncertain", "perhaps", "maybe", "seem", "suggest", "hypothetically",
    "theoretically", "debatable", "unresolved", "ambiguous", "conjectured",
}

CERTAINTY_MARKERS = {
    "definitely", "certainly", "precisely", "undoubtedly", "exact",
    "always", "proven", "known", "clearly", "established", "invariably",
    "conclusively", "unquestionably", "undeniably", "absolutely",
}

CONDITIONAL_PATTERNS = [
    r"\bif\b",
    r"\bdepends on\b",
    r"\bdepending on\b",
    r"\bassuming\b",
    r"\bin the event that\b",
    r"\bcontingent upon\b",
    r"\bwhether\b",
    r"\bprovided that\b",
]

ALTERNATIVE_PATTERNS = [
    r"\balternatively\b",
    r"\beither\b",
    r"\bon the other hand\b",
    r"\banother interpretation\b",
    r"\btwo interpretations\b",
    r"\bdiffering views\b",
    r"\bmultiple meanings\b",
    r"\bcan refer to\b",
    r"\bcan mean\b",
]

ABSTENTION_PATTERNS = [
    r"\bi do not know\b",
    r"\bcannot determine\b",
    r"\binsufficient information\b",
    r"\bnot currently known\b",
    r"\bno scientific consensus\b",
    r"\bno consensus\b",
    r"\bunknown to science\b",
    r"\bcannot be definitively\b",
    r"\bunresolved by science\b",
    r"\bcurrently unknown\b",
    r"\bopen question\b",
    r"\bremains unknown\b",
    r"\bnot definitively known\b",
    r"\bno one knows\b",
]

EXPLICIT_UNCERTAINTY_PATTERNS = [
    r"\bis ambiguous\b",
    r"\bambiguous question\b",
    r"\bdepends on which\b",
    r"\bdepends on how\b",
    r"\binterpretation\b",
    r"\bcontext is required\b",
    r"\bmultiple people\b",
    r"\bmultiple entities\b",
]

def tokenize_words(text: str) -> List[str]:
    """Tokenize lowercase alphanumeric words."""
    return re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", text.lower())

def extract_canonical_features(response_text: str, question_text: str) -> Dict[str, float]:
    """Compute the 6 canonical single-response features using official FeatureExtractor logic."""
    return FeatureExtractor.extract_single_response_features(question_text, response_text)

def extract_extended_features(
    response_text: str,
    question_text: str,
    q_vec: np.ndarray,
    r_vec: np.ndarray
) -> Dict[str, float]:
    """Compute 11 extended single-response linguistic and contextual features."""
    words = tokenize_words(response_text)
    total_words = max(1, len(words))
    lower_text = response_text.lower()
    
    # Hedging density
    hedging_count = sum(1 for w in words if w in HEDGING_TERMS)
    hedging_density = float(hedging_count / total_words)
    
    # Epistemic marker density
    epistemic_count = sum(1 for w in words if w in EPISTEMIC_MARKERS)
    epistemic_marker_density = float(epistemic_count / total_words)
    
    # Certainty marker density
    certainty_count = sum(1 for w in words if w in CERTAINTY_MARKERS)
    certainty_marker_density = float(certainty_count / total_words)
    
    # Conditional phrase count
    conditional_phrase_count = float(sum(len(re.findall(pat, lower_text)) for pat in CONDITIONAL_PATTERNS))
    
    # Alternative answer count
    alternative_answer_count = float(sum(len(re.findall(pat, lower_text)) for pat in ALTERNATIVE_PATTERNS))
    
    # Abstention indicator (binary 0 or 1)
    abstention_indicator = 1.0 if any(re.search(pat, lower_text) for pat in ABSTENTION_PATTERNS) else 0.0
    
    # Explicit uncertainty indicator (binary 0 or 1)
    explicit_uncertainty_indicator = 1.0 if any(re.search(pat, lower_text) for pat in EXPLICIT_UNCERTAINTY_PATTERNS) else 0.0
    
    # Sentences
    raw_sentences = [s.strip() for s in re.split(r"[.!?]+", response_text) if s.strip()]
    sentence_count = float(max(1, len(raw_sentences)))
    avg_sentence_len = float(total_words / sentence_count)
    
    # Answer directness: length of first sentence
    first_sentence_words = len(tokenize_words(raw_sentences[0])) if raw_sentences else total_words
    answer_directness = float(first_sentence_words)
    
    # Question-response semantic similarity
    if q_vec is not None and r_vec is not None and len(q_vec) > 0 and len(r_vec) > 0:
        q_norm = np.linalg.norm(q_vec)
        r_norm = np.linalg.norm(r_vec)
        if q_norm > 1e-9 and r_norm > 1e-9:
            sim = float(np.dot(q_vec, r_vec) / (q_norm * r_norm))
            question_response_sim = float(np.clip(sim, -1.0, 1.0))
        else:
            question_response_sim = 0.0
    else:
        question_response_sim = 0.0
        
    return {
        "hedging_density": hedging_density,
        "epistemic_marker_density": epistemic_marker_density,
        "certainty_marker_density": certainty_marker_density,
        "conditional_phrase_count": conditional_phrase_count,
        "alternative_answer_count": alternative_answer_count,
        "abstention_indicator": abstention_indicator,
        "explicit_uncertainty_indicator": explicit_uncertainty_indicator,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "question_response_sim": question_response_sim,
        "answer_directness": answer_directness,
    }

class SingleResponseFeatureExtractor:
    """Manages feature extraction and embedding generation for the benchmark."""

    def __init__(self, embedding_engine=None):
        self.embedding_engine = embedding_engine

    def extract_all(
        self,
        records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract all feature matrices from unified records for responses[0]."""
        n = len(records)
        canonical_mat = np.zeros((n, len(CANONICAL_6_FEATURES)), dtype=np.float32)
        extended_mat = np.zeros((n, len(EXTENDED_11_FEATURES)), dtype=np.float32)
        all_ling_mat = np.zeros((n, len(ALL_17_LINGUISTIC_FEATURES)), dtype=np.float32)
        
        response_texts = [r["responses"][0]["text"] for r in records]
        question_texts = [r["question"] for r in records]
        
        # Generate embeddings
        if self.embedding_engine is not None:
            logger.info("Computing embeddings for %d questions and %d responses...", n, n)
            q_embeddings = self.embedding_engine.embed_batch(question_texts)
            r_embeddings = self.embedding_engine.embed_batch(response_texts)
        else:
            dim = 384
            q_embeddings = np.zeros((n, dim), dtype=np.float32)
            r_embeddings = np.zeros((n, dim), dtype=np.float32)
            
        feature_dicts = []
        for i, (r_txt, q_txt) in enumerate(zip(response_texts, question_texts)):
            # Exact canonical features matching existing pipeline
            c_dict = extract_canonical_features(r_txt, q_txt)
            e_dict = extract_extended_features(
                r_txt, q_txt, q_embeddings[i], r_embeddings[i]
            )
            
            for c_idx, fn in enumerate(CANONICAL_6_FEATURES):
                canonical_mat[i, c_idx] = c_dict[fn]
                all_ling_mat[i, c_idx] = c_dict[fn]
                
            for e_idx, fn in enumerate(EXTENDED_11_FEATURES):
                extended_mat[i, e_idx] = e_dict[fn]
                all_ling_mat[i, len(CANONICAL_6_FEATURES) + e_idx] = e_dict[fn]
                
            merged_dict = {**c_dict, **e_dict}
            feature_dicts.append(merged_dict)
            
        return {
            "canonical_features": canonical_mat,
            "extended_features": extended_mat,
            "all_linguistic_features": all_ling_mat,
            "response_embeddings": r_embeddings,
            "question_embeddings": q_embeddings,
            "response_texts": response_texts,
            "question_texts": question_texts,
            "feature_dicts": feature_dicts,
            "canonical_names": list(CANONICAL_6_FEATURES),
            "extended_names": list(EXTENDED_11_FEATURES),
            "all_linguistic_names": list(ALL_17_LINGUISTIC_FEATURES),
        }
