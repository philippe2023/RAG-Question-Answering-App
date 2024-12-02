# utils.py

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder
import streamlit as st

@st.cache_resource
def load_cross_encoder_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def re_rank_cross_encoders(prompt: str, documents: List[str]) -> Tuple[str, List[int], List[float]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""
    try:
        encoder_model = load_cross_encoder_model()
        # Create pairs of (prompt, document)
        pairs = [(prompt, doc) for doc in documents]
        # Get scores
        scores = encoder_model.predict(pairs)
        # Normalize scores to 0-1 range
        max_score = max(scores)
        min_score = min(scores)
        normalized_scores = [
            (s - min_score) / (max_score - min_score) if max_score != min_score else 1.0
            for s in scores
        ]
        # Get indices sorted by scores in descending order
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Select top 3 documents
        top_k = 3
        relevant_text = ""
        relevant_text_ids = []
        relevant_scores = []
        for idx in sorted_indices[:top_k]:
            relevant_text += documents[idx] + " "
            relevant_text_ids.append(idx)
            relevant_scores.append(normalized_scores[idx])
        return relevant_text.strip(), relevant_text_ids, relevant_scores
    except Exception as e:
        logging.error(f"An error occurred during document re-ranking: {e}")
        st.error(f"An error occurred during document re-ranking: {e}")
        return "", [], []

def normalize_scores(distances: List[float]) -> List[float]:
    """Normalizes a list of distances to a confidence score between 0 and 1."""
    max_distance = max(distances)
    min_distance = min(distances)
    # Handle the case where all distances are equal
    if max_distance == min_distance:
        return [1.0 for _ in distances]
    normalized_scores = [(max_distance - d) / (max_distance - min_distance) for d in distances]
    return normalized_scores

def get_confidence_color(score: float) -> str:
    """Returns a color code based on the confidence score."""
    if score > 0.75:
        return "green"
    elif score > 0.5:
        return "orange"
    else:
        return "red"