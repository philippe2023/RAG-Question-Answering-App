import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder
import streamlit as st

@st.cache_resource
def load_cross_encoder_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def re_rank_cross_encoders(prompt: str, documents: List[str]) -> Tuple[str, List[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""
    try:
        relevant_text = ""
        relevant_text_ids = []

        encoder_model = load_cross_encoder_model()

        # Create pairs of (prompt, document)
        pairs = [(prompt, doc) for doc in documents]

        # Get scores
        scores = encoder_model.predict(pairs)

        # Get indices sorted by scores in descending order
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Select top 3 documents
        top_k = 3
        for idx in sorted_indices[:top_k]:
            relevant_text += documents[idx] + " "
            relevant_text_ids.append(idx)

        return relevant_text.strip(), relevant_text_ids
    except Exception as e:
        logging.error(f"An error occurred during document re-ranking: {e}")
        st.error(f"An error occurred during document re-ranking: {e}")
        return "", []