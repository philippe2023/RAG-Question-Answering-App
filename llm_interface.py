# llm_interface.py

import logging
from typing import Generator

import ollama
import streamlit as st
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

system_prompt = """
You are an AI assistant that provides detailed answers based solely on the given context.

Instructions:

- Use **only** the information in the "Context" to answer the "Question".
- Do **not** include any external knowledge or assumptions.
- If the context doesn't contain sufficient information to answer the question, respond: "The context does not provide enough information to answer this question."

Formatting Guidelines:

- Use clear and concise language.
- Organize your answer into paragraphs for readability.
- Use bullet points or numbered lists to break down complex information when appropriate.
- Include headings or subheadings if relevant.
- Ensure proper grammar, punctuation, and spelling.

Remember: Base your entire response solely on the information provided in the context.
"""

def call_llm(context: str, prompt: str) -> Generator[str, None, None]:
    """Calls the language model with context and prompt to generate a response."""
    try:
        response = ollama.chat(
            model=config['llm_model'],
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {prompt}",
                },
            ],
        )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break
    except Exception as e:
        logging.error(f"An error occurred while generating the response: {e}")
        st.error(f"An error occurred while generating the response: {e}")