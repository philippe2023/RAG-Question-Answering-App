# chat.py

import streamlit as st
import sys
import ollama
from langchain.schema import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

def chat_interface():
    st.header("Chat with the Assistant")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Select LLM model
    models = get_models()
    selected_model = st.selectbox("Select LLM:", models, key="selected_chat_model")
    st.session_state["selected_chat_model"] = selected_model

    # Display chat history
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)

    # Chat input
    prompt = st.chat_input("Type your message here...")
    if prompt:
        # Display user message
        st.chat_message("user").write(prompt)
        st.session_state["chat_history"].append(HumanMessage(content=prompt))

        # Get response from LLM
        with st.spinner("Assistant is typing..."):
            llm = ChatOllama(model=selected_model, temperature=0.7)
            response = ""
            for chunk in llm.stream(prompt):
                response += chunk

        # Display assistant message
        st.chat_message("assistant").write(response)
        st.session_state["chat_history"].append(AIMessage(content=response))

def get_models():
    models = ollama.list()
    if not models:
        st.error("No models found. Please visit https://ollama.dev/models to download models.")
        return []
    return [model["name"] for model in models["models"]]