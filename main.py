import os
import logging

import streamlit as st
from streamlit.runtime.state import SessionState

from document_processing import process_document, is_document_already_processed
from vector_store import (
    add_to_vector_collection,
    query_collection,
    list_uploaded_documents,
    delete_document,
)
from llm_interface import call_llm
from chat import chat_interface
from utils import re_rank_cross_encoders, normalize_scores, get_confidence_color
from deep_translator import GoogleTranslator
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="RAG Question Answer", layout="wide")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

def main():
    # Sidebar
    with st.sidebar:
        st.title("Document Assistant")
        st.markdown("Upload documents and ask questions based on their content.")

        # Language selection
        language_options = {
            "English": "en",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
        }
        selected_language_name = st.selectbox("Select Output Language:", list(language_options.keys()))
        selected_language = language_options[selected_language_name]
        # Store the selected language code in session state
        st.session_state['selected_language'] = selected_language

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "html"],
            accept_multiple_files=True
        )
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        # Check if the document has already been processed
                        if is_document_already_processed(uploaded_file.name):
                            st.warning(
                                f"Document '{uploaded_file.name}' has already been processed."
                            )
                            continue
                        # Process the document
                        docs = process_document(
                            uploaded_file,
                            chunk_size=config["chunk_size"],
                            chunk_overlap=config["chunk_overlap"],
                        )
                        # Add to vector collection
                        add_to_vector_collection(docs, uploaded_file.name)
            else:
                st.warning("Please upload at least one document.")

    # Main Content
    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Document Library", "Chat"])

    with tab1:
        st.header("Ask a Question")
        question = st.text_area("Enter your question:", key="question_input")
        n_results = st.slider("Number of documents to use:", 1, 20, 10, key="n_results_slider")
        if st.button("Get Answer", key="get_answer_button"):
            if question:
                with st.spinner("Retrieving answer..."):
                    # Query the vector store and generate an answer
                    results = query_collection(question, n_results)
                    if results and 'documents' in results and 'distances' in results:
                        documents = results['documents'][0]  # Assuming single query
                        distances = results['distances'][0]
                        metadatas = results['metadatas'][0]  # Retrieve metadata
                        # Normalize retrieval scores
                        retrieval_scores = normalize_scores(distances)
                        # Re-rank documents
                        relevant_text, relevant_indices, re_rank_scores = re_rank_cross_encoders(question, documents)
                        # Combine scores for the top documents
                        combined_scores = []
                        for i in range(len(relevant_indices)):
                            idx = relevant_indices[i]
                            combined_score = (retrieval_scores[idx] + re_rank_scores[i]) / 2
                            combined_scores.append(combined_score)
                        # Compute overall confidence score
                        if combined_scores:
                            confidence_score = sum(combined_scores) / len(combined_scores)
                        else:
                            confidence_score = 0.0
                        # Display the confidence score
                        color = get_confidence_color(confidence_score)
                        st.markdown(f"**Confidence Score:** <span style='color:{color}'>{confidence_score:.2f}</span>", unsafe_allow_html=True)
                        # Retrieve the selected language
                        selected_language = st.session_state.get('selected_language', 'en')
                        # Generate the answer
                        response_generator = call_llm(relevant_text, question, selected_language)
                        # Display the answer using a placeholder
                        answer_placeholder = st.empty()
                        full_response = ""

                        # Collect the full response
                        for chunk in response_generator:
                            full_response += chunk

                        if selected_language == 'en':
                            # Display the response in English
                            answer_placeholder.markdown(full_response)
                        else:
                            # Translate and display the response
                            translated_answer = GoogleTranslator(source='auto', target=selected_language).translate(full_response)
                            answer_placeholder.markdown(translated_answer)

                        # Display sources
                        st.subheader("Sources")
                        sources = []
                        for i, idx in enumerate(relevant_indices):
                            metadata = metadatas[idx]
                            file_name = metadata.get('file_name', 'Unknown')
                            chunk_number = metadata.get('chunk', 'N/A')
                            source_info = f"**Source {i+1}:** {file_name} (Chunk {chunk_number})"
                            st.markdown(source_info)
                            sources.append(source_info)

                        # Combine answer and sources into downloadable content
                        download_content = f"**Answer:**\n{full_response}\n\n**Sources:**\n" + "\n".join(sources)

                        # Add a download button
                        st.download_button(
                            label="Download Answer and Sources",
                            data=download_content,
                            file_name="answer_and_sources.txt",
                            mime="text/plain",
                        )
                    else:
                        st.warning("No relevant documents found.")
            else:
                st.warning("Please enter a question.")

    with tab2:
        st.header("Your Documents")
        documents = list_uploaded_documents()
        if documents:
            for doc in documents:
                st.markdown(f"**{doc}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Reprocess {doc}", key=f"reprocess_{doc}"):
                        # Reprocess the document
                        st.info(f"Reprocessing {doc}...")
                        # (Reprocessing logic)
                with col2:
                    if st.button(f"Delete {doc}", key=f"delete_{doc}"):
                        # Delete the document
                        delete_document(doc)
                        st.info(f"Deleted {doc}")
        else:
            st.info("No documents uploaded yet.")

    #with tab3:
        #chat_interface()

if __name__ == "__main__":
    main()