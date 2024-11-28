import os
import logging

import streamlit as st

from document_processing import process_document, is_document_already_processed
from vector_store import add_to_vector_collection, query_collection, list_uploaded_documents, delete_document
from llm_interface import call_llm
from utils import re_rank_cross_encoders
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="RAG Question Answer", layout="wide")

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def main():
    # Sidebar
    with st.sidebar:
        st.title("Document Assistant")
        st.markdown("Upload documents and ask questions based on their content.")

        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        normalize_uploaded_file_name = uploaded_file.name.translate(
                            str.maketrans({"-": "_", ".": "_", " ": "_"})
                        )
                        if is_document_already_processed(normalize_uploaded_file_name):
                            st.info(f"Document '{uploaded_file.name}' has already been processed.")
                            continue
                        all_splits = process_document(
                            uploaded_file, chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap']
                        )
                        if all_splits:
                            add_to_vector_collection(all_splits, normalize_uploaded_file_name)
            else:
                st.warning("Please upload at least one PDF file.")

    # Main Content
    tab1, tab2 = st.tabs(["Ask Questions", "Document Library"])

    with tab1:
        st.header("Ask a Question")
        question = st.text_area("Enter your question:")
        n_results = st.slider("Number of documents to use:", 1, 20, 10)
        if st.button("Get Answer"):
            if question:
                with st.spinner("Retrieving answer..."):
                    results = query_collection(question, n_results=n_results)
                    if results and results.get("documents"):
                        context = results.get("documents")[0]
                        relevant_text, relevant_text_ids = re_rank_cross_encoders(question, context)
                        if relevant_text:
                            response = call_llm(context=relevant_text, prompt=question)
                            placeholder = st.empty()
                            response_text = ""
                            for chunk in response:
                                response_text += chunk
                                placeholder.markdown(response_text)
                            with st.expander("See retrieved documents"):
                                st.write(results)
                            with st.expander("See most relevant document ids"):
                                st.write(relevant_text_ids)
                                st.write(relevant_text)
                            # Feedback Section
                            st.markdown("### Feedback")
                            feedback = st.radio(
                                "Was this answer helpful?",
                                ("Yes", "Somewhat", "No"),
                                horizontal=True,
                                key=f"feedback_{question}"
                            )
                            if st.button("Submit Feedback", key=f"submit_feedback_{question}"):
                                # Store feedback (implement storage logic)
                                st.success("Thank you for your feedback!")
                        else:
                            st.warning("No relevant context found to answer the question.")
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
                        # Implement reprocessing logic
                        # For reprocessing, you may need to re-upload the document or store the original content
                        # Here, we'll assume the original document is accessible
                        # For simplicity, this part is left as a placeholder
                with col2:
                    if st.button(f"Delete {doc}", key=f"delete_{doc}"):
                        # Delete the document
                        delete_document(doc)
                        st.info(f"Deleted {doc}")
        else:
            st.info("No documents uploaded yet.")

if __name__ == "__main__":
    main()