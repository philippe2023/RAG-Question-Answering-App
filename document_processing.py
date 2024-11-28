# document_processing.py

import os
import tempfile
import logging
from typing import List

from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st

from vector_store import list_uploaded_documents

def process_document(uploaded_file: UploadedFile, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Processes an uploaded PDF file by converting it to text chunks."""
    logging.info("Processing document...")
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        os.unlink(temp_file_path)  # Delete temp file

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        logging.error(f"An error occurred while processing the document: {e}")
        st.error(f"An error occurred while processing the document: {e}")
        return []

def is_document_already_processed(document_name: str) -> bool:
    """Checks if the document has already been processed."""
    documents = list_uploaded_documents()
    return document_name in documents