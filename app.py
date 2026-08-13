"""Streamlit entrypoint: upload a PDF, ask questions, get cited answers."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config import require_openrouter_key
from src.embeddings import build_or_load_collection, collection_name_for
from src.generation import answer_question
from src.ingestion import chunk_pdf
from src.llm_client import get_llm_client

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="centered")


def process_pdf(uploaded_file):
    """Chunk + embed an uploaded PDF, returning its Chroma collection."""
    file_bytes = uploaded_file.getvalue()
    collection_name = collection_name_for(file_bytes)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks = chunk_pdf(tmp_path, source_file=uploaded_file.name)
        collection = build_or_load_collection(chunks, collection_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return collection, len(chunks)


def main():
    st.title("📄 PDF Q&A Chatbot")
    st.caption("Upload a PDF, ask questions, get answers with page citations.")

    try:
        api_key = require_openrouter_key()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    llm_client = get_llm_client(api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "active_file" not in st.session_state:
        st.session_state.active_file = None

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.active_file:
        with st.spinner("Reading and indexing the document..."):
            collection, num_chunks = process_pdf(uploaded_file)
        st.session_state.collection = collection
        st.session_state.active_file = uploaded_file.name
        st.session_state.messages = []
        st.success(f"Indexed {uploaded_file.name} ({num_chunks} chunks). Ask away!")

    if st.session_state.collection is None:
        st.info("Upload a PDF to get started.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the document")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        stream, chunks = answer_question(
            question, st.session_state.collection, llm_client
        )
        answer = st.write_stream(stream)

        if chunks:
            pages = sorted({c.page_number for c in chunks})
            with st.expander(f"Sources: pages {', '.join(map(str, pages))}"):
                for c in chunks:
                    st.markdown(f"**Page {c.page_number}**")
                    st.text(c.text[:500] + ("..." if len(c.text) > 500 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
