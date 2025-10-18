import streamlit as st
import traceback
import os

try:
    from src.preprocess import load_and_chunk
    from src.embed_store import create_vector_store
    from src.qa_pipeline import create_qa_chain
except Exception as e:
    st.error("⚠️ Error importing files. Make sure your folder structure is correct.")
    st.text(traceback.format_exc())

st.set_page_config(page_title="RAG Assistant", page_icon="📘")
st.title("📘 Retrieval-Augmented Generation (RAG) Assistant")

if "ready" not in st.session_state:
    st.session_state.ready = False

with st.sidebar:
    st.header("🔧 Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Process Document") and uploaded_file:
        try:
            temp_path = f"data/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            from src.preprocess import load_and_chunk
            from src.embed_store import create_vector_store

            chunks = load_and_chunk(temp_path)
            create_vector_store(chunks)
            st.session_state.ready = True
            st.success(" Vector store created successfully!")
        except Exception as e:
            st.error("❌ Error while processing the document")
            st.text(traceback.format_exc())

if st.session_state.ready:
    try:
        from src.qa_pipeline import create_qa_chain
        qa_chain = create_qa_chain()
        query = st.text_input("Ask a question about your document:")
        if query:
            answer = qa_chain.run(query)
            st.write("### 🤖 Answer:")
            st.write(answer)
    except Exception as e:
        st.error("❌ Error running the QA chain")
        st.text(traceback.format_exc())
