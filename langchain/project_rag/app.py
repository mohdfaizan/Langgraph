import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="📄 Gemini PDF Q&A", layout="centered")
st.title("📄 Ask Questions About Your PDF (Gemini + LangChain)")

uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        temp_dir = tempfile.mkdtemp()

        all_docs = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(all_docs, embeddings)
        retriever = vectordb.as_retriever()

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("PDFs processed. You can now ask questions!")

        user_question = st.text_input("Ask a question about the PDF content:")

        if user_question:
            with st.spinner("Thinking..."):
                result = qa_chain(user_question)

            st.markdown("### 💬 Answer:")
            st.write(result["result"])

            with st.expander("📚 Source Chunks"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
