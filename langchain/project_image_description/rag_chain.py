from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GooglePalm

def get_retriever():
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.load_local("vector_db", embeddings)
    return vectordb.as_retriever()

def get_rag_chain():
    retriever = get_retriever()
    llm = GooglePalm(google_api_key=os.getenv("GEMINI_API_KEY"))
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
