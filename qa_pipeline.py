import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

def create_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    You are a helpful assistant. Use only the provided context to answer.
    If the answer isn't in the context, say:
    "I couldn’t find this information in the provided documents."

    Context:
    {context}

    Question:
    {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain
