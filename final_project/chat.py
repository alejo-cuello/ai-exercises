"""
chat.py — LangChain + Chroma retrieval logic.

Points at the Chroma persist directory (local path in dev, the GCS FUSE
mount path in prod — see CHROMA_PERSIST_DIR). Swap the embeddings/LLM
provider imports below for whichever provider you're using.
"""

import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "../data/vectors/chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "langchain_docs_index")
API_KEY = os.environ["API_KEY"]

@st.cache_resource
def get_vectordb():
    embeddings = ChatGoogleGenerativeAI()
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIR)


@st.cache_resource
def get_qa_chain():
    #TODO: ver qué carajo meter en esta funcion. Luego mirar los archivos docker, probar en local y luego deployar
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=API_KEY, task_type="RETRIEVAL_DOCUMENT")

    # Rewrites the latest question into a standalone query using chat history,
    # so follow-up questions like "what about the second one?" retrieve correctly.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, "
                   "rephrase the question to be a standalone question. "
                   "Do not answer it, just reformulate it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the retrieved context "
                   "below to answer the question. If you don't know the "
                   "answer, say so.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)


def format_history(db_messages: list[dict]):
    """Convert stored {role, content} rows into LangChain message objects."""
    history = []
    for m in db_messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def get_answer(question: str, db_messages: list[dict]) -> str:
    chain = get_qa_chain()
    chat_history = format_history(db_messages)
    result = chain.invoke({"input": question, "chat_history": chat_history})
    return result["answer"]
