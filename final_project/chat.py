"""
chat.py — LangChain + Chroma retrieval logic.

Points at the Chroma persist directory (local path in dev, the GCS FUSE
mount path in prod — see CHROMA_PERSIST_DIR). Swap the embeddings/LLM
provider imports below for whichever provider you're using.

Retrieval combines multi-query expansion (an LLM rewrites the question into
several phrasings), an ensemble of BM25 (keyword) + MMR (diverse semantic)
retrievers, and a long-context reorder pass so the most relevant chunks sit
at the start/end of the prompt rather than the middle. Chat memory keeps the
last few messages verbatim and summarizes anything older.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

load_dotenv()

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "../data/vectors/chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "langchain_docs_index")
API_KEY = os.environ.get("API_KEY","")

# Messages kept verbatim in the prompt; anything older gets summarized instead.
RECENT_MESSAGES_WINDOW = 4


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.0)


@st.cache_resource
def get_vectordb():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=API_KEY)
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIR)


@st.cache_resource
def get_retriever():
    """Multi-query retrieval over an ensemble of BM25 + MMR, reranked with long-context reorder."""
    vectordb = get_vectordb()
    llm = get_llm()

    all_docs = vectordb.get(include=["documents", "metadatas"])

    docs_from_chroma = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]


    bm25_retriever = BM25Retriever.from_documents(docs_from_chroma)
    bm25_retriever.k = 3

    mmr_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6, "lambda_mult": 0.5},
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mmr_retriever], weights=[0.5, 0.5]
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=ensemble_retriever, llm=llm)

    reordering = LongContextReorder()
    return multi_query_retriever | RunnableLambda(lambda docs: list(reordering.transform_documents(docs)))


@st.cache_resource
def get_summarizer_chain():
    llm = get_llm()
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following past conversation history into a "
                   "single concise paragraph focusing on vital user details."),
        MessagesPlaceholder("older_history"),
    ])
    return summary_prompt | llm | StrOutputParser()


@st.cache_resource
def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()

    # Rewrites the latest question into a standalone query using recent chat
    # history, so follow-up questions like "what about the second one?"
    # retrieve correctly.
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
                   "answer, say so.\n\n"
                   "Summary of the earlier conversation: {summary}\n\n"
                   "Context:\n{context}"),
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


def split_history(history: list):
    """Keep the most recent messages verbatim; anything older gets summarized."""
    if len(history) <= RECENT_MESSAGES_WINDOW:
        return history, []
    return history[-RECENT_MESSAGES_WINDOW:], history[:-RECENT_MESSAGES_WINDOW]


def get_answer(question: str, db_messages: list[dict]) -> str:
    history = format_history(db_messages)
    recent_history, older_history = split_history(history)

    if older_history:
        summary = get_summarizer_chain().invoke({"older_history": older_history})
    else:
        summary = "No previous history to summarize."

    chain = get_qa_chain()
    result = chain.invoke({
        "input": question,
        "chat_history": recent_history,
        "summary": summary,
    })
    return result["answer"]
