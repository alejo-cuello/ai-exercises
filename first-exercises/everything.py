import os
import streamlit as st

from dotenv import load_dotenv
from operator import itemgetter
from functools import partial
from typing import Sequence

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_classic.indexes import SQLRecordManager
from langchain_classic.schema import Document, StrOutputParser, HumanMessage, AIMessage
from langchain_classic.schema.runnable import RunnableMap
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

k = 6

CONDENSE_QUESTION_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Questions generally contains different entities, so you should rephrase \
the question according to the entity that is being asked about. \
Do not made up any information. The only information you can \
use to formulate the standalone question is the conversation and the follow up \
question.

Chat History:
###
{chat_history}
###

Follow Up Input: {question}
Standalone Question:"""

SYSTEM_ANSWER_QUESTION_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about 'LLMs' with high quality answers and without making anything up.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

If you are unsure about how to import an element from the library, write something down \
but make it clear that you are unsure. In addition, include what should be the expected \
behavior of the element.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure.". Don't try to make up an answer. This is not a suggestion. This is a rule.

Anything between the following `context` html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.

<context>
    {context}
</context>

REMBEMBER: If there is no relevant information within the context, just say "Hmm, \
I'm not sure.". Don't try to make up an answer. This is not a suggestion. This is a rule. \
Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, \
not part of the conversation with the user."""

def initialize_vectorstore():
   
    api_key = os.getenv("API_KEY")
    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key)
    
    collection_name = "langchain_docs_index"
    namespace = f"chroma/{collection_name}"
    persist_directory = "data/vectors"
    
    _record_manager = SQLRecordManager(
        namespace=namespace,
        db_url=f"sqlite:///{persist_directory}/records.db",
    )

    _vectorstore = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=f"{persist_directory}/chroma_db"
    )
    
    _vector_keys = _vectorstore.get(
        ids=_record_manager.list_keys(),
        include=["documents", "metadatas"]
    )
    
    return _vectorstore, _vector_keys

def initialize_llm():
    
    api_key = os.getenv("API_KEY")
    
    llm = GoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=50000,
        timeout=None,
        max_retries=2
    )
    
    return llm
    
def initialize_retriever(_vectorstore, _docs_in_vectorstore, _llm):
        
    bm25_retriever = BM25Retriever.from_documents(_docs_in_vectorstore)
    bm25_retriever.k = 3

    semantic_retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.3,
        },
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=semantic_retriever,
        llm=_llm,
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, multi_query_retriever], weights=[0.7, 0.3]
    )
    
    return ensemble_retriever

def create_retriever_chain(_llm, _retriever, _use_chat_history):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

    if not _use_chat_history:
        initial_chain = (itemgetter("question")) | _retriever
        return initial_chain
    else:
        condense_question_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | CONDENSE_QUESTION_PROMPT
            | _llm
            | StrOutputParser()
        )
        conversation_chain = condense_question_chain | _retriever
        return conversation_chain

def get_k_or_less_documents(documents, k):
    if len(documents) <= k:
        return documents
    else:
        return documents[:k]

def reorder_documents(documents: list[Document]):
    reorder = LongContextReorder()

    for i, doc in enumerate(documents):
        doc.metadata["original_index"] = i

    return reorder.transform_documents(documents)


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs: list[str] = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{doc.metadata.get('original_index', i)}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)

def create_answer_chain(_llm,_retriever,_use_chat_history,_k):

    retriever_chain = create_retriever_chain(_llm, _retriever, _use_chat_history)
    _get_k_or_less_documents = partial(get_k_or_less_documents, k=_k)

    context = RunnableMap(
        {
            "context": (
                retriever_chain
                | _get_k_or_less_documents
                | reorder_documents
                | format_docs
            ),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", SYSTEM_ANSWER_QUESTION_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = prompt | _llm
    response_chain = context | response_synthesizer

    return response_chain

# ------------------------ Code ------------------------

vectorstore, vector_keys = initialize_vectorstore()
llm = initialize_llm()

docs_in_vectorstore = [
    Document(page_content=page_content, metadata=metadata)
    for page_content, metadata in zip(
        vector_keys["documents"], vector_keys["metadatas"]
    )
]

retriever = initialize_retriever(vectorstore, docs_in_vectorstore, llm)


st.title("'Speech and Language Processing' Q&A")
st.subheader("It uses a combination of keyword and semantic search to find answers.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is a Morpheme?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        use_chat_history = len(st.session_state.messages) > 1

        chat_history = []
        if use_chat_history:
            for message in st.session_state.messages[:-1]:
                if message["role"] == "user":
                    chat_history.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    chat_history.append(AIMessage(content=message["content"]))

        answer_chain = create_answer_chain(
            _llm=llm,
            _retriever=retriever,
            _use_chat_history=use_chat_history,
            _k=k,
        )

        message_placeholder = st.empty()
        full_response = ""
        for token in answer_chain.stream(
            {
                "question": prompt,
                "chat_history": chat_history,
            }
        ):
            full_response += token
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})