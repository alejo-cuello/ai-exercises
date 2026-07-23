"""
ingest.py — run this whenever you have new/updated documents to re-ingest.

Usage:
    python ingest.py --source ./data/raw
    python ingest.py --source ./data/raw --tag   # also tag each chunk with an LLM before indexing

Locally, this writes straight to ./chroma_db. If you're syncing to GCS for
Cloud Run, follow up with:
    gsutil -m rsync -r ./chroma_db gs://YOUR_PROJECT-chroma-db/chroma_db
"""

import argparse
import os
import re

from google.api_core.exceptions import ResourceExhausted
from langchain_classic.indexes import SQLRecordManager, index
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "langchain_docs_index")
RECORD_MANAGER_DB_URL = os.environ.get(
    "RECORD_MANAGER_DB_URL", f"sqlite:///{CHROMA_PERSIST_DIR}/records.db"
)

CHAPTER_FILENAME_RE = re.compile(r"book_chapter_(\d+)\.pdf")
KEYS_TO_KEEP = ["creationdate", "source", "total_pages", "page", "volume", "chapter", "start_index"]


class Tags(BaseModel):
    content_type: str = Field(
        description="The functional nature of the text fragment. The possible values must be: 'Narrative', 'Summary', 'Historical Notes', 'Examples', 'Exercises', 'Table of Contents', 'Algorithm Description'",
        default="Narrative"
    )
    contains_math_latex: bool = Field(
        default=False,
        description="Whether the chunk contains formal mathematical notation, probability equations, or LaTeX-style formulas.",
    )
    contains_regex: bool = Field(
        default=False,
        description="Whether the chunk includes regular expression patterns or explains regex syntax.",
    )
    contains_code_or_cli: bool = Field(
        default=False,
        description="Whether the text includes Python code, Unix commands (tr, sort, uniq), or pseudocode for algorithms.",
    )
    contains_table: bool = Field(
        default=False,
        description="Whether the fragment includes a data table, such as word counts, probabilities, or ASCII/Unicode mappings.",
    )
    talks_about_tokenization: bool = Field(
        default=False,
        description="Whether the chunk discusses the process of segmenting running input text into tokens.",
    )
    talks_about_ngrams: bool = Field(
        default=False,
        description="Whether the chunk discusses sequences of n words or Markov-based models that look n-1 words into the past.",
    )
    talks_about_language_modeling: bool = Field(
        default=False,
        description="Whether the chunk discusses the general machine learning task of predicting upcoming words or assigning probabilities to sequences.",
    )
    talks_about_morphology: bool = Field(
        default=False,
        description="Whether the chunk discusses morphemes as the minimal meaning-bearing units of language.",
    )
    linguistic_focus: str = Field(
        description="Specific languages or language families discussed as examples (e.g., Chinese, English, Spanish, Vietnamese).",
        default="General"
    )
    importance_score: str = Field(
        description="The pedagogical value of the chunk for understanding core NLP foundations. The possible values must be: 'High', 'Medium', 'Low'",
        default="Medium"
    )


def load_documents(source_dir: str):
    docs = []
    files = sorted(f for f in os.listdir(source_dir) if f.endswith(".pdf"))
    for file in files:
        match = CHAPTER_FILENAME_RE.search(file)
        if not match:
            continue
        chapter_num = match.group(1)
        loader = PyMuPDFLoader(file_path=os.path.join(source_dir, file))
        for doc in loader.lazy_load():
            doc.metadata.update({"volume": 1, "chapter": int(chapter_num)})
            docs.append(doc)
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". "]
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if k in KEYS_TO_KEEP}
    return chunks


def tag_chunks(chunks, llm):
    tagger = llm.with_structured_output(Tags)
    for idx, chunk in enumerate(chunks):
        try:
            result = tagger.invoke(chunk.page_content)
            chunk.metadata.update(result.model_dump())
        except ResourceExhausted:
            print(f"\n[429 Rate Limit] Hit at chunk {idx}. Stopping tagging; "
                  f"remaining chunks will be indexed untagged.")
            break
        except Exception as e:
            print(f"Unexpected error tagging chunk {idx}: {e}. "
                  f"Stopping tagging; remaining chunks will be indexed untagged.")
            break
    return chunks


def main(source_dir: str, tag: bool = False):
    print(f"Loading documents from {source_dir} ...")
    docs = load_documents(source_dir)
    print(f"Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    api_key = os.environ.get("API_KEY")

    if tag:
        print("Tagging chunks ...")
        llm = ChatGoogleGenerativeAI(
            api_key=api_key, model="gemini-2.5-flash", temperature=0.0, timeout=None
        )
        chunks = tag_chunks(chunks, llm)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", google_api_key=api_key, task_type="RETRIEVAL_DOCUMENT"
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    record_manager = SQLRecordManager(
        namespace=f"chroma/{COLLECTION_NAME}",
        db_url=RECORD_MANAGER_DB_URL,
    )
    record_manager.create_schema()

    result = index(
        docs_source=chunks,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )
    print(f"Indexing result: {result}")
    print(f"Done. Chroma store written to {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder of documents to ingest")
    parser.add_argument("--tag", action="store_true", help="Tag each chunk with an LLM before indexing")
    args = parser.parse_args()
    main(args.source, tag=args.tag)
