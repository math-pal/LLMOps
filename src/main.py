import os
from preprocess import load_documents, split_documents
from index import store_documents_to_qdrant
from langchain_openai import OpenAIEmbeddings
from retriever import retrieve_answer_from_docs
from utils import format_docs
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load and preprocess
    # file_path = ""
    # documents = load_documents(file_path)
    # texts = split_documents(documents)
    
    # Index
    # qdrant = store_documents_to_qdrant(texts)

    
    # Retrieve
    question = "What is the document Audition Coucke about?"
    answer = retrieve_answer_from_docs(question)
    
    print(answer)

if __name__ == "__main__":
    main()
