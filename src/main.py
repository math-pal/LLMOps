from preprocess import load_documents, split_documents
from index import store_documents_to_qdrant
from retriever import retrieve_answer_from_docs
from utils import format_docs


def main():
    # Load and preprocess
    file_path = ""
    documents = load_documents(file_path)
    texts = split_documents(documents)
    
    # Index
    qdrant = store_documents_to_qdrant(texts)
    
    # Retrieve
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    question = "What is the document Audition Coucke about?"
    formatted_docs = format_docs(retriever.retrieve(question))
    answer = retrieve_answer_from_docs(question, retriever)
    
    print(answer)

if __name__ == "__main__":
    main()
