import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from custom_logging import logger
from custom_exception import CustomException

load_dotenv()

def retrieve_answer_from_docs(question: str, retriever):
    """
    Retrieve the answer to a question from the documents.

    Args:
        question (str): The question to answer.
        retriever: The retriever instance to search for answers.

    Returns:
        str: The generated answer.
    """
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')

        prompt = PromptTemplate(
            template="""
                # Your role
                You are a brilliant expert at understanding the intent of the questioner and the crux of the question, and providing the most optimal answer  from the docs to the questioner's needs from the documents you are given.

                # Instruction
                Your task is to answer the question  using the following pieces of retrieved context delimited by XML tags.

                <retrieved context>
                Retrieved Context:
                {context}
                </retrieved context>

                # Constraint
                1. Think deeply and multiple times about the user's question\nUser's question:\n{question}\nYou must understand the intent of their question and provide the most appropriate answer.
                - Ask yourself why to understand the context of the question and why the questioner asked it, reflect on it, and provide an appropriate response based on what you understand.
                2. Choose the most relevant content(the key content that directly relates to the question) from the retrieved context and use it to generate an answer.
                3. Generate a concise, logical answer. When generating the answer, Do Not just list your selections, But rearrange them in context so that they become paragraphs with a natural flow.
                4. When you don't have retrieved context for the question or If you have a retrieved documents, but their content is irrelevant to the question, you should answer 'I can't find the answer to that question in the material I have'.
                5. Use five sentences maximum. Keep the answer concise but logical/natural/in-depth.
                6. At the end of the response provide metadata provided in the relevant docs , For example:"Metadata: page: 19, source: /content/OCR_RSCA/Analyse docs JVB + mails et convention FOOT INNOVATION.pdf'. Return Just the page and source

                # Question:
                {question}""",
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

        rag_chain = (
            {"context": retriever, "question": question}
            | prompt
            | llm
        )

        answer = rag_chain.invoke(question)
        logger.info("Answer retrieved successfully")
        return answer
    except Exception as e:
        raise CustomException(e, sys)
