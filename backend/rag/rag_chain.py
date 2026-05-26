import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from backend.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL

llm = ChatOllama(
    model=os.getenv("DOCUMIND_OLLAMA_MODEL", OLLAMA_MODEL),
    temperature=0,
    base_url=os.getenv("DOCUMIND_OLLAMA_BASE_URL", OLLAMA_BASE_URL),
)


prompt_template = """
You are an intelligent AI assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""


prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def generate_answer(context, question):

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content