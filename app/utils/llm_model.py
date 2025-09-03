import logging
from langchain_openai import ChatOpenAI

def get_llm():
    logging.info(f"ChatOpenAI created")
    return ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.3,
    )