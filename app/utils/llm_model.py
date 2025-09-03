import os

from langchain_openai import ChatOpenAI
from functools import lru_cache
import logging

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.3,
    )