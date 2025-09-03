from langchain.chat_models import init_chat_model
from functools import lru_cache
import logging

def get_llm():
    return init_chat_model("gpt-4o-mini", model_provider="openai")