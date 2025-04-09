from langchain.chat_models import init_chat_model
from functools import lru_cache

@lru_cache
def get_llm():
    return init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")