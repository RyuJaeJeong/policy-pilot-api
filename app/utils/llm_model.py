from langchain.chat_models import init_chat_model
from functools import lru_cache
import logging

@lru_cache()
def get_llm():
    logging.info("========== LLM 모델 초기화 ==========")
    return init_chat_model("gpt-4o-mini", model_provider="openai")