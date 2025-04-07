from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

@lru_cache
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")
