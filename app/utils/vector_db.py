from fastapi import Depends
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from .embedding_model import get_embeddings
from dotenv import load_dotenv
from icecream import ic
from functools import lru_cache
import os

@lru_cache
def get_client():
   client_url = os.environ["VECTOR_DB_CLIENT"]
   ic(client_url)
   client = QdrantClient(client_url)
   return client

def get_collection(col_name:str):
   client = get_client()
   collections = client.get_collections().collections
   collection_names = [col.name for col in collections]
   if col_name not in collection_names:
      vector_size = int(os.environ["VECTOR_DB_SIZE"])
      client.create_collection(
         collection_name=col_name,
         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
      )
   embedding_model = get_embeddings()
   vdb = QdrantVectorStore(
      client=client,
      collection_name=col_name,
      embedding=embedding_model
   )   
   return vdb 