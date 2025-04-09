from fastapi import Depends
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from .embedding_model import get_embeddings
from dotenv import load_dotenv
from icecream import ic
from functools import lru_cache
import os
import json


@lru_cache
def get_client():
   client_url = os.environ["VECTOR_DB_URL"]
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
   
   cnt = client.count(collection_name=col_name, exact=True)
   ic(cnt)
   if cnt.count == 0:
      list = []
      with open("./output.json", "r", encoding="utf-8") as f:
         json_data = json.load(f)
      max = 0   
      for item in json_data:
         str = f"제{item['장']}장 {item['장제목']}\n제{item['조']}조 {item['조제목']}\n{item['내용']}"
         if len(str) > max:
            max = len(str)
         list.append(Document(page_content=str))
      vdb.add_documents(documents=list)                  
   return vdb 