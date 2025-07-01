from uuid import uuid4
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import json


""" 
###################################
  * 데이터 로딩 START
###################################
"""

load_dotenv()
file_path = "./output.json"
with open(file_path, "r", encoding="utf-8") as file:
    arr = json.load(file)

docs = [Document(page_content=f"""제 {row["장"]} 장\n제 {row["조"]} 조 [{row["조제목"]}]\n{row["내용"]}""", metadata=row) for row in arr]


""" 
###################################
  * 데이터 로딩 END
###################################
"""



""" 
###################################
  * vdb 세팅 START
###################################
"""

""" collection 생성 """
client = QdrantClient(path="./app/vdb")
col_nm = "vine_policies_20250701_10"
dense_vector_nm = "dense"
if not client.collection_exists(col_nm):
    client.create_collection(
        collection_name=col_nm,
        vectors_config= VectorParams(size=3072, distance=models.Distance.COSINE)
    )

""" vectorstore 생성 """
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
qdrant = QdrantVectorStore(
    client=client,
    collection_name=col_nm,
    embedding=embeddings
)

""" 데이터 입력 """
qdrant.add_documents(documents=docs, ids=[str(uuid4()) for _ in range(len(docs))])

""" 
###################################
  * vdb 세팅 END
###################################
"""
