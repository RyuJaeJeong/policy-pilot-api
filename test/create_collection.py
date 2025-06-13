from uuid import uuid4
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
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
col_nm = "sample_policies_20250612_hybrid_02"
dense_vector_nm = "dense"
sparse_vector_nm = "sparse"
if not client.collection_exists(col_nm):
    client.create_collection(
        collection_name=col_nm,
        vectors_config={
            dense_vector_nm: models.VectorParams(size=3072, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            sparse_vector_nm: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))
        }
    )

""" vectorstore 생성 """
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
sparse_model_nm = "Qdrant/bm25"
sparse_embeddings = FastEmbedSparse(model_name=sparse_model_nm, cache_dir="./cache")
qdrant = QdrantVectorStore(
    client=client,
    collection_name=col_nm,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name=dense_vector_nm,
    sparse_vector_name=sparse_vector_nm
)

""" 데이터 입력 """
qdrant.add_documents(documents=docs, ids=[str(uuid4()) for _ in range(len(docs))])

""" 
###################################
  * vdb 세팅 END
###################################
"""
