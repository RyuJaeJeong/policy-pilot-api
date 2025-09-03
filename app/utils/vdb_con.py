import json
import logging
from icecream import ic
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def get_vector_store(vc_name: str, embeddings: Embeddings) -> QdrantVectorStore:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=vc_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    file_path = "./output.json"
    with open(file_path, "r", encoding="utf-8") as file:
        arr = json.load(file)
    docs = [Document(page_content=f"""제 {row["장"]} 장\n제 {row["조"]} 조 [{row["조제목"]}]\n{row["내용"]}""", metadata=row) for row in arr]
    ic(f"docs: {docs}")
    vector_store = QdrantVectorStore(
            client=client,
            collection_name=vc_name,
            embedding=embeddings,
    )
    return vector_store

