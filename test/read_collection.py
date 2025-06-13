from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)
client = QdrantClient(path="./app/vdb")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", cache_dir="./cache")

qdrant = QdrantVectorStore(
    client=client,
    collection_name="sample_policies_20250612_hybrid_02",
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

query = "부서 변경 발령이 난 직원은 언제 부임 해야 하나요?"
docs = qdrant.similarity_search_with_score(query)
for doc in docs:
    print(doc)
    print("=========================")


