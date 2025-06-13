from qdrant_client import QdrantClient

client = QdrantClient(path="./app/vdb")
client.delete_collection("sample_policies_20250612_hybrid_02")