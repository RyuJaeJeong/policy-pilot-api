import os

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch

class CustomEmbeddingService(Embeddings):

    def __init__(self):
        model_dir = os.environ["EMBEDDING_MODEL_DIR"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_dir, device=device)

    def embed_query(self, text: str) -> list[float]:
        embeded = self.model.encode(text, normalize_embeddings=True)
        return embeded.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeded_list = self.model.encode(texts, normalize_embeddings=True)
        embeded_to_dict = [embeded.tolist() for embeded in embeded_list]
        return embeded_to_dict
