from .base_retriever_service import BaseRetrieverService
from ..schemas.chat_schema import State
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import json


class BM25RetrieverService(BaseRetrieverService):

    def __init__(self, llm: BaseChatModel, col_nm: str):
        super().__init__(llm, col_nm)
        self.kiwi = Kiwi()
        self.stopwords = Stopwords()
        with open("./output.json", "r", encoding="utf-8") as file:
            arr = json.load(file)
        self.docs = [Document(page_content=f"""제 {row["장"]} 장\n제 {row["조"]} 조 [{row["조제목"]}]\n{row["내용"]}""", metadata=row) for row in arr]
        self.retriever = BM25Retriever.from_documents(self.docs, k=4, preprocess_func=self.korean_tokenizer)

    def retrieve(self, state: State) -> dict:
        """ BM25 검색기에 의한 문맥 검색 """
        retrieved_docs = self.retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def korean_tokenizer(self, text: str) -> list[str]:
        """ kiwi를 활용한 한국어 토큰화 """
        tokens = [token.form for token in self.kiwi.tokenize(text, normalize_coda=True, stopwords=self.stopwords)]
        return tokens
