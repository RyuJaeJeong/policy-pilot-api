from .base_retriever_service import BaseRetrieverService
from app.schemas.chat_schema import State
from langchain.retrievers.multi_query import MultiQueryRetriever

class MultiQueryRetrieverService(BaseRetrieverService):
    async def retrieve(self, state:State) -> dict:
        """ MultiQueryRetriever 에 의한 문맥 검색 """
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=self.vector_store.as_retriever(search_kwargs={"k":7}), llm=self.llm)
        retrieved_docs = await multi_query_retriever.ainvoke(state["question"])
        return {"context": retrieved_docs}