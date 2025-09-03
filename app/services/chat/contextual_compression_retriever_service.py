from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from .base_retriever_service import BaseRetrieverService
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from app.schemas.chat_schema import State
from app.utils.llm_model import get_llm

class ContextualCompressionRetrieverService(BaseRetrieverService):

    async def retrieve(self, state:State) -> dict:
        """ ContextualCompressionRetriever 에 의한 문맥 검색 """
        # base_compressor = LLMChainExtractor.from_llm(self.llm)
        llm = get_llm()
        base_compressor = LLMChainFilter.from_llm(llm)
        base_retriever = MultiQueryRetriever.from_llm(retriever=self.vector_store.as_retriever(search_kwargs={"k":7}), llm=llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=base_compressor, base_retriever=base_retriever
        )
        retrieved_docs = await compression_retriever.ainvoke(state["question"])
        return {"context": retrieved_docs}