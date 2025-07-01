from langchain_core.prompt_values import ChatPromptValue
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from ..schemas.chat_schema import State
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage
from langsmith import Client
from qdrant_client import QdrantClient
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
import os
from icecream import ic


class BaseRetrieverService:

    def __init__(self, llm:BaseChatModel, col_nm:str, memory):
        path = os.environ["VECTOR_DB_URL"]
        self.llm = llm
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name=col_nm,
            path=path
        )
        self.memory = memory
        self.workflow = StateGraph(State)
        self.langsmith_client = Client()

    async def retrieve(self, state:State) -> dict:
        """ vector store 검색기에 의한 문맥 검색 """
        retrieved_docs = await self.vector_store.asimilarity_search(state["question"], k=4)
        return {"context": retrieved_docs}

    async def generate(self, state:State) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = self.get_prompt(state['question'], docs_content)
        response = await self.llm.ainvoke(prompt)
        return {"answer": response}

    def get_prompt(self, question: str, context: str) -> ChatPromptValue:
        template = self.langsmith_client.pull_prompt("policy-pilot-api-001", include_model=False)
        return template.invoke({"question": question, "context": context})

    def build_workflow(self) -> CompiledStateGraph:
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.set_finish_point("generate")
        app = self.workflow.compile(checkpointer=self.memory)
        return app