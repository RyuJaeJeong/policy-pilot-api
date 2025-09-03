import os
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client
from app.schemas.chat_schema import State
from app.utils.llm_model import get_llm
from icecream import ic


class BaseRetrieverService:

    def __init__(self, vector_store: QdrantVectorStore, memory):
        self.vector_store = vector_store
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
        llm = get_llm()
        ic(prompt)
        response = await llm.ainvoke(prompt)
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