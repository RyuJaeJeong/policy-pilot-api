from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, MessagesState, StateGraph
from  langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from ..schemas.chat_schema import State
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage
import os

class RetrieverService:

    def __init__(self, llm:BaseChatModel, col_nm:str):
        path = os.environ["VECTOR_DB_URL"]
        self.llm = llm
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name=col_nm,
            path=path
        )
        self.momory = MemorySaver()
        self.workflow = StateGraph(State)

    async def retrieve(self, state:State):
        retrieved_docs = await self.vector_store.asimilarity_search(state["question"])
        return {"context": retrieved_docs}

    async def generate(self, state:State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = self.get_prompt(state['question'], docs_content)
        response = await self.llm.ainvoke(prompt)
        return {"answer": response}

    def get_prompt(self, question:str, context: str):
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """당신은 사내 규정 전문가입니다. 다음 규정을 참고해 질문에 답변하세요:

                [응답 규칙]
                1. 반드시 한국어로 답변
                2. 답변 시작에 [제{{장}}장 제{{조}}조] 형식으로 출처 명시(예를들어 제8장 제56조)
                3. 규정 본문을 간결하게 요약한 후 사용자 이해가 필요하다고 생각되면 설명 추가
                4. 읽기 쉽게 markdown 형식으로, 적절한 이모티콘 추가 📝 (예: **강조**, 목록)
                5. 모르는 사항은 '제공된 PDF에서 직접적인 정보는 찾을 수 없습니다.' 라고 답변과 함께 이전 대화 기록 참고해서 답변을 간단하게 생성해줘. 
                6. 표인 경우 표 형태 유지 (예: |항목|내용|)
                7. 문서 중에 관련이 없다고 생각되면 답변 생성시에 활용하지 않아도 돼
                8. 규정과 관련없는 질문의 경우 \' 확인 할 수 없는 사항입니다. \' 출력 할 것
                """
            ),
            (
                "human",
                """\
                   question: {question}
                   context: {context}
                   answer:
                """
            )
        ])
        return template.invoke({"question": question, "context": context})

    def build_workflow(self) -> CompiledStateGraph:
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.set_finish_point("generate")
        app = self.workflow.compile(checkpointer=self.momory)
        return app