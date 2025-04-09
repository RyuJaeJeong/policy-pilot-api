from ..schemas.state import State
from ..utils.vector_db import get_collection 
from ..utils.llm_model import get_llm
from icecream import ic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

memory = MemorySaver()
        

class RetrieverService:
    
    def __init__(self, state:State):
        self.vector_store = get_collection(col_name="cp_20250408")
        self.llm = get_llm()
        self.template = ChatPromptTemplate.from_messages([
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
                """question: {question}
                   context: {context}
                   answer:
                """
            ) 
        ])
        workflow = StateGraph(State)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.set_finish_point("generate")
        app = workflow.compile(checkpointer=memory)
        self.app = app
        
        
    def retrieve(self, state:State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state:State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = self.get_prompt(state, docs_content)
        response = self.llm.invoke(prompt)
        return {"answer": response}
    
    def get_prompt(self, state:State, context:str):
        return self.template.invoke({"question": state["question"], "context": context})