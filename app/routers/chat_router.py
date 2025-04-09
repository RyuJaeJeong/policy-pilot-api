from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from typing import Union
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from icecream import ic
from ..utils.vector_db import get_collection 
from ..services.retriever_service import RetrieverService
from ..schemas.state import State
import os
import uuid
import asyncio

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

"""
   ************************************
   graph 구성 start
   ************************************
"""

workflow = StateGraph(state_schema=MessagesState)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 AI Chatbot입니다, 사용자의 질문에 간략하게 답변하세요.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# 모델 호출 함수    
async def call_model(state: MessagesState):
    prompt = await prompt_template.ainvoke(state)
    ic(prompt)
    response = await model.ainvoke(prompt)
    return { "messages" : response }  

# 그래프 구성
workflow.add_node("model", call_model)
workflow.add_edge(START, "model") 

# 메모리 추가, 앱 구성
memory = MemorySaver()
app = workflow.compile(checkpointer=MemorySaver())

"""
   ************************************
   graph 구성 end
   ************************************
"""

"""
 완성 채팅 반환 
 
 DX사업팀 류재정 프로
 
 @param thread_id: 채팅방 아이디, query: 질의
 @return result.code: 상태코드, result.msg: 메세지, result.data: 채팅방아이디, 생성 텍스트
"""
@router.get("/completion")
async def completion(thread_id: Union[str, None] = None, query: Union[str, None] = None):
    if thread_id == None:
        thread_id = str(uuid.uuid4())
        
    state = State(question=query)    
    service = RetrieverService(state=state)    
    config = {"configurable": {"thread_id" : thread_id}}
    output = await service.app.ainvoke(state, config)
    ic(output)
    result = {
        "code":200, 
        "msg": "성공입니다.",
        "data": output,
    }       
    return result




"""
 스트리밍 채팅 반환 
 
 DX사업팀 류재정 프로
 
 @param thread_id: 채팅방 아이디, query: 질의
 @return 답변 텍스트
"""

async def generate_stream(app, config, state):
    async for chunk, metadata in app.astream(state, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage): 
           print(chunk.content) 
           yield f"{chunk.content}"

@router.get("/streaming")
async def streaming(thread_id: Union[str, None] = None, query: Union[str, None] = None):
    if thread_id == None:
        thread_id = str(uuid.uuid4())
    state = State(question=query)    
    service = RetrieverService(state=state)
    config = {"configurable": {"thread_id" : thread_id}}
    return StreamingResponse(generate_stream(service.app, config, state), media_type="text/event-stream")     # sse 형식의 경우, text/event-stream

# @router.get("/ask")
# async def ask(thread_id: Union[str, None] = None, query: Union[str, None] = None):
#     if thread_id == None:
#         thread_id = str(uuid.uuid4())
#     state = State(question=query)
#     service = RetrieverService(state=state)
#     config = {"configurable": {"thread_id" : thread_id}}
#     result = await service.app.ainvoke(state, config)
#     return result
    