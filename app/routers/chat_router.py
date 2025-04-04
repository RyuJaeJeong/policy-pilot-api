from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Union
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from icecream import ic
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
    return { "messages" : response}  

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
    config = {"configurable": {"thread_id" : thread_id}}
    output = await app.ainvoke({"messages": query}, config)
    messages = [{"type": obj.type, "content":obj.content} for obj in output['messages']]
    result = {
        "code":200, 
        "msg": "성공입니다.",
        "data": { "thread_id": thread_id, "messages": messages},
    }       
    return result

async def generate_stream(config, query):
    async for chunk, metadata in app.astream({"messages": query}, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage): 
           yield f"{chunk.content}\n"


"""
 스트리밍 채팅 반환 
 
 DX사업팀 류재정 프로
 
 @param thread_id: 채팅방 아이디, query: 질의
 @return 답변 텍스트
"""
@router.get("/streaming")
async def streaming(thread_id: Union[str, None] = None, query: Union[str, None] = None):
    if thread_id == None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id" : thread_id}}
    return StreamingResponse(generate_stream(config, query), media_type="text/plain")     # sse 형식의 경우, text/event-stream
    