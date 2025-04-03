from fastapi import APIRouter
from typing import Union
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from icecream import ic
import os
import uuid

router = APIRouter()
model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

"""
   ************************************
   graph 구성 start
   ************************************
"""

workflow = StateGraph(state_schema=MessagesState)

# 모델 호출 함수    
async def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
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

@router.get("/chat")
async def chat(thread_id: Union[str, None] = None, query: Union[str, None] = None):
    config = {"configurable" : {"thread_id" : thread_id}}
    if thread_id == None:
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id" : thread_id}}
    output = await app.ainvoke({"messages": query}, config)
    print(output)
    print(type(output))
    output["thread_id"] = thread_id
    return output    
    