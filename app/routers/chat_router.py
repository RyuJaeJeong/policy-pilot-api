from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler
from ..schemas.chat_schema import State, ChatResponse
from ..services.contextual_compression_retriever_service import ContextualCompressionRetrieverService
from typing import Union
from ..utils.llm_model import get_llm
import logging
import uuid
import os

# field
router = APIRouter(prefix="/chat", tags=["chat"])
langfuse_host = os.environ["LANGFUSE_HOST"]
langfuse_public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
langfuse_secret_key = os.environ["LANGFUSE_SECRET_KEY"]
langfuse_handler = CallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host=langfuse_host
)
service = ContextualCompressionRetrieverService(llm=get_llm(), col_nm="vine_policies_20250605_001")
app = service.build_workflow()


# functions
@router.get("/completion")
async def completion(thread_id: Union[str, None] = None, query: Union[str, None] = None) -> ChatResponse:
    """ 완성형 채팅 반환 """
    try:
        if not query:
            logging.error("=========== 질문이 없습니다. ===========")
            return ChatResponse(code=400, msg="질문이 없습니다", data=None)
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        state = State(question=query, context=None, answer=None)
        config = RunnableConfig(configurable={"thread_id": thread_id}, callbacks=[langfuse_handler])
        output = await app.ainvoke(state, config)
        return ChatResponse(code=200, msg="성공입니다", data=output)
    except Exception as e:
        logging.error(f"에러 발생 : {e}")
        return ChatResponse(code=500, msg="서버 내부 오류", data=None)


@router.get("/streaming")
async def streaming(thread_id: Union[str, None] = None, query: Union[str, None] = None) -> StreamingResponse:
    """ 스트리밍 채팅 반환 """
    try:
        if not query:
            logging.error("=========== 질문이 없습니다. ===========")
            raise ValueError("질문이 없습니다")
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        state = State(question=query, context=None, answer=None)
        config = RunnableConfig(configurable={"thread_id": thread_id}, callbacks=[langfuse_handler])
        return StreamingResponse(generate_stream(state, config), media_type="text/event-stream")
    except Exception as e:
        logging.error(f"에러 발생 : {e}")
        return StreamingResponse(error_stream(e), media_type="text/event-stream")


async def generate_stream(state:State, config:RunnableConfig):
    async for chunk, metadata in app.astream(state, config, stream_mode="messages"):
        if metadata['langgraph_node'] == "generate" and isinstance(chunk, AIMessage):
            yield f"{chunk.content}"


async def error_stream(e):
    yield f"data: [ERROR] {str(e)}\n\n"
    yield f"data: [DONE]\n\n"
