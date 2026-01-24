import logging
import os
import json
from typing import Union

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, message_to_dict
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from app.services.chat.contextual_compression_retriever_service import ContextualCompressionRetrieverService
from ..schemas.chat_schema import State
from ..utils.vdb_con import get_vector_store


# field
langfuse_handler = CallbackHandler(
    host=os.environ["LANGFUSE_HOST"],
    public_key= os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"]
)

checkpointer = InMemorySaver()
col_nm = "VC_M_01"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = get_vector_store(col_nm, embeddings)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/streaming")
async def streaming(thread_id: Union[str, None] = None, query: Union[str, None] = None) -> StreamingResponse:
    """ 스트리밍 채팅 반환 """
    try:
        if not query:
            logging.error("No query, No response")
            raise ValueError("query can not be null")
        if not thread_id:
            logging.error("No query, No response")
            raise ValueError("tread_id can not be null")

        async def generate_stream():
            service = ContextualCompressionRetrieverService(vector_store=vector_store, memory=checkpointer)
            app = service.build_workflow()
            state = State(question=query, context=None, answer=None)
            config = RunnableConfig(configurable={"thread_id": thread_id}, callbacks=[langfuse_handler])
            async for chunk, metadata in app.astream(state, config, stream_mode="messages"):
                if metadata['langgraph_node'] == "generate" and isinstance(chunk, AIMessage):
                    chunk_json = chunk.model_dump_json()
                    yield f"data: {chunk_json}\n\n"
            yield "[DONE]"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    except Exception as e:
        logging.error(f"에러 발생 : {e}")
        return StreamingResponse(error_stream(e), media_type="text/event-stream")


async def error_stream(e):
    yield f"data: [ERROR] {str(e)}\n\n"
    yield f"data: [DONE]\n\n"
