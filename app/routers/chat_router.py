from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.mysql.asyncmy import AsyncMySaver
from ..schemas.chat_schema import State
from app.services.chat.contextual_compression_retriever_service import ContextualCompressionRetrieverService
from app.services.embedding.custom_embedding_service import CustomEmbeddingService
from typing import Union
from contextlib import asynccontextmanager
from ..utils.llm_model import get_llm
from ..utils.vdb_con import get_vector_store
import logging
import os
import gc

# field
langfuse_handler = CallbackHandler(
    public_key= os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"]
)

embeddings = None
vector_store = None
col_nm = "VC_M_01"

@asynccontextmanager
async def lifespan(app: APIRouter):
    global embeddings
    global vector_store
    embeddings = CustomEmbeddingService()
    vector_store = get_vector_store(col_nm, embeddings)
    logging.info(f"vs loaded")

    yield

    embeddings = None
    vector_store = None
    logging.info(f"bye bye embedding")
    gc.collect()

DB_URL = os.environ["DB_URL"]
router = APIRouter(prefix="/chat", tags=["chat"], lifespan=lifespan)


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
            async with AsyncMySaver.from_conn_string(DB_URL) as checkpointer:
                service = ContextualCompressionRetrieverService(vector_store=vector_store, memory=checkpointer)
                app = service.build_workflow()
                state = State(question=query, context=None, answer=None)
                config = RunnableConfig(configurable={"thread_id": thread_id}, callbacks=[langfuse_handler])
                async for chunk, metadata in app.astream(state, config, stream_mode="messages"):
                    if metadata['langgraph_node'] == "generate" and isinstance(chunk, AIMessage):
                        logging.info(chunk)
                        yield f"{chunk}"
                yield "[DONE]"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    except Exception as e:
        logging.error(f"에러 발생 : {e}")
        return StreamingResponse(error_stream(e), media_type="text/event-stream")


async def error_stream(e):
    yield f"data: [ERROR] {str(e)}\n\n"
    yield f"data: [DONE]\n\n"
