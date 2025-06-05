from typing_extensions import List, TypedDict
from typing import Union, Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseMessage
from pydantic import BaseModel

class State(TypedDict):
    question: str
    context: Union[List[Document], None]
    answer: Union[BaseMessage, None]

class ChatResponse(BaseModel):
    code: int
    msg: str
    data: Union[dict[str, Any], None]