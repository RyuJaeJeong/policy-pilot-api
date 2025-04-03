from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")
messages = [
    SystemMessage(content="당신은 ai 챗봇입니다. 사용자의 질문에 적절하게 응답하세요."),
    HumanMessage(content="hi!"),
]
result = model.invoke(messages)
print(result)