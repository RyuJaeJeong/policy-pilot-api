from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from app.schemas.state import State



chat_template = ChatPromptTemplate.from_messages([
    ("system", "test : {test_param}")
])


c_template = chat_template.invoke({"test_param", "222"})

print(c_template.to_messages())
print(type(c_template.to_messages()))