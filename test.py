from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from icecream import ic
import bs4


load_dotenv()

llm = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = QdrantClient(":memory:")

collections = client.get_collections().collections
collection_names = [col.name for col in collections]
ic(collection_names)

if "test2" not in collection_names:
    client.create_collection(
        collection_name="test2",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test2",
    embedding=embeddings,
)

    

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 그래프 빌더 생성
graph_builder = StateGraph(State)

# 각 노드를 명시적으로 추가
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

# 노드 간 연결 정의
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")

# 끝 지점 정의
graph_builder.set_finish_point("generate")

# 컴파일
graph = graph_builder.compile()
result = graph.invoke({"question": "What is ReAct?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')