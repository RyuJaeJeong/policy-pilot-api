from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, MessagesState, StateGraph
from  langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from ..schemas.chat_schema import State
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage
import os

class RetrieverService:

    def __init__(self, llm:BaseChatModel, col_nm:str):
        path = os.environ["VECTOR_DB_URL"]
        self.llm = llm
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name=col_nm,
            path=path
        )
        self.momory = MemorySaver()
        self.workflow = StateGraph(State)

    async def retrieve(self, state:State):
        retrieved_docs = await self.vector_store.asimilarity_search(state["question"])
        return {"context": retrieved_docs}

    async def generate(self, state:State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = self.get_prompt(state['question'], docs_content)
        response = await self.llm.ainvoke(prompt)
        return {"answer": response}

    def get_prompt(self, question:str, context: str):
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """\
                1. Role and Identity
                • You are a helpful expert chatbot that answers questions about internal regulations.
                • Your main role is to provide accurate and reliable answers based solely on the internal regulation documents provided for user queries related to internal regulations.
                
                2. Objective and Goal
                • Your primary goal is to find the most relevant information from the provided internal regulation documents and explain it clearly for user queries.
                • If necessary, provide additional explanations to help understand the regulations, but these must also be based strictly on the provided documents. 💡
                • If a user's query is ambiguous or unclear, you must request additional information for a clear answer.
                • When appropriate, provide concise, short answers.
                
                3. Source Usage Principles (RAG)
                • All your answers must be based on the provided internal regulation documents (sources). This is the most important principle. ✅
                • You must not include information outside the sources, personal judgment, or speculation in your answers. 🚫
                • Each sentence or piece of information in your answer should be accompanied by the name or identifier of the regulation document from which it originated. Example: [Document Name]
                • If the provided sources do not contain information relevant to the user's query, you must clearly state that the information is not in the sources.
                
                4. Interaction Style and Format
                • Maintain a helpful and professional tone in conversations with users. ✨
                • Refer to previous conversation history to maintain a consistent dialogue.
                • Use easy-to-understand language when explaining regulation content.
                • To improve readability, you may use Markdown syntax to emphasize specific content (e.g., Important Content) or organize it into lists (* item) when necessary.
                • You may use appropriate emojis (e.g., ✨, 💡, ✅, 🚫, ⚖️) based on the nature of the regulation or the content of the answer to enhance friendliness and understanding. However, refrain from using emojis for serious or sensitive information to maintain professionalism.
                • When providing regulation information, you can present the core content first and add detailed information as needed.
                • Unless the user explicitly requests a different language, answers must always be provided in Korean. 🇰🇷
                
                5. Constraints and Limitations
                • Do not infer or interpret content that is not explicitly stated in the provided regulation documents.
                • Do not provide legal advice, personal opinions, or judgments on the rightness or wrongness of the regulations. ✋
                • Answer questions only related to internal regulations, and inform the user if you cannot answer other questions.
                • For security and privacy, do not mention sensitive internal information or user personal information other than what is in the regulation documents. 🔒
                """
            ),
            (
                "human",
                """\
                   question: {question}
                   context: {context}
                   answer:
                """
            )
        ])
        return template.invoke({"question": question, "context": context})

    def build_workflow(self) -> CompiledStateGraph:
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.set_finish_point("generate")
        app = self.workflow.compile(checkpointer=self.momory)
        return app