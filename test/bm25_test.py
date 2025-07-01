from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import json


kiwi = Kiwi()
stopwords = Stopwords()

def korean_tokenizer(text:str)->list[str]:
    tokens = [token.form for token in kiwi.tokenize(text, normalize_coda=True, stopwords=stopwords)]
    return tokens

file_path = "./output.json"
with open(file_path, "r", encoding="utf-8") as file:
    arr = json.load(file)

docs = [Document(page_content=f"""제 {row["장"]} 장\n제 {row["조"]} 조 [{row["조제목"]}]\n{row["내용"]}""", metadata=row) for row in arr]


# retriever = BM25Retriever.from_documents(docs, preprocess_func=korean_tokenizer)
retriever = BM25Retriever.from_documents(docs)
result = retriever.invoke("수습기간", kwargs={"k":2})
print(result)

