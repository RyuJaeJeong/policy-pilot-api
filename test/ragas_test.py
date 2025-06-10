from datasets import load_dataset
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, SemanticSimilarity, ResponseRelevancy, LLMContextPrecisionWithoutReference
import requests
import pandas as pd
import logging

load_dotenv(override=True)
dataset = load_dataset("json", data_files="./static/poc_test_data.json")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def get_answer(thread_id:str, query:str):
    url = "http://127.0.0.1:8000/chat/completion"
    params = {
        "thred_id":thread_id,
        "query":query,
    }
    response = requests.get(url, params=params)
    return response.json()
arr = []
for i, row in enumerate(dataset["train"]):
    result = get_answer("test001", row["question"])
    arr.append({
        "user_input": row["question"],
        "retrieved_contexts":[str['page_content'] for str in result["data"]["context"]],
        "response":result["data"]["answer"]["content"],
        "reference":row["ground_truth"]
    })
    print("■"*(i+1), end="")
    print("□"*(len(dataset["train"])-(i+1)))

try:
    evaluation_dataset = EvaluationDataset.from_list(arr)
    result = evaluate(dataset=evaluation_dataset,
                      metrics=[
                          Faithfulness(),
                          SemanticSimilarity(),
                          ResponseRelevancy(),
                          LLMContextPrecisionWithoutReference()
                      ],
                      embeddings=embeddings,
                      llm=LangchainLLMWrapper(llm))
    print(result)
    ragas_result_df = result.to_pandas()
    ragas_result_df.to_csv("ragas_text.csv")
except Exception as e:
    print(e)