from datasets import load_dataset
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics import Faithfulness, SemanticSimilarity, ResponseRelevancy, LLMContextPrecisionWithoutReference
import requests
import asyncio

load_dotenv(override=True)
dataset = load_dataset("json", data_files="./static/poc_test_data.json")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def get_answer(thread_id:str, query:str):
    url = "http://127.0.0.1:8000/chat/completion"
    params = {
        "thread_id":thread_id,
        "query":query,
    }
    response = requests.get(url, params=params)
    return response.json()

async def adapt_prompt(relavancy):
    adapted_prompts = await relevancy.adapt_prompts(language="korean", llm=llm_factory())
    return adapted_prompts

arr = []
for i, row in enumerate(dataset["train"]):
    result = get_answer("test001", row["question"])
    arr.append({
        "user_input": row["question"],
        "retrieved_contexts":[ctx['page_content'] for ctx in result["data"]["context"]],
        "response":result["data"]["answer"]["content"],
        "reference":row["ground_truth"]
    })
    print("■"*(i+1), end="")
    print("□"*(len(dataset["train"])-(i+1)))

try:
    evaluation_dataset = EvaluationDataset.from_list(arr)
    relevancy = ResponseRelevancy()
    adapted_prompts = asyncio.run(adapt_prompt(relevancy))
    relevancy.set_prompts(**adapted_prompts)
    result = evaluate(dataset=evaluation_dataset,
                      metrics=[
                          Faithfulness(),
                          SemanticSimilarity(),
                          relevancy,
                          LLMContextPrecisionWithoutReference()
                      ],
                      embeddings=embeddings,
                      llm=LangchainLLMWrapper(llm))
    print(result)
    ragas_result_df = result.to_pandas()
    ragas_result_df.to_csv("dense_multi_02.csv")
except Exception as e:
    print(e)