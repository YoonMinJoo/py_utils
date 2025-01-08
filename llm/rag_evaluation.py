from pymilvus import model, MilvusClient, DataType
from tqdm import tqdm
import pandas as pd
import yaml

from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
from nltk.translate.meteor_score import meteor_score

import json

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualRecallMetric

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

import os

os.environ["OPENAI_API_KEY"] = api_key

milvus_client = MilvusClient(uri="http://192.168.2.195:19530")

sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2', # Specify the model name
    device='cuda:0' # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

def emb_text(texts):
    return sentence_transformer_ef(texts)


def search_context(question, milvus_client, collection_name, limit=3):
    milvus_client.load_collection(collection_name)
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(question)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=limit,  # Return top 3 results
        search_params={"metric_type": "COSINE"},  # Ensure Inner Product is set for search
        output_fields=["text"],  # Return the text field
    )
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # context = "\n".join(
    #     [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    # )
    
    return [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]


def rag_inference(user_prompt, model_type="gpt-4o-mini", temperature=0.05):
    
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions referenced from the contextual passage snippets provided.
    Answer based on the terminology used in the question and context, if possible.
    """
    

    from openai import OpenAI

    openai_client = OpenAI()
    
    response = openai_client.chat.completions.create(
        # model="gpt-4o-mini",
        # model="gpt-3.5-turbo",
        model=model_type,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature
    )
    
    
    return response.choices[0].message.content


def measure_bleu(answer, ground_truth):
    answer, ground_truth = answer.split(), ground_truth.split()
    score = sentence_bleu([answer], ground_truth, weights=(1,0,0,0))
    return round(score, 3)


def measure_rouge(answer, ground_truth):
    scorer = RougeScorer(['rouge1'], use_stemmer=True)
    score = scorer.score(answer, ground_truth)['rouge1'].fmeasure
    return round(score, 3)


def measure_relevancy(questions, answers, ground_truths, contexts):
    model_type = "gpt-4o-mini"
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.9,
        model=model_type,
        include_reason=False
    )
    
    context_relevancy = ContextualRecallMetric(
        threshold=0.8,
        model=model_type,
        include_reason=True
    )

    result = evaluate([LLMTestCase(
                        input=questions[i],
                        actual_output=answers[i],
                        expected_output=ground_truths[i],
                        retrieval_context=contexts[i]
                      ) for i in range(len(questions))], [answer_relevancy, context_relevancy])
    return result
