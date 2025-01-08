from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Queries OpenAI's GPT model to generate a response based on the given input query.
def gpt(query):
    from openai import OpenAI
    client = OpenAI(api_key = api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": query
            }
        ]
    )

    return completion.choices[0].message.content


# Builds a Milvus vector store index from documents in a specified directory and returns the index.
def get_index(db_path = './milvus_demo.db'):
    import openai

    openai.api_key = api_key
    
    import os
    from tqdm import tqdm
    
    parent_dir = 'graphrag/cve1000/input/'
    text_fnames = os.listdir(parent_dir)
    
    contents = []
    for text_fname in tqdm(text_fnames):
        with open(parent_dir + text_fname) as f:
            content = f.read()
        contents.append(content)
    
    import pandas as pd
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Document
    

    documents = [
        Document(text=content)
        for content in contents
    ]
    
       
    splitter = SentenceSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    
    
    vector_store = MilvusVectorStore(
        uri=db_path, dim=1536, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    documents = [Document(text=node.text) for node in nodes]
    
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index


# Uses a vector-based query engine to retrieve relevant information from a given index.
def vectorrag(query, index):
    from llama_index.llms.openai import OpenAI   
    import openai

    llm = OpenAI(model="gpt-4o-mini")
    openai.api_key = api_key

    query_engine = index.as_query_engine(similarity_top_k=10, verbose=True, llm=llm)
    res = query_engine.query(query)

    return res.response

# Executes a subprocess to run a graphrag query with the specified method and root path, then returns the response.
def graphrag(query, query_type):
    import subprocess
    import os 
    os.chdir("/data/llamaindex/graphrag")

    command = [
        "graphrag",
        "query",
        "--root", "./cve1000",
        "--method", query_type,
        "--query", query
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        response = result.stdout
        return response.split("Search Response:")[1]
    else:
        return result.stderr    

# Combines results from vectorrag and graphrag, then generates a hybrid response using GPT by combining both data sources.
def hybridrag(query, query_type, index):

    vector_res = vectorrag(query, index)
    graph_res = graphrag(query, query_type)

    hybrid_query = f"""
        Answer query based on vectorag, graphrag information, utilizing the full potential of each."
        <query>{query}/<query>
        <vectorrag>{vector_res}</vectorrag>
        <graphrag>{graph_res}</graphrag>
        """

    hybrid_res = gpt(hybrid_query)
    return vector_res, graph_res, hybrid_res
