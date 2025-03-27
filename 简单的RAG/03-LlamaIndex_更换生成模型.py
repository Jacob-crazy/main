# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/27 11:48
@Author  : Jacob
@Site    : 
@File    : 03-LlamaIndex_更换生成模型.py
@Software: PyCharm
@Description: 
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek

from dotenv import load_dotenv
import os

load_dotenv()
embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
llm = DeepSeek(model="deepseek-reasoner", api_key=os.getenv("DEEPSEEK_API_KEY"))

documents = SimpleDirectoryReader(input_files=[r'E:\Jacob_zb\LLM\AI_Agent\文案\黑悟空\设定.txt']).load_data()

index = VectorStoreIndex.from_documents(documents, embed_model=embedding)

query_engine = index.as_query_engine(llm=llm)

print(query_engine.query("黑神话悟空中有什么战斗工具？"))
