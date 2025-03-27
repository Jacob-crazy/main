# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/27 10:59
@Author  : Jacob
@Site    : 
@File    : 02-LlamaIndex_更换嵌入模型.py
@Software: PyCharm
@Description: 
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

documents = SimpleDirectoryReader(input_dir='文案/黑悟空/设定.txt', recursive=True).load_data()

index = VectorStoreIndex.from_documents(documents, embed_modek=embedding)

query_engine = index.as_query_engine()

print(query_engine.query('黑神话悟空中有什么战斗工具？'))
