# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/27 10:30
@Author  : Jacob
@Site    : 
@File    : 01-LlamaIndex.py
@Software: PyCharm
@Description: 
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["E:\Jacob_zb\LLM\AI_Agent\文案\黑悟空\设定.txt"]).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

print(query_engine.query('黑悟空中有哪些战斗工具？'))
