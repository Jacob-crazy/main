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
llm = DeepSeek(model="deepseek-reasoner", api_key=os.getenv("DEEPSEEK_API_KEY"))


