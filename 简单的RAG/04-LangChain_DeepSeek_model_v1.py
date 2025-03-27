# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/27 16:58
@Author  : Jacob
@Site    : 
@File    : 04-LangChain_DeepSeek_model_v1.py
@Software: PyCharm
@Description: 
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

loader = WebBaseLoader(
    web_paths=(r"https://baike.baidu.com/item/%E9%BB%91%E7%A5%9E%E8%AF%9D%EF%BC%9A%E6%82%9F%E7%A9%BA/53303078",))

docs = loader.load()

# 分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitters.split_documents(docs)

# 嵌入
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"},
                                   encode_kwargs={"normalize_embeddings": True})
# 创建向量
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 用户查询
questions = '黑悟空有哪些游戏场景?'

# 搜索存储中相关文档，并准备上下内容
retrieve_docs = vector_store.similarity_search(questions, k=3)
docs_content = "\n\n".join(doc.paragraphs for doc in retrieve_docs)

from langchain_core.prompts import ChatPromptTemplate

# 提示词模板
prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，请说"我无法从提供的上下文中找到相关信息"。
                上下文：{context}
                问题：{question}
                回答：""")

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model='deepseek-chat',
    template=0.7,
    max_tokens=1024,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

answer = llm.invoke(prompt.format(questions=questions, context=docs_content))
print(answer)
