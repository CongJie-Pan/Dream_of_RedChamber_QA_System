# First to let LLM to generate multiple queries from the original query, and 
# the queries transfer to the vector database to get the most relevant documents
# Final to use the LLM to generate the  final answer. 
# ================================
"""
# ========================================================================
# 檔案名稱: origin_expansion_queries.py
# 
# 程式任務說明:
# 此程式實現了進階檢索增強生成(RAG)技術中的查詢擴展(Query Expansion)方法。
# 主要功能包括:
# 1. 讀取並處理PDF文件(微軟年度報告)
# 2. 將文本分割成適當大小的片段
# 3. 使用SentenceTransformer生成文本嵌入
# 4. 創建向量數據庫(ChromaDB)存儲文檔嵌入
# 5. 實現多查詢生成(Multi-Query)技術，使用OpenAI模型擴展原始查詢
# 6. 使用擴展查詢進行檢索，提高檢索結果的相關性和覆蓋率
# 7. 可視化查詢和檢索結果在嵌入空間中的分布
#
# This program implements Query Expansion method in Advanced Retrieval Augmented Generation (RAG).
# Main functionalities include:
# 1. Reading and processing PDF documents (Microsoft annual report)
# 2. Splitting text into appropriate chunks
# 3. Generating text embeddings using SentenceTransformer
# 4. Creating a vector database (ChromaDB) to store document embeddings
# 5. Implementing Multi-Query technique using OpenAI models to expand original queries
# 6. Retrieving documents using expanded queries to improve relevance and coverage
# 7. Visualizing queries and retrieved results in the embedding space
# ========================================================================
"""

# 導入必要的庫和工具 (Import necessary libraries and utilities)
from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv


from pypdf import PdfReader
import numpy as np
import umap


# 從.env檔案載入環境變數 (Load environment variables from .env file)
load_dotenv()

# 獲取OpenAI API密鑰並初始化客戶端 (Get OpenAI API key and initialize client)
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# 讀取PDF文件 (Read PDF document)
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# 過濾空字符串 (Filter empty strings)
pdf_texts = [text for text in pdf_texts if text]


# 導入文本分割器 (Import text splitters)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# 使用字符分割器將文本分割成較大的塊 (Use character splitter to split text into larger chunks)
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))


# 使用基於token的分割器進一步分割文本 (Use token-based splitter to further split text)
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# 導入ChromaDB和SentenceTransformer嵌入函數 (Import ChromaDB and SentenceTransformer embedding function)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 初始化嵌入函數 (Initialize embedding function)
embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

# 實例化ChromaDB客戶端並創建集合 (Instantiate ChromaDB client and create collection)
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# 提取文本嵌入並添加到集合中 (Extract text embeddings and add to collection)
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

# 定義查詢 (Define query)
query = "What was the total revenue for the year?"

# 執行查詢並獲取結果 (Execute query and get results)
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# 列印檢索出來的文件
# for document in retrieved_documents:
#     print(word_wrap(document))
#     print("\n")

# ============

# 定義多查詢生成函數 (Define multi-query generation function)
def generate_multi_query(query, model="gpt-4o-mini"):
    """
    使用OpenAI模型生成多個相關查詢，以擴展原始查詢的覆蓋範圍。
    
    參數:
        query (str): 原始查詢
        model (str): 使用的OpenAI模型名稱
        
    返回:
        list: 生成的相關查詢列表
        
    Generate multiple related queries using OpenAI model to expand the coverage of the original query.
    
    Args:
        query (str): Original query
        model (str): Name of the OpenAI model to use
        
    Returns:
        list: List of generated related queries
    """

    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


# 定義原始查詢並生成擴展查詢 (Define original query and generate expanded queries)
original_query = (
    "What details can you provide about the factors that led to revenue growth?"
)
aug_queries = generate_multi_query(original_query)

# 1. 顯示擴展查詢 (Show the augmented queries)
print("Augmented Queries: \n")

for query in aug_queries:
    print("\n", query)

# 2. 將原始查詢與擴展查詢合併 (Concatenate the original query with the augmented queries)
joint_query = [
    original_query
] + aug_queries  # 原始查詢放在列表中，因為chroma可以處理多個查詢 (original query is in a list because chroma can handle multiple queries)


# print("======> \n\n", joint_query)

# 合併查詢檢索並輸出檢索結果工作

# 使用合併查詢進行檢索 (Retrieve documents using joint queries)
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# 對檢索到的文檔進行去重 (Deduplicate the retrieved documents)
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)


# 輸出檢索結果 (Output the retrieved documents)

for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Retrieved documents Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)

#  === 將檢索狀態視覺化 ===

# 獲取嵌入並使用UMAP進行降維 (Get embeddings and use UMAP for dimensionality reduction)
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 4. 在嵌入空間中可視化結果 (Visualize the results in the embedding space)
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)


# 投影查詢嵌入 (Project query embeddings)
project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

# 獲取檢索結果的嵌入 (Get embeddings of retrieved results)
retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

# 導入繪圖庫 (Import plotting library)
import matplotlib.pyplot as plt


# 在嵌入空間中繪製查詢和檢索到的文檔 (Plot the projected query and retrieved documents in the embedding space)
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # 顯示圖表 (display the plot)