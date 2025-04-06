"""
# ================================================================================================
# 進階 RAG (檢索增強生成) 系統實現 (Advanced RAG System Implementation)
# ================================================================================================
#
# 此檔案實現了一個進階的RAG（檢索增強生成）系統，主要功能包括：
# This file implements an advanced RAG (Retrieval-Augmented Generation) system with the following main features:
#
# 1. PDF文件處理和文本分割
#    PDF document processing and text splitting
#
# 2. 向量資料庫的建立和查詢
#    Vector database creation and querying
#
# 3. 查詢擴充技術的實現
#    Implementation of query expansion techniques
#
# 4. 使用UMAP進行向量空間的可視化
#    Visualization of vector spaces using UMAP
#
# 主要技術特點：
# Key technical features:
#
# - 使用多層次的文本分割策略
#   Multi-level text splitting strategy
#
# - 整合OpenAI API進行查詢擴充
#   Integration of OpenAI API for query expansion
#
# - 使用ChromaDB進行向量存儲和檢索
#   Using ChromaDB for vector storage and retrieval
#
# - 使用UMAP進行高維向量的降維視覺化
#   Using UMAP for dimensionality reduction and visualization of high-dimensional vectors
#
# 此系統通過查詢擴充技術顯著提高了檢索的準確性和相關性，特別是對於複雜查詢。
# This system significantly improves retrieval accuracy and relevance through query expansion techniques,
# especially for complex queries.
# ================================================================================================
"""

# 導入必要的函式庫和工具
# Import necessary libraries and tools
from helper_utils import project_embeddings, word_wrap  # 導入自定義工具函數 (Import custom utility functions)
from pypdf import PdfReader  # 用於讀取PDF檔案 (For reading PDF files)
import os
from openai import OpenAI  # OpenAI API客戶端 (OpenAI API client)
from dotenv import load_dotenv  # 用於載入環境變數 (For loading environment variables)

from pypdf import PdfReader
import umap  # 用於降維視覺化 (For dimensionality reduction visualization)

# 載入環境變數 (Load environment variables)
load_dotenv()

# 設置OpenAI API金鑰並初始化客戶端 (Set up OpenAI API key and initialize client)
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# 讀取PDF文件並提取文本 (Read PDF document and extract text)
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# 過濾空字串 (Filter empty strings)
pdf_texts = [text for text in pdf_texts if text]

# 列印data文件中，第一個pdf的內容，列印前100個字
# Print the first 100 characters of the first PDF in the data file
# print(
#     word_wrap(
#         pdf_texts[0],
#         width=100,
#     )
# )

# === 文本切分工作 (Text Splitting Work) ===

# 使用兩種不同的文本分割器進行文本切分
# Use two different text splitters for text splitting
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# 使用遞迴字符分割器，按照不同的分隔符號進行分割
# Use recursive character splitter to split text based on different separators
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# 列印character_split_texts的第10個元素，列印前100個字
# Print the 10th element of character_split_texts, first 100 characters
# print(word_wrap(character_split_texts[10]))
# print(f"\nTotal chunks: {len(character_split_texts)}")

# === 使用基於SentenceTransformers的文本分割器，按照256個token進行分割 ===
# === Use SentenceTransformers-based text splitter to split text by 256 tokens ===
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# 列印token_split_texts的第10個元素，列印前100個字
# Print the 10th element of token_split_texts, first 100 characters
# print(word_wrap(token_split_texts[10]))
# print(f"\nTotal chunks: {len(token_split_texts)}")


# === 創建ChromaDB客戶端和集合 (Create ChromaDB client and collection) ===
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 創建embedding函數實例 (Create embedding function instance)
embedding_function = SentenceTransformerEmbeddingFunction()

# 列印embedding 字詞 (Print embedding words)
# print(embedding_function([token_split_texts[10]]))

# === 創建ChromaDB客戶端和集合，並實行向量檢索 ===
# === Create ChromaDB client and collection, and implement vector retrieval ===

# 創建ChromaDB客戶端和集合 (Create ChromaDB client and collection)
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# 將文本片段加入向量資料庫 (Add text chunks to vector database)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

# 定義查詢問題 (Define query question)
query = "What was the total revenue for the year?"

# 執行向量檢索 (Perform vector retrieval)
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# 列印檢索內容 (Print retrieved content)
# for document in retrieved_documents:
#     print(word_wrap(document))
#     print("\n")

# === 先行使用擴充查詢 (Use query expansion first) === 
# 先讓 AI 生成一個模擬的答案(具有幻覺)，再擴充查詢，提升查詢結果的品質。
# First let AI generate a simulated answer (with hallucinations), then expand the query to improve the quality of query results.

def augment_query_generated(query, model="gpt-4o-mini"):
    """
    使用 OpenAI 的語言模型來生成查詢問題的示例答案
    Use OpenAI's language model to generate example answers to query questions
    
    這個函數的主要目的是通過 AI 生成一個模擬的答案，這個答案可能會出現在財務報告中。
    這種技術可以幫助擴展原始查詢，提高文檔檢索的相關性和準確性。
    The main purpose of this function is to generate a simulated answer through AI, which might appear in financial reports.
    This technique helps expand the original query, improving the relevance and accuracy of document retrieval.
    
    參數 (Parameters):
        query (str): 用戶的原始查詢問題 (The user's original query question)
        model (str): 使用的 OpenAI 模型名稱，預設使用 'gpt-4o-mini' (The OpenAI model name used, default is 'gpt-4o-mini')
        
    返回 (Returns):
        str: AI 生成的示例答案 (Example answer generated by AI)
    """
    # 設定系統提示，定義 AI 助手的角色和任務
    # Set system prompt, define the role and task of the AI assistant
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    
    # 構建發送給 OpenAI API 的消息列表
    # Build a message list to send to the OpenAI API
    messages = [
        {
            "role": "system",  # 系統角色：設定 AI 的行為和身份 (System role: set AI behavior and identity)
            "content": prompt,
        },
        {"role": "user",      # 用戶角色：包含實際的查詢內容 (User role: contains the actual query content)
         "content": query},
    ]

    # 調用 OpenAI API 生成回應
    # Call OpenAI API to generate response
    response = client.chat.completions.create(
        model=model,          # 使用指定的語言模型 (Use the specified language model)
        messages=messages,     # 傳入準備好的消息列表 (Pass in the prepared message list)
    )
    
    # 從回應中提取生成的內容
    # Extract generated content from the response
    content = response.choices[0].message.content
    return content

# 設定原始查詢並生成擴充查詢
# Set original query and generate expanded query
"""
查詢擴充（Query Expansion）的實現：
Implementation of Query Expansion:

這是一種進階的RAG技術，通過以下步驟提高檢索準確性：
This is an advanced RAG technique that improves retrieval accuracy through the following steps:

1. 首先設定一個原始查詢（original_query）
   First set an original query (original_query)
   
2. 使用AI生成一個假設性答案（hypothetical_answer）
   Use AI to generate a hypothetical answer (hypothetical_answer)
   
3. 將兩者結合形成更豐富的查詢（joint_query）
   Combine both to form a richer query (joint_query)

這種方法的優點：
Advantages of this method:

- 通過AI生成的假設性答案，擴充了原始查詢的語義信息
  Through AI-generated hypothetical answers, the semantic information of the original query is expanded
  
- 幫助系統更好地理解查詢意圖
  Helps the system better understand query intent
  
- 增加了相關關鍵詞，提高檢索相關性
  Adds relevant keywords, improving retrieval relevance
"""

# 原始查詢：詢問年度利潤及其與去年的比較
# Original query: Ask about annual profit and its comparison with last year
original_query = "What was the total profit for the year, and how does it compare to the previous year?"

# 使用AI生成假設性答案，這個答案模擬了可能在年報中出現的回答格式和內容
# 注意：這是AI生成的模擬答案，不是實際數據
# Use AI to generate a hypothetical answer, which simulates the answer format and content that might appear in an annual report
# Note: This is an AI-generated simulated answer, not actual data
hypothetical_answer = augment_query_generated(original_query)

# 將原始查詢和假設性答案組合在一起，形成更豐富的查詢語句
# 這樣的組合可以幫助系統找到更相關的文檔片段
# Combine the original query and hypothetical answer to form a richer query statement
# Such a combination can help the system find more relevant document fragments
joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))

# 使用擴充後的查詢進行檢索
# Use expanded query for retrieval
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

# 列印 擴充查詢 的 檢索結果
# Print retrieval results of expanded query
# for doc in retrieved_documents:
#     print(word_wrap(doc))
#     print("")

# 獲取 擴充查詢 的 檢索結果、原始查詢和擴充查詢的embeddings
# Get retrieval results, original query and expanded query embeddings
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

# 使用UMAP進行降維投影
# Use UMAP for dimensionality reduction projection
projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

# 使用matplotlib進行視覺化
# Use matplotlib for visualization
import matplotlib.pyplot as plt

# 繪製投影後的查詢和檢索文檔在embedding空間中的分布
# Plot the distribution of projected queries and retrieved documents in embedding space
plt.figure()

# 繪製所有文檔的embeddings（灰色點）
# Plot embeddings of all documents (gray dots)
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)

# 繪製檢索到的文檔（綠色圓圈）
# Plot retrieved documents (green circles)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)

# 繪製原始查詢（紅色X）
# Plot original query (red X)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)

# 繪製擴充查詢（橙色X）
# Plot expanded query (orange X)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

# 設置圖表屬性
# Set chart properties
plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # 顯示圖表 (Display chart)



