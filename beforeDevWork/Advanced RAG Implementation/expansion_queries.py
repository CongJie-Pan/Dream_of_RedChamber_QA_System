# First to let LLM to generate multiple queries from the original query, and 
# the queries transfer to the vector database to get the most relevant documents
# Final to use the LLM to generate the  final answer. 
# ================================
"""
# ========================================================================
# 檔案名稱: expansion_queries.py
# 
# 程式任務說明:
# 此程式實現了進階檢索增強生成(RAG)技術中的查詢擴展(Query Expansion)方法。
# 主要功能包括:
# 1. 讀取並處理PDF文件(微軟年度報告)
# 2. 將文本分割成適當大小的片段
# 3. 使用OpenAI API生成文本嵌入
# 4. 創建向量數據庫(ChromaDB)存儲文檔嵌入
# 5. 實現多查詢生成(Multi-Query)技術，使用OpenAI模型擴展原始查詢
# 6. 使用擴展查詢進行檢索，提高檢索結果的相關性和覆蓋率
# 7. 可視化查詢和檢索結果在嵌入空間中的分布
#
# This program implements Query Expansion method in Advanced Retrieval Augmented Generation (RAG).
# Main functionalities include:
# 1. Reading and processing PDF documents (Microsoft annual report)
# 2. Splitting text into appropriate chunks
# 3. Generating text embeddings using OpenAI API
# 4. Creating a vector database (ChromaDB) to store document embeddings
# 5. Implementing Multi-Query technique using OpenAI models to expand original queries
# 6. Retrieving documents using expanded queries to improve relevance and coverage
# 7. Visualizing queries and retrieved results in the embedding space
# ========================================================================
"""

# 導入必要的庫和工具 (Import necessary libraries and utilities)
from helper_utils import project_embeddings, word_wrap, extract_text_from_file
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import numpy as np
import umap
import matplotlib.pyplot as plt

# 設置日誌記錄器
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 從.env檔案載入環境變數 (Load environment variables from .env file)
load_dotenv()

# 導入文本分割器 (Import text splitters)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 獲取OpenAI API密鑰並初始化客戶端 (Get OpenAI API key and initialize client)
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# 導入ChromaDB和嵌入函數 (Import ChromaDB and embedding functions)
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# 導入jieba中文分詞
try:
    import jieba
    JIEBA_AVAILABLE = True
    logger.info("Jieba successfully imported for Chinese word segmentation")
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("Jieba not available. Chinese word segmentation will be limited.")

def process_chinese_text(text, use_segmentation=True):
    """
    Process Chinese text by applying jieba word segmentation and optimized splitting
    
    Args:
        text (str): The text to process
        use_segmentation (bool): Whether to use jieba for segmentation
        
    Returns:
        str: Processed text
    """
    # Check if text contains Chinese characters
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    if not has_chinese:
        return text
    
    logger.info("Processing Chinese text with optimized methods")
    
    # Apply jieba word segmentation if requested
    if use_segmentation and JIEBA_AVAILABLE:
        try:
            logger.info("Using jieba to segment Chinese text")
            # Force loading of user dictionaries if any exist
            user_dict_path = os.path.join(os.path.dirname(__file__), "jieba_userdict.txt")
            if os.path.exists(user_dict_path):
                jieba.load_userdict(user_dict_path)
                logger.info(f"Loaded jieba user dictionary from {user_dict_path}")
            
            # Cut the text with jieba
            words = jieba.cut(text)
            text = " ".join(words)
            logger.info("Text segmented with jieba")
        except Exception as e:
            logger.warning(f"Error during Chinese word segmentation: {str(e)}")
    
    # Apply additional Chinese-specific processing
    
    # 1. Replace traditional Chinese punctuation with spaces for better tokenization
    chinese_punctuation = "，。！？；：""''「」【】《》（）～"
    for char in chinese_punctuation:
        text = text.replace(char, f" {char} ")
    
    # 2. Normalize whitespace
    text = " ".join(text.split())
    
    # 3. Add sentence boundary markers to help with context
    for char in "。！？":
        text = text.replace(f" {char} ", f" {char} [SEP] ")
    
    logger.info(f"Chinese text processed: '{text[:50]}...'")
    return text

def read_pdf(file_path="data/microsoft-annual-report.pdf"):
    """
    讀取PDF文件並提取文本
    
    Args:
        file_path (str): PDF文件路徑
        
    Returns:
        list: 從PDF頁面提取的文本列表
    """
    logger.info(f"Reading document from: {file_path}")
    
    # 使用公共的提取文本函數
    try:
        text, section_count = extract_text_from_file(file_path)
        
        # 將文本按段落分割
        pdf_texts = [t.strip() for t in text.split("\n\n") if t.strip()]

# 過濾空字符串 (Filter empty strings)
pdf_texts = [text for text in pdf_texts if text]
        logger.info(f"Extracted {len(pdf_texts)} text sections")
        return pdf_texts
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def split_text(pdf_texts, chunk_size=1000, debug=False):
    """
    Split text using improved methods for better chunking, especially for Chinese text
    
    Args:
        pdf_texts (list): List of text content from PDF
        chunk_size (int): Size of chunks for character-based splitting
        debug (bool): Whether to print debug information
    
    Returns:
        list: List of text chunks after splitting
    """
    logger.info("Starting text splitting process")
    
    # Check if the text contains Chinese characters
    has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in text) for text in pdf_texts)
    
    # Use different separators and chunk settings based on text language
    if has_chinese:
        logger.info("Chinese text detected, using optimized splitting for Chinese")
        # Chinese text needs different separators that respect Chinese punctuation
        separators = ["。", "！", "？", "\n\n", "\n", ". ", " ", ""]
        # Use smaller chunks for Chinese text for better coherence
        actual_chunk_size = min(chunk_size, 500)
        # Add overlap to maintain context across chunks
        chunk_overlap = 50
        
        # Use jieba for Chinese word segmentation if Chinese text is detected
        if JIEBA_AVAILABLE:
            logger.info("Using jieba for Chinese word segmentation")
            
            # Process each text chunk with jieba
            segmented_texts = []
            for text in pdf_texts:
                if any('\u4e00' <= char <= '\u9fff' for char in text):
                    # Only apply jieba to text containing Chinese characters
                    words = jieba.cut(text)
                    segmented_text = " ".join(words)
                    segmented_texts.append(segmented_text)
                else:
                    segmented_texts.append(text)
            
            # Replace the original texts with segmented ones
            pdf_texts = segmented_texts
            
            if debug:
                print("\nSegmented Chinese text sample:")
                print(word_wrap(pdf_texts[0][:200]))
        else:
            logger.warning("Jieba not available. Chinese word segmentation skipped.")
    else:
        logger.info("Non-Chinese text detected, using standard separators")
        separators = ["\n\n", "\n", ". ", " ", ""]
        actual_chunk_size = chunk_size
        chunk_overlap = 20
    
    # Use recursive character splitter with appropriate separators
    character_splitter = RecursiveCharacterTextSplitter(
        separators=separators, 
        chunk_size=actual_chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Split the text using the character splitter
    text_chunks = character_splitter.split_text("\n\n".join(pdf_texts))
    
    if debug:
        print(f"\nAfter splitting - Total chunks: {len(text_chunks)}")
        print("Sample chunk (character split):")
        print(word_wrap(text_chunks[min(10, len(text_chunks)-1)]))
    
    # Skip SentenceTransformersTokenTextSplitter entirely to avoid [UNK] token issues
    logger.info(f"Text splitting complete. Generated {len(text_chunks)} chunks")
    logger.info(f"SentenceTransformersTokenTextSplitter step skipped to avoid [UNK] token issues")
    
    return text_chunks

def create_vector_db(text_chunks, collection_name="microsoft-collection"):
    """
    Create and populate a ChromaDB vector database with text chunks
    
    Args:
        text_chunks (list): List of text chunks to embed and store
        collection_name (str): Name for the ChromaDB collection
        
    Returns:
        tuple: (ChromaDB collection, embedding function)
    """
    logger.info(f"Creating vector database with {len(text_chunks)} chunks")
    
    # 檢查OpenAI API密鑰是否可用
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    
    # 始終使用OpenAI嵌入函數，尤其對中文支持更好
    logger.info("Using OpenAI embedding function for better language support")
    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002"
    )
        
    # Create a persistent ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create collection
    chroma_collection = chroma_client.create_collection(
        name=collection_name, 
        embedding_function=embedding_function
    )
    logger.info(f"Created new collection: {collection_name}")
    
    # Add text chunks to vector database
    ids = [str(i) for i in range(len(text_chunks))]
    chroma_collection.add(ids=ids, documents=text_chunks)
    logger.info(f"Added {len(text_chunks)} documents to collection")
    
    return chroma_collection, embedding_function

def generate_multi_query(query, model="gpt-4o-mini"):
    """
    使用OpenAI模型生成多個相關查詢，以擴展原始查詢的覆蓋範圍。
    支持多語言查詢包括中文。
    
    參數:
        query (str): 原始查詢
        model (str): 使用的OpenAI模型名稱
        
    返回:
        list: 生成的相關查詢列表
        
    Generate multiple related queries using OpenAI model to expand the coverage of the original query.
    Supports multilingual queries including Chinese.
    
    Args:
        query (str): Original query
        model (str): Name of the OpenAI model to use
        
    Returns:
        list: List of generated related queries
    """
    logger.info(f"Generating multiple queries using model: {model}")
    
    # 檢測查詢是否包含中文
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
    
    # 如果是中文查詢，對查詢進行預處理
    if has_chinese:
        logger.info("Chinese query detected, applying Chinese text preprocessing")
        processed_query = process_chinese_text(query, use_segmentation=True)
        logger.info(f"Original query: '{query[:50]}...'")
        logger.info(f"Processed query: '{processed_query[:50]}...'")
    else:
        processed_query = query

    # 多語言系統提示，支持中英文查詢
    prompt = """
    You are a knowledgeable research assistant. 
    Your users are inquiring about a document or report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
    
    If the original query is in Chinese, please also respond with Chinese questions.
    如果原始查詢是中文，請也用中文回答相關問題。
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": processed_query if has_chinese else query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    
    # 處理回應，分割成多個查詢
    content = content.split("\n")
    # 移除空白查詢
    content = [q.strip() for q in content if q.strip()]
    
    logger.info(f"Generated {len(content)} related queries")
    return content

def perform_query(collection, query_text, n_results=5):
    """
    Perform vector retrieval on the database with optimized processing for Chinese
    
    Args:
        collection: ChromaDB collection
        query_text (str): Query text to search for
        n_results (int): Number of results to return
        
    Returns:
        dict: Query results
    """
    logger.info(f"Performing query: '{query_text[:50]}...'")
    
    # Check if query contains Chinese characters
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query_text)
    
    # Apply the same preprocessing to query as we do to documents
    if has_chinese:
        logger.info("Chinese query detected, applying comprehensive Chinese text preprocessing")
        
        # Always use text processing for Chinese to ensure consistency with document processing
        processed_query = process_chinese_text(query_text, use_segmentation=True)
        
        # Log the transformation for debugging
        if processed_query != query_text:
            logger.info(f"Original query: '{query_text[:50]}...'")
            logger.info(f"Processed query: '{processed_query[:50]}...'")
            logger.info("Query processing applied to match document processing")
    else:
        processed_query = query_text
    
    # Perform the query with processed text
    try:
        results = collection.query(
            query_texts=[processed_query], 
            n_results=n_results,
            include=["documents", "embeddings"]
        )
        
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        return results
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        
        # If error occurs with processed query, try again with original as fallback
        if processed_query != query_text:
            logger.warning("Error occurred with processed query. Trying again with original query as fallback.")
            results = collection.query(
                query_texts=[query_text], 
                n_results=n_results,
                include=["documents", "embeddings"]
            )
            logger.info(f"Retrieved {len(results['documents'][0])} documents using original query")
            return results
        else:
            # Re-raise the exception if we can't recover
            raise

def visualize_embeddings(
    embeddings, 
    umap_transform, 
    original_query_embedding,
    augmented_query_embeddings,
    retrieved_embeddings,
    original_query
):
    """
    Visualize the embeddings using UMAP projection
    
    Args:
        embeddings: All document embeddings
        umap_transform: Fitted UMAP transformer
        original_query_embedding: Embedding of original query
        augmented_query_embeddings: Embeddings of augmented queries
        retrieved_embeddings: Embeddings of retrieved documents
        original_query: Original query text for plot title
    """
    logger.info("Generating visualization of embeddings")
    
    # Project embeddings to 2D using UMAP
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
    projected_original_query = project_embeddings(original_query_embedding, umap_transform)
    projected_augmented_queries = project_embeddings(augmented_query_embeddings, umap_transform)
    
    # Flatten retrieved embeddings list
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]
projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

    # Create the visualization plot
    plt.figure(figsize=(10, 8))

    # Plot all document embeddings (gray dots)
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
        alpha=0.5,
        label="All Documents"
)

    # Plot augmented queries (orange X)
plt.scatter(
        projected_augmented_queries[:, 0],
        projected_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
        label="Augmented Queries"
)

    # Plot retrieved documents (green circles)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
        label="Retrieved Documents"
)

    # Plot original query (red X)
plt.scatter(
        projected_original_query[:, 0],
        projected_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
        label="Original Query"
)

plt.gca().set_aspect("equal", "datalim")
    plt.title(f"Query: {original_query}")
plt.axis("off")
    plt.legend()
    plt.show()

def main():
    """
    主函數：執行完整的查詢擴展RAG流程
    Main function: Execute the complete query expansion RAG pipeline
    """
    # 讀取PDF文件 (Read PDF document)
    pdf_texts = read_pdf()
    
    # 分割文本為適當大小的塊 (Split text into appropriate chunks)
    text_chunks = split_text(pdf_texts, debug=True)
    
    # 創建向量數據庫 (Create vector database)
    chroma_collection, embedding_function = create_vector_db(text_chunks)
    
    # 定義原始查詢 (Define original query)
    original_query = "What details can you provide about the factors that led to revenue growth?"
    
    # 檢查是否有中文查詢，如需測試中文，可以將下面一行取消註釋
    # original_query = "請提供有關導致收入增長的因素的詳細信息"
    
    # 檢測查詢是否包含中文字符
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in original_query)
    if has_chinese:
        print("\n檢測到中文查詢，將使用優化的中文處理方法")
        print(f"原始查詢: {original_query}")
        processed_original_query = process_chinese_text(original_query)
        if processed_original_query != original_query:
            print(f"處理後查詢: {processed_original_query}")
    
    # 生成擴展查詢 (Generate expanded queries)
    aug_queries = generate_multi_query(original_query)
    
    # 顯示擴展查詢 (Show the augmented queries)
    print("\n原始查詢 (Original Query):")
    print(original_query)
    print("\n擴展查詢 (Augmented Queries):")
    for query in aug_queries:
        print(f"- {query}")
    
    # 將原始查詢與擴展查詢合併 (Concatenate the original query with the augmented queries)
    joint_query = [original_query] + aug_queries
    
    # 處理中文擴展查詢（如果有）
    if has_chinese:
        processed_queries = []
        for query in joint_query:
            if any('\u4e00' <= char <= '\u9fff' for char in query):
                processed_query = process_chinese_text(query)
                processed_queries.append(processed_query)
            else:
                processed_queries.append(query)
        joint_query = processed_queries
    
    # 使用每個查詢進行檢索 (Retrieve documents using each query)
    all_retrieved_documents = []
    all_retrieved_embeddings = []
    
    for i, query in enumerate(joint_query):
        print(f"\n執行查詢 {i+1}/{len(joint_query)}: {query[:50]}...")
        # 使用優化的查詢函數進行檢索
        results = perform_query(chroma_collection, query, n_results=5)
        all_retrieved_documents.append(results["documents"][0])
        all_retrieved_embeddings.append(results["embeddings"][0])
    
    # 對檢索到的文檔進行去重 (Deduplicate the retrieved documents)
    unique_documents = set()
    for documents in all_retrieved_documents:
        for document in documents:
            unique_documents.add(document)
    
    print(f"\n檢索到 {len(unique_documents)} 個唯一文檔")
    
    # 輸出檢索結果 (Output the retrieved documents)
    for i, documents in enumerate(all_retrieved_documents):
        print(f"\n查詢 (Query): {joint_query[i][:50]}...")
        print("\n檢索到的文檔 (Retrieved documents):")
        for j, doc in enumerate(documents):
            print(f"\n文檔 {j+1}:")
            print(word_wrap(doc[:300] + "..." if len(doc) > 300 else doc))
        print("-" * 80)
    
    # 獲取全部嵌入並使用UMAP進行降維 (Get all embeddings and use UMAP for dimensionality reduction)
    all_embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(all_embeddings)
    
    # 獲取原始查詢和擴展查詢的嵌入
    # 確保使用相同的預處理作為查詢時使用的
    if has_chinese:
        processed_original_query = process_chinese_text(original_query)
        original_query_embedding = embedding_function([processed_original_query])
    else:
        original_query_embedding = embedding_function([original_query])
    
    # Get embeddings for all queries using appropriate processing
    if has_chinese:
        processed_joint_query = [process_chinese_text(q) if any('\u4e00' <= char <= '\u9fff' for char in q) else q 
                                 for q in joint_query]
        augmented_query_embeddings = embedding_function(processed_joint_query)
    else:
        augmented_query_embeddings = embedding_function(joint_query)
    
    # Visualize the embeddings
    visualize_embeddings(
        all_embeddings,
        umap_transform,
        original_query_embedding,
        augmented_query_embeddings,
        all_retrieved_embeddings,
        original_query
    )

if __name__ == "__main__":
    main()