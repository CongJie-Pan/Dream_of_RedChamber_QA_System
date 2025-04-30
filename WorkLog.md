# 紅樓夢問答系統開發日誌

## 2025年4月6日

### 與兆基學長討論

今天詢問兆基學長，我問道：「兆基，我還可以詢問你三國問答系統是基於微軟的GraphRAG實作的嗎？思考功能又是如何嵌入的呢？是利用Agent RAG架構實現的嗎？我記得上次面談時，你有說到三國系統沒有用到Agent RAG。因為紅樓夢問答項目我想計畫先做成三國系統的結果，但網上還沒查到相關GraphRAG搭配 Agent RAG的範例。」

該位學長回覆了：「它不是用微軟的graphRAG，也引導先從需求方面想問題，再想要用哪些工具，不要一開始就想架構，把項目框住了。

我進而說明了：「兆基學長，我正在依照你們的建議，規劃紅樓夢問答系統的項目，想練習從文本中抽取出結構化的數據，轉成智能表格。

除了實現人物關系問答外，我也希望系統能支持綜合問答，並在回複時附上相關段落的來源參考。

目前我想到一個比較合適的技術組合是 GraphRAG 配合智能體（Agent)，讓智能體來思考判斷需要檢索哪些內容，再由 GraphRAG 提供結構化的知識支持。

只是我現在卡在技術點上：不太清楚如何把 GraphRAG 和智能體結合起來。學長有沒有什麽建議，或者可以分享一些相關的思路或經驗嗎？」


兆基學長回答道：GraphRAG和Agentic RAG更像是一種理念，我把它們看得太割裂了。因為它們要發論文的關係，所以會各自把兩種概念系統化，造成一個workflow，所以看起來是兩個不一樣的系統，但實際上它們只是一種理念，是可以相互使用搭配的。你看GraphRAG每個公司做的方法都不一樣，但本質都是一樣就是要從圖數據去檢索資料。所以要把它們看成工具，而不是不可融合的整體。簡單來說，GraphRAG 本身不負責思考，它負責提供結構化的「線索」；真正的「思考」（推理、總結）是 LLM 的工作。兩者搭配起來就像是「圖書館員（GraphRAG）找書 + 作家（LLM）寫文章」的合作模式。」

學長建議我可以先從RAG、GraphRAG、Agentic RAG的實作開始做起，可以使用別人的成果進行實作，這樣會更有對於技術本身的應用更有體會，而不只是單止於知識面。因此，我規劃先試試看RAG、GraphRAG、Agentic RAG 各用紅樓夢測試文本(前三回)試驗其效果。

### RAG實驗進度

- 綜整今天的進度：
  - 已完成基本RAG系統的搭建
  - 使用Streamlit UI 搭建測試程序
  - 已實現文檔上傳、處理和檢索的基本功能
  - 已添加查詢擴展功能以提高檢索效果
  - 已實現多種文件格式的支持（PDF、DOCX、TXT、HTML）

- 遇到的問題：
  - 中文文本處理問題：縱使選擇了Chinese的text-embedding模型，仍然遇到檢索出來的文件顯示為`[UNK] [UNK] [UNK]`的問題
  - 可能的解決方案：嘗試使用OpenAI的text-embedding模型，這可能會提供更好的中文支持

### 下一步計劃

- 解決中文文本處理問題
- 由開源項目，實現GraphRAG系統
- 由開源項目，實現Agentic RAG系統
- 比較三種系統在紅樓夢文本上的效果

---

## 2025年4月9日

### 查詢擴展系統優化進度

今日繼續改進基本 RAG 系統中文文本處理的問題，特別針對查詢擴展系統進行了一系列優化：

- **已完成的優化**：
  - 移除了 SentenceTransformer 嵌入函數，改為只使用 OpenAI 嵌入模型，以避免中文 [UNK] 標記問題
  - 優化了 jieba 中文分詞的實現方式，提升中文文本分割效果
  - 使用了共用的 `extract_text_from_file` 函數處理文檔，統一文本讀取流程
  - 增強了中文文本處理邏輯，特別是對中文標點符號和分割的處理
  - 去除了不必要的警告和條件檢查，使代碼更加清晰

- **遇到的問題**：
  - 儘管進行了上述優化，仍然遇到部分中文查詢結果顯示 [UNK] 標記的問題
  - 經過調查，發現問題可能出在嵌入模型對中文字符的處理上，特別是在處理一些生僻字或特殊符號時
  - 即使使用 OpenAI 的嵌入模型，某些中文文本的向量表示仍不夠準確，導致檢索質量下降

- **可能的解決方案**：
  - 嘗試更進一步優化中文文本預處理流程，包括特殊字符替換和標準化
  - 考慮使用更適合中文處理的分詞工具或方法，如 HanLP 或 THULAC
  - 探索其他專為中文優化的嵌入模型，如 BGE（BAAI General Embeddings）等
  - 實現查詢後處理邏輯，過濾和替換包含 [UNK] 標記的結果

### 下一步計劃

- 止損此RAG專案，改進行lightRAG專案。

---

## 2025/4/11

- 完成架設Basic RAG 和 light RAG env、API，以及虛擬終端環境建立好。

### 下一步計畫

- Readme 檔案中 Running the Implementations 部分

---

## 2025/4/25

### lightRAG 測試結果與問題分析

- lightRAG 的答覆品質簡略不太好。試試看單純的 RAG Agent。
- 閱讀 lightRAG 的核心檔案以了解其實現機制。
- 詳細分析持續出現 "file path not found" 錯誤的原因。

### 品質問題診斷

- **lightRAG 品質不佳的主要原因**：
  - 品質取決於數據，而文本大都以列點為主，並非直接的段落文章。
  - 系統中混入了品質不好的紅樓夢文本，存在雜訊干擾。
  - 這些因素導致檢索結果不夠連貫和準確，影響最終回答品質。

### 下一步計劃

- 優化數據質量，移除低品質文本
- 實現單純的 RAG Agent 模型進行對比測試
- 解決路徑問題導致的檢索失敗

---

## 2025/4/29
已藉由實作textParser軟件，完成將data文本轉為較高品質之繁體中文段落markdown格式文本。

- 優化數據質量，移除低品質文本ok
- 解決lightRAG路徑問題導致的檢索失敗
- 實現單純的 RAG Agent 模型進行對比測試
- textParser 軟件(app.py)在cursur ai輸出code的最新版本(對話紀錄: streamlit介面開發與deepseek整合，新版的對話紀錄已經不見。)會有以下問題: "當按下處理文件時，軟件會一直在執行，而沒有實際動作。已經盡量在程式碼做更多錯誤日誌紀錄，錯誤日誌還是沒有任何發現。" 所以就返回了可以執行的版本。

---

## 2025/4/30

### BasicRAG與LightRAG系統比較研究

今日完成了BasicRAG和LightRAG兩個系統的輸出品質比較，獲得以下初步觀察結果：

1. **系統差異分析**：
   - **LightRAG**：能夠理解和提供更延伸的知識內容，但輸出內容相對簡短。系統對知識的關聯性掌握較佳。
   - **BasicRAG**：輸出更為完整且詳細，提供的解釋和論述更加豐富，但有時會缺少一些知識的延伸連結。

2. **解決問題**：
   - 之前LightRAG系統中出現的路徑問題導致檢索失敗的錯誤已經修復，不再出現。

3. **後續工作**：
   - 需要進行更多的系統比較測試，計劃使用Perplexity等工具進行更客觀的評估。

### BasicRAG系統優化

在進行BasicRAG系統開發時遇到了一些界面問題：

1. **界面錯誤**：
   - streamlit_app.py中複製文字的按鈕功能無法正常顯示
   - 運行時出現終端機錯誤，嚴重影響使用體驗

2. **解決方案**：
   - 暫時回退到之前的穩定版本，確保系統可以順利運行
   - 計劃在後續版本中重新實現此功能

<details>
<summary>錯誤代碼</summary>

```
2025-04-30 21:33:58.526 Examining the path of torch.classes raised:
Traceback (most recent call last):
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\streamlit\web\bootstrap.py", line 347, in run
    if asyncio.get_running_loop().is_running():
       ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: no running event loop

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\streamlit\watcher\local_sources_watcher.py", line 217, in get_module_paths
    potential_paths = extract_paths(module)
                      ^^^^^^^^^^^^^^^^^^^^^
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\streamlit\watcher\local_sources_watcher.py", line 210, in <lambda>
    lambda m: list(m.__path__._path),
                   ^^^^^^^^^^^^^^^^
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\torch\_classes.py", line 13, in __getattr__
    proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
2025-04-30 21:34:31,834 - rag_agent - INFO - Successfully loaded metadata from ./chroma_db\import_metadata.json
────────────────────────── Traceback (most recent call last) ───────────────────────────
  D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-a
  gent\BasicRAG\venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py:121
   in exec_func_with_error_handling                                               

                                                                                  

  D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-a
  gent\BasicRAG\venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py
  :640 in code_to_exec                                                            

                                                                                  

  D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-a
  gent\BasicRAG\streamlit_app.py:572 in <module>                                  

                                                                                  

    569                                                                           

    570 # Application entry point                                                 

    571 if __name__ == "__main__":                                                

  ❱ 572 │   asyncio.run(main())                                                   

    573                                                                           

                                                                                  

  C:\Users\USER\AppData\Local\Programs\Python\Python312\Lib\asyncio\runners.py:194 in
  run                                                                             

                                                                                  

    191 │   │   │   "asyncio.run() cannot be called from a running event loop")   

    192 │                                                                         

    193 │   with Runner(debug=debug, loop_factory=loop_factory) as runner:        

  ❱ 194 │   │   return runner.run(main)                                           

    195                                                                           

    196                                                                           

    197 def _cancel_all_tasks(loop):                                              

                                                                                  

  C:\Users\USER\AppData\Local\Programs\Python\Python312\Lib\asyncio\runners.py:118 in
  run                                                                             

                                                                                  

    115 │   │                                                                     

    116 │   │   self._interrupt_count = 0                                         

    117 │   │   try:                                                              

  ❱ 118 │   │   │   return self._loop.run_until_complete(task)                    

    119 │   │   except exceptions.CancelledError:                                 

    120 │   │   │   if self._interrupt_count > 0:                                 

    121 │   │   │   │   uncancel = getattr(task, "uncancel", None)                

                                                                                  

  C:\Users\USER\AppData\Local\Programs\Python\Python312\Lib\asyncio\base_events.py:687
  in run_until_complete                                                           

                                                                                  

     684 │   │   if not future.done():                                            

     685 │   │   │   raise RuntimeError('Event loop stopped before Future completed.')
     686 │   │                                                                    

  ❱  687 │   │   return future.result()                                           

     688 │                                                                        

     689 │   def stop(self):                                                      

     690 │   │   """Stop running the event loop.                                  

                                                                                  

  D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-a
  gent\BasicRAG\streamlit_app.py:501 in main                                      

                                                                                  

    498 │   for msg in st.session_state.messages:                                 

    499 │   │   if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
    500 │   │   │   for part in msg.parts:                                        

  ❱ 501 │   │   │   │   display_message_part(part)                                

    502 │                                                                         

    503 │   # Chat input for the user                                             

    504 │   user_input = st.chat_input("輸入您的問題...")  # Enter your question...
                                                                                  

  D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-a
  gent\BasicRAG\streamlit_app.py:141 in display_message_part                      

                                                                                  

    138 │   # Tool calls - show in expander for debugging (hidden by default)     

    139 │   elif part.part_kind == 'tool-call':                                   

    140 │   │   with st.expander("工具呼叫 (調試)", expanded=False):  # Tool Call (Deb
  ❱ 141 │   │   │   st.markdown(f"**工具:** {part.name}")  # Tool                 

    142 │   │   │   st.markdown(f"**參數:** {part.arguments}")  # Arguments       

    143 │   # Tool returns - show in expander for debugging (hidden by default)   

    144 │   elif part.part_kind == 'tool-return':                                 

────────────────────────────────────────────────────────────────────────────────────────
AttributeError: 'ToolCallPart' object has no attribute 'name'
2025-04-30 21:34:33.179 Examining the path of torch.classes raised:
Traceback (most recent call last):
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\streamlit\web\bootstrap.py", line 347, in run
    if asyncio.get_running_loop().is_running():
       ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: no running event loop

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\streamlit\watcher\local_sources_watcher.py", line 217, in get_module_paths
    potential_paths = extract_paths(module)
                      ^^^^^^^^^^^^^^^^^^^^^
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\streamlit\watcher\local_sources_watcher.py", line 210, in <lambda>
    lambda m: list(m.__path__._path),
                   ^^^^^^^^^^^^^^^^
  File "D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\BasicRAG\venv\Lib\site-packages\torch\_classes.py", line 13, in __getattr__
    proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
```

</details>

### 未來功能規劃 - Agentic RAG增強

為了提升BasicRAG系統的智能程度，提出了以下功能增強計劃：

1. **思考Agent集成**：
   - 開發智能思考Agent模組，使系統能夠自主思考需要提取哪些檔案
   - 實現智能閱讀功能，提高文本理解能力
   - 加入思考-輸出機制，使回答更加有條理和邏輯性

2. **界面優化**：
   - 已在系統特點中加入"Agentic RAG系統"的描述，為下一階段開發做準備

這些改進將使BasicRAG系統在保持其詳細、完整輸出優勢的同時，進一步提升知識關聯和延伸能力。


