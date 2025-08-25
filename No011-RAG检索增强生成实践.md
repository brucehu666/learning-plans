# No011-RAG检索增强生成实践：连接知识库与大语言模型

## 1. RAG简介

检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了信息检索和生成模型的技术，旨在通过外部知识库来增强大语言模型（LLM）的回答能力。RAG系统能够在生成回答之前，从外部知识库中检索相关信息，并将这些信息作为上下文提供给大语言模型，从而生成更加准确、相关和最新的回答。

### RAG的主要优势
- **提高准确性**：通过引用外部知识，减少模型幻觉
- **知识更新**：无需重新训练模型即可更新知识
- **领域适应**：可以针对特定领域的知识库进行定制
- **可解释性**：可以提供回答所基于的信息来源

## 2. RAG系统的基本架构

一个完整的RAG系统通常包含以下核心组件：
1. **文档加载器**：负责从各种来源加载文档
2. **文本分割器**：将长文档分割成适当大小的片段
3. **嵌入模型**：将文本转换为向量表示
4. **向量数据库**：存储和检索向量化的文本片段
5. **检索器**：从向量数据库中检索与查询相关的文档片段
6. **大语言模型**：根据检索到的信息生成回答
7. **提示模板**：构建包含检索内容和查询的提示

## 3. 安装必要的库

### 实践示例：在Win11系统上安装RAG所需的库

```bash
# 安装基本库
pip install transformers torch sentence-transformers faiss-cpu pypdf python-dotenv

# 如果有支持CUDA的GPU（如RTX 3060），可以安装GPU版本
pip install faiss-gpu

# 安装用于处理不同文档格式的库
pip install docx2txt openpyxl

# 安装LangChain库，它提供了构建RAG系统的便捷工具
pip install langchain

# 可选：安装用于本地部署LLM的库
pip install llama-cpp-python
```

## 4. 构建简单的RAG系统

### 理论知识点
构建简单RAG系统的基本步骤：
1. 准备文档数据源
2. 对文档进行分割和向量化
3. 将向量存储在向量数据库中
4. 实现检索功能
5. 集成大语言模型
6. 构建完整的问答流程

### 实践示例：使用FAISS和Hugging Face模型构建简单的RAG系统

```python
import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

class SimpleRAGSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", 
                 llm_model_name="gpt2", # 注意：这是一个较小的模型，仅用于演示
                 vector_dim=384, 
                 storage_dir="d:\\rag_vector_db"):
        """初始化简单的RAG系统"""
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name, device=DEVICE)
        
        # 初始化向量数据库
        self.vector_dim = vector_dim
        self.storage_dir = storage_dir
        self.index = faiss.IndexFlatL2(self.vector_dim)  # 使用简单的L2索引
        self.text_chunks = []  # 存储原始文本片段
        
        # 确保存储目录存在
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        # 初始化大语言模型
        # 注意：这里使用的是一个小模型，仅用于演示
        # 在实际应用中，你可能需要使用更大的模型或部署本地LLM
        self.llm_pipeline = pipeline("text-generation", model=llm_model_name, device=0 if DEVICE == "cuda" else -1)
    
    def load_document(self, file_path):
        """从文件加载文档"""
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext.lower() == '.pdf':
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    text = "\n".join([page.extract_text() for page in reader.pages])
                except ImportError:
                    print("请安装pypdf库: pip install pypdf")
                    return False
            elif ext.lower() == '.docx':
                try:
                    import docx2txt
                    text = docx2txt.process(file_path)
                except ImportError:
                    print("请安装docx2txt库: pip install docx2txt")
                    return False
            else:
                print(f"不支持的文件格式: {ext}")
                return False
            
            # 分割文档
            self._split_document(text)
            print(f"成功加载并分割文档: {file_path}")
            return True
        except Exception as e:
            print(f"加载文档失败: {e}")
            return False
    
    def _split_document(self, text, chunk_size=500, chunk_overlap=50):
        """将文档分割成适当大小的片段"""
        # 简单的文本分割逻辑
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            # 尝试在句子边界处分割
            if end < len(text):
                # 查找下一个句号、问号或感叹号
                punctuation_pos = text.find(".", end)
                if punctuation_pos == -1:
                    punctuation_pos = text.find("?", end)
                if punctuation_pos == -1:
                    punctuation_pos = text.find("!", end)
                
                # 如果找到了合适的标点，就在那里分割
                if punctuation_pos != -1 and punctuation_pos < end + 100:
                    end = punctuation_pos + 1
            
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
        
        # 更新文本片段
        self.text_chunks.extend(chunks)
        
        # 向量化并添加到向量数据库
        self._vectorize_and_add_chunks(chunks)
    
    def _vectorize_and_add_chunks(self, chunks):
        """将文本片段向量化并添加到向量数据库"""
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        embeddings = np.array(embeddings, dtype='float32')
        
        # 添加到向量数据库
        self.index.add(embeddings)
    
    def add_text(self, text):
        """直接添加文本到系统"""
        # 分割文本
        self._split_document(text)
        return True
    
    def retrieve(self, query, k=3):
        """根据查询检索相关的文本片段"""
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype='float32')
        
        # 执行检索
        distances, indices = self.index.search(query_embedding, k)
        
        # 收集检索到的文本片段
        retrieved_chunks = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.text_chunks):
                retrieved_chunks.append({
                    "text": self.text_chunks[idx],
                    "distance": distances[0][i]
                })
        
        return retrieved_chunks
    
    def generate_answer(self, query, retrieved_chunks, max_length=300):
        """使用大语言模型生成回答"""
        # 构建提示
        context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = f"根据以下信息回答问题:\n\n{context}\n\n问题: {query}\n\n回答:"
        
        # 使用大语言模型生成回答
        try:
            response = self.llm_pipeline(prompt, max_length=max_length, 
                                        num_return_sequences=1, 
                                        truncation=True, 
                                        pad_token_id=50256)  # GPT-2的pad_token_id
            
            # 提取生成的回答
            generated_text = response[0]["generated_text"].split("回答: ")[-1]
            return generated_text
        except Exception as e:
            print(f"生成回答失败: {e}")
            return "抱歉，我无法生成回答。"
    
    def qa(self, query, k=3, max_length=300):
        """执行完整的问答流程"""
        # 检索相关文本片段
        retrieved_chunks = self.retrieve(query, k=k)
        
        # 如果没有检索到相关内容
        if not retrieved_chunks:
            return "抱歉，我没有找到相关信息来回答这个问题。"
        
        # 生成回答
        answer = self.generate_answer(query, retrieved_chunks, max_length=max_length)
        
        # 返回回答和检索到的来源
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
    
    def save(self, index_name="rag_index"):
        """保存RAG系统的状态"""
        try:
            # 保存向量索引
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            faiss.write_index(self.index, index_path)
            
            # 保存文本片段
            texts_path = os.path.join(self.storage_dir, f"{index_name}_texts.txt")
            with open(texts_path, 'w', encoding='utf-8') as f:
                for chunk in self.text_chunks:
                    f.write(chunk + "\n\n---\n\n")  # 使用分隔符区分不同的片段
            
            print(f"RAG系统状态已保存到 {self.storage_dir}")
            return True
        except Exception as e:
            print(f"保存RAG系统状态失败: {e}")
            return False
    
    def load(self, index_name="rag_index"):
        """加载RAG系统的状态"""
        try:
            # 加载向量索引
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            self.index = faiss.read_index(index_path)
            
            # 加载文本片段
            texts_path = os.path.join(self.storage_dir, f"{index_name}_texts.txt")
            with open(texts_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.text_chunks = [chunk.strip() for chunk in content.split("\n\n---\n\n") if chunk.strip()]
            
            print(f"成功加载RAG系统状态，共包含 {len(self.text_chunks)} 个文本片段")
            return True
        except Exception as e:
            print(f"加载RAG系统状态失败: {e}")
            return False

# 测试简单的RAG系统
def test_simple_rag_system():
    # 创建RAG系统实例
    rag_system = SimpleRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="gpt2",
        vector_dim=384,
        storage_dir="d:\\rag_vector_db"
    )
    
    # 添加一些示例文本
    sample_texts = [
        "Python是一种高级编程语言，由Guido van Rossum创建，于1991年首次发布。它以简洁的语法和可读性著称，支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。Python被广泛应用于Web开发、数据分析、人工智能、科学计算等领域。",
        "机器学习是人工智能的一个分支，它使计算机系统能够通过经验和数据自动改进性能，而无需明确编程。机器学习算法可以从数据中学习模式，并用于预测或决策。常见的机器学习技术包括监督学习、无监督学习和强化学习。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人类大脑的工作方式。深度学习在图像识别、自然语言处理、语音识别等领域取得了显著成功。常见的深度学习架构包括卷积神经网络(CNN)、循环神经网络(RNN)和 transformer 模型。",
        "PyTorch是一个开源的机器学习框架，由Facebook的人工智能研究实验室开发。它提供了强大的张量计算能力和自动微分功能，使其成为研究人员和开发者的首选工具之一。PyTorch的动态计算图特性使其在调试和实验方面具有优势。",
        "TensorFlow是另一个流行的开源机器学习框架，由Google开发。它提供了全面的工具和库，用于构建和部署机器学习模型。TensorFlow支持分布式计算，可以在CPU和GPU上运行，适合从研究到生产的各种应用场景。"
    ]
    
    # 添加文本到RAG系统
    for text in sample_texts:
        rag_system.add_text(text)
    
    # 测试问答功能
    queries = [
        "什么是Python？它有哪些特点？",
        "机器学习和深度学习的区别是什么？",
        "PyTorch和TensorFlow有什么不同？"
    ]
    
    print("\n===== RAG系统测试结果 =====")
    for query in queries:
        print(f"\n问题: {query}")
        result = rag_system.qa(query, k=2)
        print(f"回答: {result['answer']}")
        print("\n使用的来源:")
        for i, source in enumerate(result['sources']):
            print(f"来源 {i+1}: {source['text'][:100]}... (距离: {source['distance']:.4f})")
    
    # 保存RAG系统状态
    rag_system.save("sample_rag_index")
    
    # 创建新的RAG系统实例并加载状态
    new_rag_system = SimpleRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="gpt2",
        vector_dim=384,
        storage_dir="d:\\rag_vector_db"
    )
    
    print("\n===== 加载保存的RAG系统状态 =====")
    if new_rag_system.load("sample_rag_index"):
        # 测试加载后的系统
        test_query = "什么是机器学习？"
        result = new_rag_system.qa(test_query, k=2)
        print(f"\n测试加载后的系统 - 问题: {test_query}")
        print(f"回答: {result['answer']}")

if __name__ == "__main__":
    test_simple_rag_system()
```

## 5. 高级RAG技术

### 理论知识点
高级RAG技术主要包括以下几方面的优化：
1. **重排序**：对检索结果进行精细排序，提高相关性
2. **元数据管理**：利用文档元数据进行更精确的检索
3. **索引优化**：使用更高效的向量索引结构
4. **多模态检索**：支持文本、图像等多种数据类型的检索
5. **查询扩展**：扩展用户查询以提高检索效果

### 实践示例：实现带重排序的优化RAG系统

```python
import os
import torch
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

class OptimizedRAGSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", 
                 rerank_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 llm_model_name="gpt2",
                 vector_dim=384,
                 storage_dir="d:\\optimized_rag_vector_db"):
        """初始化优化的RAG系统"""
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name, device=DEVICE)
        
        # 初始化重排序模型
        self.rerank_model = CrossEncoder(rerank_model_name, device=DEVICE)
        
        # 初始化向量数据库 - 使用IVF索引以提高性能
        self.vector_dim = vector_dim
        self.storage_dir = storage_dir
        
        # 创建目录
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        # 使用IVFFlat索引
        nlist = 100  # 聚类中心数量
        self.quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.vector_dim, nlist, faiss.METRIC_L2)
        self.is_trained = False
        
        # 存储文本和元数据
        self.text_chunks = []
        self.metadata = []
        
        # 初始化大语言模型
        self.llm_pipeline = pipeline("text-generation", model=llm_model_name, 
                                     device=0 if DEVICE == "cuda" else -1)
    
    def add_document(self, text, metadata=None):
        """添加文档到系统"""
        # 分割文档
        chunks, chunk_metadata = self._split_document(text, metadata)
        
        # 更新文本片段和元数据
        self.text_chunks.extend(chunks)
        self.metadata.extend(chunk_metadata)
        
        # 向量化并添加到向量数据库
        self._vectorize_and_add_chunks(chunks)
        
        return True
    
    def _split_document(self, text, metadata=None, chunk_size=500, chunk_overlap=50):
        """将文档分割成适当大小的片段"""
        # 简单的文本分割逻辑
        chunks = []
        chunk_metadata = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            # 尝试在句子边界处分割
            if end < len(text):
                # 查找下一个句号、问号或感叹号
                punctuation_pos = text.find(".", end)
                if punctuation_pos == -1:
                    punctuation_pos = text.find("?", end)
                if punctuation_pos == -1:
                    punctuation_pos = text.find("!", end)
                
                # 如果找到了合适的标点，就在那里分割
                if punctuation_pos != -1 and punctuation_pos < end + 100:
                    end = punctuation_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
                # 创建元数据
                cm = {}
                if metadata:
                    cm.update(metadata)
                cm["chunk_id"] = len(chunks) - 1
                cm["chunk_start"] = start
                cm["chunk_end"] = end
                
                chunk_metadata.append(cm)
            
            start = end - chunk_overlap
        
        return chunks, chunk_metadata
    
    def _vectorize_and_add_chunks(self, chunks):
        """将文本片段向量化并添加到向量数据库"""
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        embeddings = np.array(embeddings, dtype='float32')
        
        # 如果索引尚未训练，先训练
        if not self.is_trained and len(embeddings) >= 100:
            self.index.train(embeddings)
            self.is_trained = True
            print("向量索引已训练")
        
        # 添加到向量数据库
        if self.is_trained:
            self.index.add(embeddings)
        else:
            # 使用暴力搜索作为后备
            temp_index = faiss.IndexFlatL2(self.vector_dim)
            temp_index.add(embeddings)
            self.index = temp_index
            self.is_trained = True
    
    def retrieve(self, query, k=10):
        """根据查询检索相关的文本片段"""
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype='float32')
        
        # 设置搜索参数
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10  # 增加搜索的聚类中心数量以提高精度
        
        # 执行搜索
        distances, indices = self.index.search(query_embedding, min(k, len(self.text_chunks)))
        
        # 收集检索到的文本片段
        retrieved_chunks = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.text_chunks):
                retrieved_chunks.append({
                    "text": self.text_chunks[idx],
                    "metadata": self.metadata[idx],
                    "distance": distances[0][i]
                })
        
        return retrieved_chunks
    
    def rerank(self, query, retrieved_chunks, k=3):
        """对检索结果进行重排序"""
        if not retrieved_chunks or k <= 0:
            return retrieved_chunks[:k]
        
        # 准备用于重排序的输入对
        rerank_inputs = [[query, chunk["text"]] for chunk in retrieved_chunks]
        
        # 获取重排序分数
        rerank_scores = self.rerank_model.predict(rerank_inputs)
        
        # 将分数添加到检索结果中
        for i, score in enumerate(rerank_scores):
            retrieved_chunks[i]["rerank_score"] = score
        
        # 根据重排序分数排序
        rerank_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # 返回前k个结果
        return rerank_chunks[:k]
    
    def generate_answer(self, query, retrieved_chunks, max_length=300, temperature=0.7):
        """使用大语言模型生成回答"""
        # 构建上下文
        context = "\n\n".join([f"[来源 {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        
        # 构建优化的提示
        prompt = f"根据以下信息，用中文准确回答问题。确保回答基于提供的信息，不要添加外部知识。\n\n{context}\n\n问题: {query}\n\n回答:"
        
        # 使用大语言模型生成回答
        try:
            response = self.llm_pipeline(prompt, 
                                        max_length=max_length, 
                                        num_return_sequences=1, 
                                        truncation=True, 
                                        temperature=temperature, 
                                        pad_token_id=50256)  # GPT-2的pad_token_id
            
            # 提取生成的回答
            generated_text = response[0]["generated_text"].split("回答: ")[-1]
            
            return generated_text.strip()
        except Exception as e:
            print(f"生成回答失败: {e}")
            return "抱歉，我无法生成回答。"
    
    def qa(self, query, k=10, rerank_top_k=3, max_length=300, temperature=0.7):
        """执行完整的问答流程"""
        # 记录开始时间
        start_time = time.time()
        
        # 检索相关文本片段
        retrieved_chunks = self.retrieve(query, k=k)
        
        # 对检索结果进行重排序
        reranked_chunks = self.rerank(query, retrieved_chunks, k=rerank_top_k)
        
        # 如果没有检索到相关内容
        if not reranked_chunks:
            return {
                "query": query,
                "answer": "抱歉，我没有找到相关信息来回答这个问题。",
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        # 生成回答
        answer = self.generate_answer(query, reranked_chunks, 
                                     max_length=max_length, 
                                     temperature=temperature)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "sources": reranked_chunks,
            "processing_time": processing_time
        }
    
    def batch_qa(self, queries, k=10, rerank_top_k=3, max_length=300, temperature=0.7):
        """批量处理多个查询"""
        results = []
        
        for query in queries:
            result = self.qa(query, k=k, rerank_top_k=rerank_top_k, 
                            max_length=max_length, temperature=temperature)
            results.append(result)
        
        return results
    
    def save(self, index_name="optimized_rag_index"):
        """保存RAG系统的状态"""
        try:
            # 保存向量索引
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            faiss.write_index(self.index, index_path)
            
            # 保存文本片段和元数据
            data_path = os.path.join(self.storage_dir, f"{index_name}_data.json")
            with open(data_path, 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    "text_chunks": self.text_chunks,
                    "metadata": self.metadata,
                    "is_trained": self.is_trained
                }, f, ensure_ascii=False, indent=2)
            
            print(f"RAG系统状态已保存到 {self.storage_dir}")
            return True
        except Exception as e:
            print(f"保存RAG系统状态失败: {e}")
            return False
    
    def load(self, index_name="optimized_rag_index"):
        """加载RAG系统的状态"""
        try:
            # 加载向量索引
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            self.index = faiss.read_index(index_path)
            
            # 加载文本片段和元数据
            data_path = os.path.join(self.storage_dir, f"{index_name}_data.json")
            with open(data_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                self.text_chunks = data["text_chunks"]
                self.metadata = data["metadata"]
                self.is_trained = data["is_trained"]
            
            print(f"成功加载RAG系统状态，共包含 {len(self.text_chunks)} 个文本片段")
            return True
        except Exception as e:
            print(f"加载RAG系统状态失败: {e}")
            return False

# 测试优化的RAG系统
def test_optimized_rag_system():
    # 创建RAG系统实例
    rag_system = OptimizedRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        rerank_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model_name="gpt2",
        vector_dim=384,
        storage_dir="d:\\optimized_rag_vector_db"
    )
    
    # 添加一些示例文本和元数据
    sample_docs = [
        {
            "text": "Python是一种高级编程语言，由Guido van Rossum创建，于1991年首次发布。它以简洁的语法和可读性著称，支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。Python被广泛应用于Web开发、数据分析、人工智能、科学计算等领域。",
            "metadata": {"source": "python_intro.txt", "category": "programming", "language": "python"}
        },
        {
            "text": "机器学习是人工智能的一个分支，它使计算机系统能够通过经验和数据自动改进性能，而无需明确编程。机器学习算法可以从数据中学习模式，并用于预测或决策。常见的机器学习技术包括监督学习、无监督学习和强化学习。",
            "metadata": {"source": "ml_intro.txt", "category": "ai", "topic": "machine_learning"}
        },
        {
            "text": "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人类大脑的工作方式。深度学习在图像识别、自然语言处理、语音识别等领域取得了显著成功。常见的深度学习架构包括卷积神经网络(CNN)、循环神经网络(RNN)和 transformer 模型。",
            "metadata": {"source": "dl_intro.txt", "category": "ai", "topic": "deep_learning"}
        },
        {
            "text": "PyTorch是一个开源的机器学习框架，由Facebook的人工智能研究实验室开发。它提供了强大的张量计算能力和自动微分功能，使其成为研究人员和开发者的首选工具之一。PyTorch的动态计算图特性使其在调试和实验方面具有优势。",
            "metadata": {"source": "pytorch_intro.txt", "category": "framework", "tool": "pytorch"}
        },
        {
            "text": "数据科学是一门结合了统计学、计算机科学和领域知识的学科，旨在从数据中提取有价值的见解。数据科学家使用各种工具和技术，包括Python、R、SQL、机器学习和深度学习，来分析和解释复杂的数据。数据科学的应用包括预测分析、推荐系统、欺诈检测等。",
            "metadata": {"source": "ds_intro.txt", "category": "field", "topic": "data_science"}
        }
    ]
    
    # 添加文档到RAG系统
    for doc in sample_docs:
        rag_system.add_document(doc["text"], doc["metadata"])
    
    # 测试单条问答
    queries = [
        "什么是机器学习？",
        "Python主要应用于哪些领域？",
        "深度学习和机器学习的关系是什么？"
    ]
    
    print("\n===== 单条问答测试 =====")
    for query in queries:
        result = rag_system.qa(query, k=5, rerank_top_k=2)
        print(f"\n问题: {query}")
        print(f"回答: {result['answer']}")
        print(f"处理时间: {result['processing_time']:.4f} 秒")
        print("使用的来源:")
        for i, source in enumerate(result['sources']):
            print(f"来源 {i+1}: {source['text'][:80]}... (相关性分数: {source['rerank_score']:.4f})")
            print(f"  元数据: {source['metadata']}")
    
    # 测试批量问答
    print("\n===== 测试批量问答 =====")
    batch_queries = [
        "什么是机器学习？",
        "Python主要应用于哪些领域？",
        "什么是深度学习？",
        "数据科学涉及哪些技术？"
    ]
    
    # 测量批量处理时间
    start_time = time.time()
    batch_results = rag_system.batch_qa(batch_queries, k=3, rerank_top_k=2)
    batch_time = time.time() - start_time
    
    print(f"批量处理 {len(batch_queries)} 个查询，耗时 {batch_time:.2f} 秒")
    
    # 输出批量处理结果摘要
    for i, result in enumerate(batch_results):
        print(f"\n查询 {i+1}: {result['query']}")
        print(f"回答: {result['answer'][:100]}...")
        print(f"来源数量: {len(result['sources'])}")

# 执行测试
if __name__ == "__main__":
    test_optimized_rag_system()
```

## 6. RAG系统性能优化策略

### 6.1 检索优化

**理论知识点**
- **索引优化**：选择合适的向量索引类型对检索性能至关重要
- **批次处理**：批量处理多个查询可以提高整体效率
- **缓存机制**：缓存常见查询的结果以减少重复计算

**实践示例：优化FAISS索引和批量检索**

```python
import faiss
import numpy as np
import time

def optimize_faiss_index(vectors, vector_dim=384):
    """优化FAISS索引以提高检索性能"""
    # 根据数据规模选择合适的索引类型
    n_data = len(vectors)
    
    if n_data < 1000:
        # 小规模数据：使用暴力搜索
        index = faiss.IndexFlatL2(vector_dim)
    elif n_data < 10000:
        # 中等规模数据：使用IVFFlat索引
        nlist = min(100, n_data // 10)  # 聚类中心数量
        quantizer = faiss.IndexFlatL2(vector_dim)
        index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)
        index.train(vectors)
    else:
        # 大规模数据：使用IVFPQ索引进行压缩
        nlist = min(100, n_data // 100)  # 聚类中心数量
        m = 8  # 每个向量分割的子向量数量
        quantizer = faiss.IndexFlatL2(vector_dim)
        index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 8)  # 8位量化
        index.train(vectors)
    
    # 添加向量
    index.add(vectors)
    
    # 优化搜索参数
    if hasattr(index, 'nprobe'):
        index.nprobe = min(10, nlist)  # 搜索的聚类中心数量
    
    return index

def batch_retrieval(index, query_vectors, k=10):
    """批量执行检索操作"""
    # 优化搜索参数
    if hasattr(index, 'nprobe'):
        index.nprobe = 10
    
    # 批量搜索
    distances, indices = index.search(query_vectors, k)
    
    return distances, indices

# 性能测试示例
def test_performance_optimization():
    # 模拟数据
    vector_dim = 384
    n_data = 5000
    n_queries = 100
    
    # 生成随机向量
    np.random.seed(42)
    data_vectors = np.random.random((n_data, vector_dim)).astype('float32')
    query_vectors = np.random.random((n_queries, vector_dim)).astype('float32')
    
    # 测试不同索引的性能
    print("\n===== FAISS索引性能测试 =====")
    
    # 1. IndexFlatL2 (暴力搜索)
    print("\n1. 测试暴力搜索索引:")
    index_flat = faiss.IndexFlatL2(vector_dim)
    index_flat.add(data_vectors)
    
    # 单条查询
    start_time = time.time()
    for i in range(n_queries):
        index_flat.search(query_vectors[i:i+1], 10)
    single_time = time.time() - start_time
    print(f"单条查询总时间: {single_time:.4f}秒, 每条: {single_time/n_queries*1000:.2f}毫秒")
    
    # 批量查询
    start_time = time.time()
    index_flat.search(query_vectors, 10)
    batch_time = time.time() - start_time
    print(f"批量查询时间: {batch_time:.4f}秒, 每条: {batch_time/n_queries*1000:.2f}毫秒")
    print(f"批量优化比: {single_time/batch_time:.2f}x")
    
    # 2. IndexIVFFlat (倒排文件索引)
    print("\n2. 测试IVFFlat索引:")
    nlist = 50
    quantizer = faiss.IndexFlatL2(vector_dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)
    index_ivf.train(data_vectors)
    index_ivf.add(data_vectors)
    index_ivf.nprobe = 10
    
    # 单条查询
    start_time = time.time()
    for i in range(n_queries):
        index_ivf.search(query_vectors[i:i+1], 10)
    single_time = time.time() - start_time
    print(f"单条查询总时间: {single_time:.4f}秒, 每条: {single_time/n_queries*1000:.2f}毫秒")
    
    # 批量查询
    start_time = time.time()
    index_ivf.search(query_vectors, 10)
    batch_time = time.time() - start_time
    print(f"批量查询时间: {batch_time:.4f}秒, 每条: {batch_time/n_queries*1000:.2f}毫秒")
    print(f"批量优化比: {single_time/batch_time:.2f}x")
    print(f"与暴力搜索相比的加速比: {batch_time_flat/batch_time:.2f}x")

    # 3. 对比使用优化函数的结果
    print("\n3. 测试优化函数创建的索引:")
    index_optimized = optimize_faiss_index(data_vectors, vector_dim)
    
    # 批量检索
    start_time = time.time()
    distances, indices = batch_retrieval(index_optimized, query_vectors, k=10)
    optimized_time = time.time() - start_time
    print(f"优化索引批量查询时间: {optimized_time:.4f}秒")

if __name__ == "__main__":
    test_performance_optimization()
```

### 6.2 生成优化

**理论知识点**
- **提示工程**：设计更有效的提示可以提高生成质量
- **模型选择**：为特定任务选择合适的模型
- **参数调优**：调整模型参数以平衡质量和速度

**实践示例：优化提示和生成参数**

```python
def optimize_prompt_and_generation(query, context_chunks):
    """优化提示和生成参数以提高RAG系统的回答质量"""
    # 构建优化的提示模板
    # 根据不同的查询类型选择不同的提示策略
    prompt_template = ""
    
    # 分析查询类型
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["什么是", "定义", "概念", "含义"]):
        # 定义类查询
        prompt_template = f"请根据以下信息，简洁明了地解释'{query}'的定义。确保回答准确，基于提供的信息，不要添加外部知识。\n\n{context}\n\n回答:"
    elif any(keyword in query_lower for keyword in ["如何", "步骤", "方法", "怎么做"]):
        # 方法类查询
        prompt_template = f"请根据以下信息，详细说明如何{query[2:]}。请分步骤解释，确保信息准确，基于提供的内容，不要添加外部知识。\n\n{context}\n\n回答:"
    elif any(keyword in query_lower for keyword in ["区别", "不同", "比较", "对比"]):
        # 比较类查询
        prompt_template = f"请根据以下信息，比较{query[2:]}。请列出主要区别点，确保信息准确，基于提供的内容，不要添加外部知识。\n\n{context}\n\n回答:"
    else:
        # 通用查询
        prompt_template = f"请根据以下信息，回答问题：{query}。确保回答准确，基于提供的内容，不要添加外部知识。\n\n{context}\n\n回答:"
    
    # 构建上下文
    context = "\n\n".join([f"[来源 {i+1}]: {chunk['text']}" for i, chunk in enumerate(context_chunks)])
    
    # 生成最终提示
    prompt = prompt_template.replace("{context}", context)
    
    # 根据查询类型选择合适的生成参数
    generation_params = {
        "max_length": 300,
        "temperature": 0.7,
        "num_return_sequences": 1,
        "truncation": True,
        "pad_token_id": 50256  # GPT-2的pad_token_id
    }
    
    # 根据查询类型调整参数
    if any(keyword in query_lower for keyword in ["定义", "准确", "精确"]):
        # 对于需要精确回答的查询，降低temperature
        generation_params["temperature"] = 0.3
    elif "详细" in query_lower:
        # 对于需要详细回答的查询，增加max_length
        generation_params["max_length"] = 500
    
    return prompt, generation_params

# 测试优化的提示和生成参数
def test_optimized_generation(llm_pipeline, query, context_chunks):
    """测试优化的提示和生成参数"""
    # 获取优化的提示和参数
    prompt, params = optimize_prompt_and_generation(query, context_chunks)
    
    # 生成回答
    try:
        response = llm_pipeline(prompt, **params)
        
        # 提取生成的回答
        generated_text = response[0]["generated_text"].split("回答: ")[-1]
        
        return generated_text.strip(), params
    except Exception as e:
        print(f"生成回答失败: {e}")
        return "抱歉，我无法生成回答。", params
```

## 7. RAG系统完整项目实践

### 理论知识点
一个完整的RAG系统项目通常包括以下阶段：
1. **需求分析**：明确系统的目标和功能要求
2. **数据准备**：收集、清洗和结构化文档数据
3. **系统设计**：选择合适的模型和架构
4. **实现与集成**：编写代码并整合各个组件
5. **测试与调优**：评估系统性能并进行优化
6. **部署与监控**：部署系统并设置监控机制

### 实践示例：构建一个针对技术文档的RAG问答系统

```python
import os
import torch
import faiss
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 创建项目目录
def create_project_structure(project_dir="d:\\tech_docs_rag_system"):
    """创建项目目录结构"""
    # 定义目录结构
    dirs = [
        project_dir,
        os.path.join(project_dir, "data"),
        os.path.join(project_dir, "data", "raw"),
        os.path.join(project_dir, "data", "processed"),
        os.path.join(project_dir, "vector_db"),
        os.path.join(project_dir, "models"),
        os.path.join(project_dir, "logs"),
        os.path.join(project_dir, "config")
    ]
    
    # 创建目录
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")
    
    # 创建配置文件
    config_path = os.path.join(project_dir, "config", "config.json")
    if not os.path.exists(config_path):
        config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm_model": "gpt2",  # 实际应用中可替换为更大的模型
            "vector_dim": 384,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k_retrieve": 10,
            "top_k_rerank": 3,
            "max_answer_length": 300,
            "temperature": 0.7,
            "batch_size": 4
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"创建配置文件: {config_path}")
    
    return project_dir

# 文档加载器
class DocumentLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_documents(self):
        """加载目录中的所有文档"""
        documents = []
        supported_extensions = ['.txt', '.pdf', '.docx']
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() in supported_extensions:
                    try:
                        text = self._read_file(file_path)
                        if text.strip():
                            metadata = {
                                "source": file_path,
                                "filename": file,
                                "extension": ext.lower(),
                                "load_time": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            documents.append({"text": text, "metadata": metadata})
                            print(f"成功加载文档: {file_path}")
                    except Exception as e:
                        print(f"加载文档失败 {file_path}: {e}")
                else:
                    print(f"跳过不支持的文件格式: {file_path}")
        
        return documents
    
    def _read_file(self, file_path):
        """读取不同格式的文件"""
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext.lower() == '.pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                return "\n".join([page.extract_text() for page in reader.pages])
            except ImportError:
                print("请安装pypdf库: pip install pypdf")
                return ""
        elif ext.lower() == '.docx':
            try:
                import docx2txt
                return docx2txt.process(file_path)
            except ImportError:
                print("请安装docx2txt库: pip install docx2txt")
                return ""
        else:
            return ""

# 文档处理器
class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_documents(self, documents):
        """处理文档，包括清理和分割"""
        processed_docs = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            # 清理文本
            cleaned_text = self._clean_text(text)
            
            # 分割文本
            chunks, chunk_metadata = self._split_text(cleaned_text, metadata)
            
            # 添加到处理后的文档列表
            for chunk, cm in zip(chunks, chunk_metadata):
                processed_docs.append({"text": chunk, "metadata": cm})
        
        return processed_docs
    
    def _clean_text(self, text):
        """清理文本，移除多余的空白和特殊字符"""
        # 替换多个换行符为一个
        text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
        # 替换多个空格为一个
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _split_text(self, text, metadata):
        """将文本分割成适当大小的片段"""
        chunks = []
        chunk_metadata = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            # 尝试在句子边界处分割
            if end < len(text):
                # 查找下一个句号、问号或感叹号
                punctuation_pos = text.find(".", end)
                if punctuation_pos == -1:
                    punctuation_pos = text.find("?", end)
                if punctuation_pos == -1:
                    punctuation_pos = text.find("!", end)
                
                # 如果找到了合适的标点，就在那里分割
                if punctuation_pos != -1 and punctuation_pos < end + 100:
                    end = punctuation_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                # 创建元数据副本
                cm = metadata.copy()
                cm["chunk_id"] = len(chunks) - 1
                cm["chunk_start"] = start
                cm["chunk_end"] = end
                chunk_metadata.append(cm)
            
            start = end - self.chunk_overlap
        
        return chunks, chunk_metadata

# 向量存储管理器
class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.vector_dim = config["vector_dim"]
        self.vector_store_path = os.path.join(config["project_dir"], "vector_db", "tech_docs.index")
        
        # 初始化FAISS索引
        self._init_index()
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(config["embedding_model"], device=DEVICE)
        
        # 存储文本和元数据
        self.texts = []
        self.metadata = []
    
    def _init_index(self):
        """初始化FAISS索引"""
        # 使用IVFFlat索引以提高性能
        nlist = 100  # 聚类中心数量
        self.quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.vector_dim, nlist, faiss.METRIC_L2)
        self.is_trained = False
    
    def add_documents(self, documents):
        """向向量存储中添加文档"""
        # 提取文本
        texts = [doc["text"] for doc in documents]
        metadata = [doc["metadata"] for doc in documents]
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        embeddings = np.array(embeddings, dtype='float32')
        
        # 如果索引尚未训练，先训练
        if not self.is_trained and len(embeddings) >= 100:
            self.index.train(embeddings)
            self.is_trained = True
            print("向量索引已训练")
        
        # 添加到向量存储
        if self.is_trained:
            self.index.add(embeddings)
        else:
            # 使用暴力搜索作为后备
            temp_index = faiss.IndexFlatL2(self.vector_dim)
            temp_index.add(embeddings)
            self.index = temp_index
            self.is_trained = True
        
        # 更新文本和元数据存储
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        print(f"成功添加 {len(documents)} 个文档片段到向量存储")
        return True
    
    def search(self, query, k=10):
        """根据查询搜索相关文档"""
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype='float32')
        
        # 设置搜索参数
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10  # 增加搜索的聚类中心数量以提高精度
        
        # 执行搜索
        distances, indices = self.index.search(query_embedding, min(k, len(self.texts)))
        
        # 处理搜索结果
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "distance": distances[0][i]
                })
        
        return results
    
    def save(self):
        """保存向量存储"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, self.vector_store_path)
            
            # 保存文本和元数据
            data_path = self.vector_store_path.replace('.index', '_data.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "texts": self.texts,
                    "metadata": self.metadata,
                    "is_trained": self.is_trained
                }, f, ensure_ascii=False, indent=2)
            
            print(f"向量存储已保存到 {self.vector_store_path}")
            return True
        except Exception as e:
            print(f"保存向量存储失败: {e}")
            return False
    
    def load(self):
        """加载向量存储"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.vector_store_path):
                print(f"向量存储文件不存在: {self.vector_store_path}")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(self.vector_store_path)
            
            # 加载文本和元数据
            data_path = self.vector_store_path.replace('.index', '_data.json')
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.texts = data["texts"]
                    self.metadata = data["metadata"]
                    self.is_trained = data["is_trained"]
                
                print(f"成功加载向量存储，包含 {len(self.texts)} 个文档片段")
                return True
            else:
                print(f"文本和元数据文件不存在: {data_path}")
                return False
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return False

# RAG系统主类
class TechDocsRAGSystem:
    def __init__(self, project_dir="d:\\tech_docs_rag_system"):
        self.project_dir = project_dir
        
        # 加载配置
        config_path = os.path.join(project_dir, "config", "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.config["project_dir"] = project_dir
        else:
            # 如果配置文件不存在，创建默认配置
            self.config = {
                "project_dir": project_dir,
                "embedding_model": "all-MiniLM-L6-v2",
                "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "llm_model": "gpt2",
                "vector_dim": 384,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k_retrieve": 10,
                "top_k_rerank": 3,
                "max_answer_length": 300,
                "temperature": 0.7,
                "batch_size": 4
            }
        
        # 初始化组件
        self.document_loader = DocumentLoader(os.path.join(project_dir, "data", "raw"))
        self.document_processor = DocumentProcessor(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        self.vector_store_manager = VectorStoreManager(self.config)
        
        # 初始化重排序模型
        self.rerank_model = CrossEncoder(self.config["rerank_model"], device=DEVICE)
        
        # 初始化LLM管道
        self.llm_pipeline = pipeline("text-generation", model=self.config["llm_model"], device=0 if DEVICE == "cuda" else -1)
    
    def initialize(self):
        """初始化RAG系统"""
        # 尝试加载已有的向量存储
        if self.vector_store_manager.load():
            print("RAG系统已使用现有向量存储初始化")
            return True
        
        # 如果没有已有的向量存储，创建新的
        print("未找到现有向量存储，开始创建新的向量存储...")
        
        # 加载文档
        documents = self.document_loader.load_documents()
        if not documents:
            print("没有找到可加载的文档，请在data/raw目录下添加文档")
            return False
        
        # 处理文档
        processed_docs = self.document_processor.process_documents(documents)
        if not processed_docs:
            print("文档处理失败")
            return False
        
        # 将处理后的文档添加到向量存储
        if not self.vector_store_manager.add_documents(processed_docs):
            print("添加文档到向量存储失败")
            return False
        
        # 保存向量存储
        if not self.vector_store_manager.save():
            print("保存向量存储失败")
            return False
        
        print("RAG系统初始化成功")
        return True
    
    def retrieve_and_rerank(self, query, k_retrieve=10, k_rerank=3):
        """检索并重新排序相关文档"""
        # 检索相关文档
        retrieved_docs = self.vector_store_manager.search(query, k=k_retrieve)
        
        # 如果没有检索到文档，直接返回
        if not retrieved_docs:
            return []
        
        # 使用重排序模型对检索结果进行精细排序
        if k_rerank > 0 and k_rerank <= len(retrieved_docs):
            # 准备用于重排序的输入对
            rerank_inputs = [[query, doc["text"]] for doc in retrieved_docs]
            
            # 获取重排序分数
            rerank_scores = self.rerank_model.predict(rerank_inputs)
            
            # 将分数添加到检索结果中
            for i, score in enumerate(rerank_scores):
                retrieved_docs[i]["rerank_score"] = score
            
            # 根据重排序分数排序
            retrieved_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
            
            # 只保留前k_rerank个结果
            retrieved_docs = retrieved_docs[:k_rerank]
        
        return retrieved_docs
    
    def generate_answer(self, query, retrieved_docs, max_length=300, temperature=0.7):
        """使用LLM生成回答"""
        # 构建上下文
        context = "\n\n".join([f"[来源 {i+1}]: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        # 构建提示
        prompt = f"根据以下技术文档内容，准确回答问题。确保回答基于提供的信息，不要添加外部知识。如果无法从提供的信息中找到答案，请说'无法根据提供的信息回答该问题'。\n\n{context}\n\n问题: {query}\n\n回答:"
        
        # 使用LLM生成回答
        try:
            response = self.llm_pipeline(prompt, 
                                        max_length=max_length, 
                                        num_return_sequences=1, 
                                        truncation=True, 
                                        temperature=temperature, 
                                        pad_token_id=50256)  # GPT-2的pad_token_id
            
            # 提取生成的回答
            generated_text = response[0]["generated_text"].split("回答: ")[-1]
            
            return generated_text.strip()
        except Exception as e:
            print(f"生成回答失败: {e}")
            return "抱歉，我无法生成回答。"
    
    def answer_query(self, query):
        """回答用户查询"""
        # 记录开始时间
        start_time = time.time()
        
        # 检索并重新排序相关文档
        retrieved_docs = self.retrieve_and_rerank(
            query, 
            k_retrieve=self.config["top_k_retrieve"],
            k_rerank=self.config["top_k_rerank"]
        )
        
        # 如果没有检索到相关文档
        if not retrieved_docs:
            return {
                "query": query,
                "answer": "抱歉，我没有找到相关信息来回答这个问题。",
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        # 生成回答
        answer = self.generate_answer(
            query, 
            retrieved_docs,
            max_length=self.config["max_answer_length"],
            temperature=self.config["temperature"]
        )
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "sources": retrieved_docs,
            "processing_time": processing_time
        }
    
    def add_new_document(self, file_path):
        """添加新文档到系统"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False
        
        # 创建临时文档加载器
        temp_loader = DocumentLoader(os.path.dirname(file_path))
        
        # 读取文件
        text = temp_loader._read_file(file_path)
        if not text.strip():
            print(f"无法读取文件内容或文件为空: {file_path}")
            return False
        
        # 创建文档对象
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "extension": os.path.splitext(file_path)[1].lower(),
            "load_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        document = [{"text": text, "metadata": metadata}]
        
        # 处理文档
        processed_docs = self.document_processor.process_documents(document)
        if not processed_docs:
            print("文档处理失败")
            return False
        
        # 添加到向量存储
        if not self.vector_store_manager.add_documents(processed_docs):
            print("添加文档到向量存储失败")
            return False
        
        # 保存更新后的向量存储
        if not self.vector_store_manager.save():
            print("保存向量存储失败")
            return False
        
        print(f"成功添加新文档: {file_path}")
        return True

# 主函数
def main():
    # 创建项目结构
    project_dir = create_project_structure()
    
    # 提示用户添加文档
    print("\n请在以下目录中添加技术文档：")
    print(f"{os.path.join(project_dir, 'data', 'raw')}")
    print("支持的文件格式：.txt, .pdf, .docx")
    input("添加文档后按Enter键继续...")
    
    # 初始化RAG系统
    rag_system = TechDocsRAGSystem(project_dir)
    if not rag_system.initialize():
        print("RAG系统初始化失败")
        return
    
    # 启动交互式问答
    print("\n===== 技术文档RAG问答系统 =====")
    print("输入'quit'或'exit'退出系统")
    
    while True:
        query = input("\n请输入您的问题: ")
        
        if query.lower() in ['quit', 'exit', '退出']:
            print("谢谢使用，再见！")
            break
        
        # 回答查询
        result = rag_system.answer_query(query)
        
        # 输出结果
        print(f"\n回答: {result['answer']}")
        print(f"处理时间: {result['processing_time']:.2f} 秒")
        
        # 输出来源信息
        if result['sources']:
            print(f"\n参考来源 ({len(result['sources'])}):")
            for i, source in enumerate(result['sources']):
                filename = source['metadata'].get('filename', '未知')
                score = source.get('rerank_score', 0)
                print(f"  {i+1}. {filename} (相关性分数: {score:.4f})")

if __name__ == "__main__":
    main()
```

## 8. 常见问题解决方案

### 8.1 内存优化问题
**问题**: 在处理大量文档时，内存不足。
**解决方案**: 
1. 分批处理文档，每次只加载和处理一部分文档
2. 使用FAISS的压缩索引类型（如IndexIVFPQ）来减少内存使用
3. 对大型文档进行更细粒度的分割
4. 在Windows系统上，增加虚拟内存大小

### 8.2 GPU内存错误
**问题**: 在RTX 3060 12G等GPU上运行时出现CUDA内存不足错误。
**解决方案**: 
```python
# 优化CUDA内存使用
import torch

def optimize_cuda_memory():
    # 清理未使用的缓存
    torch.cuda.empty_cache()
    
    # 设置混合精度训练
    torch.set_float32_matmul_precision('medium')  # 或 'high' 用于更高精度
    
    # 限制并行数据加载
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=8,  # 减小批大小
        shuffle=True, 
        num_workers=2  # 限制工作线程数
    )
    
    # 监控CUDA内存使用
    def monitor_memory():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"CUDA内存使用: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")
    
    return monitor_memory
```

### 8.3 检索质量不佳
**问题**: 检索到的文档与查询相关性不高。
**解决方案**: 
1. 尝试使用不同的嵌入模型，如`all-mpnet-base-v2`或领域特定的嵌入模型
2. 优化文本分割策略，确保片段大小适合你的任务
3. 增加重排序步骤，使用专门的重排序模型
4. 调整检索参数，如增加检索数量或调整向量数据库的搜索参数

### 8.4 生成回答质量问题
**问题**: 生成的回答不准确或不相关。
**解决方案**: 
1. 优化提示模板，提供更明确的指导
2. 使用更大、更适合任务的语言模型
3. 增加检索到的文档数量，提供更多上下文
4. 对生成结果进行后处理，如移除不相关内容或格式化输出

## 9. 下一步学习建议

### 9.1 深入RAG技术
- 学习多模态RAG，结合文本、图像等多种数据类型
- 探索知识图谱增强的RAG系统
- 研究自适应检索策略和动态提示工程

### 9.2 扩展应用场景
- 将RAG系统与对话系统集成，构建知识型对话机器人
- 为特定领域（如医疗、法律、金融）构建专业RAG系统
- 开发支持多语言的RAG应用

### 9.3 性能优化
- 学习模型量化技术，减小模型大小和内存占用
- 探索分布式RAG系统架构，处理大规模知识库
- 研究缓存策略，提高高频查询的响应速度

### 9.4 部署与运维
- 学习如何将RAG系统部署到生产环境
- 研究监控和评估RAG系统性能的方法
- 探索持续优化和更新RAG系统的策略

## 10. 总结

RAG（检索增强生成）技术为大语言模型提供了访问外部知识的能力，有效解决了模型幻觉和知识更新的问题。本教程介绍了RAG系统的基本架构、关键组件和实现方法，从简单的RAG系统到高级优化技术，再到完整的项目实践。通过结合向量数据库、嵌入模型、重排序模型和大语言模型，我们可以构建高性能的RAG系统，为用户提供准确、相关的信息和回答。

在Win11系统和RTX 3060 12G显卡环境下，我们优化了代码以充分利用硬件资源，同时提供了处理常见问题的解决方案。接下来，我们将学习如何在本地环境部署大语言模型，为构建更强大的AI应用打下基础。

下一章：[No012-本地LLM部署.md] 将介绍如何在本地环境部署和运行大语言模型，为RAG系统提供更强大的生成能力。