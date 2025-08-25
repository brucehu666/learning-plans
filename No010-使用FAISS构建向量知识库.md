# No010-使用FAISS构建向量知识库：高效相似性搜索

## 1. FAISS简介

FAISS（Facebook AI Similarity Search）是Facebook AI Research开发的一个高效的相似性搜索库，专为大规模向量搜索而设计。它提供了多种算法来处理高维向量的相似性搜索问题，并且针对CPU和GPU都进行了优化。在构建向量知识库时，FAISS是一个非常强大的工具。

## 2. 安装FAISS

### 理论知识点
FAISS可以安装为CPU版本或GPU版本。对于Windows 11系统，安装FAISS需要注意版本兼容性。

### 实践示例：在Win11系统上安装FAISS

```bash
# 在Windows 11系统上安装FAISS CPU版本
pip install faiss-cpu

# 如果你有支持CUDA的GPU（如RTX 3060），可以安装GPU版本
pip install faiss-gpu
```

## 3. FAISS的基本概念

### 理论知识点
FAISS的核心概念包括：
- **向量索引**：用于存储和快速检索向量的数据结构
- **距离度量**：用于计算向量间相似度的方法（如欧氏距离、余弦相似度等）
- **量化方法**：用于压缩向量以减少内存使用的技术

### 实践示例：理解FAISS的基本组件

```python
# 导入FAISS库
import faiss
import numpy as np

# 创建一些示例向量
# 假设我们有1000个128维的向量
np.random.seed(42)  # 设置随机种子，确保结果可复现
vector_dim = 128  # 向量维度
num_vectors = 1000  # 向量数量

# 生成随机向量
vectors = np.random.random((num_vectors, vector_dim)).astype('float32')

# 打印向量的形状
print(f"向量形状: {vectors.shape}")

# 创建一个简单的索引
# IndexFlatL2是最简单的索引类型，它使用欧氏距离（L2距离）进行精确搜索
index = faiss.IndexFlatL2(vector_dim)

# 检查索引是否为空
print(f"索引是否为空: {index.is_trained}")

# 添加向量到索引
index.add(vectors)

# 检查索引中的向量数量
print(f"索引中的向量数量: {index.ntotal}")

# 创建查询向量
query_vector = np.random.random((1, vector_dim)).astype('float32')

# 执行搜索
k = 5  # 返回最相似的5个向量
 distances, indices = index.search(query_vector, k)

# 打印搜索结果
print(f"搜索结果（距离）: {distances}")
print(f"搜索结果（索引）: {indices}")

# 验证搜索结果
# 计算查询向量与第一个结果向量的欧氏距离
first_result_idx = indices[0][0]
manual_distance = np.linalg.norm(query_vector - vectors[first_result_idx])
print(f"手动计算的距离: {manual_distance}")
print(f"FAISS计算的距离: {distances[0][0]}")
print(f"距离匹配误差: {abs(manual_distance**2 - distances[0][0])}")  # L2距离是欧氏距离的平方
```

## 4. FAISS的索引类型

### 理论知识点
FAISS提供了多种索引类型，每种类型都有其特定的优势和适用场景：
- **IndexFlatL2**：精确的L2距离搜索，适用于小规模数据
- **IndexIVFFlat**：倒排文件索引，通过分区加速搜索
- **IndexIVFPQ**：乘积量化索引，在保持搜索速度的同时减少内存使用
- **IndexHNSWFlat**：基于图的索引，适用于高维数据

### 实践示例：创建和比较不同类型的索引

```python
import faiss
import numpy as np
import time

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 生成测试数据
vector_dim = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, vector_dim)).astype('float32')

# 生成查询向量
num_queries = 10
query_vectors = np.random.random((num_queries, vector_dim)).astype('float32')

# 定义比较函数

def compare_indices(index_type, index, vectors, query_vectors, k=10):
    """比较不同索引的性能"""
    # 添加向量到索引
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"{index_type} - 添加时间: {add_time:.4f}秒")
    
    # 执行搜索
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time
    print(f"{index_type} - 搜索时间: {search_time:.4f}秒")
    print(f"{index_type} - 平均每个查询的时间: {(search_time/num_queries)*1000:.2f}毫秒")
    
    # 返回搜索结果和性能指标
    return {
        "distances": distances,
        "indices": indices,
        "add_time": add_time,
        "search_time": search_time
    }

# 1. IndexFlatL2 - 精确的L2距离搜索
print("\n1. IndexFlatL2:")
flat_index = faiss.IndexFlatL2(vector_dim)
flat_results = compare_indices("IndexFlatL2", flat_index, vectors, query_vectors)

# 2. IndexIVFFlat - 倒排文件索引
print("\n2. IndexIVFFlat:")
# 定义聚类中心数量（通常为向量数量的平方根左右）
nlist = 100
# 创建倒排文件索引，使用IndexFlatL2作为量化器
quantizer = faiss.IndexFlatL2(vector_dim)
ivf_index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)
# 训练索引
ivf_index.train(vectors)
# 确保索引已训练
assert ivf_index.is_trained
# 执行比较
ivf_results = compare_indices("IndexIVFFlat", ivf_index, vectors, query_vectors)

# 3. IndexIVFPQ - 乘积量化索引
print("\n3. IndexIVFPQ:")
# 定义乘积量化的参数
m = 8  # 每个向量分割成的子向量数量
bits = 8  # 每个子向量使用的位数
# 创建乘积量化索引
pq_index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, bits)
# 训练索引
pq_index.train(vectors)
# 确保索引已训练
assert pq_index.is_trained
# 执行比较
pq_results = compare_indices("IndexIVFPQ", pq_index, vectors, query_vectors)

# 4. IndexHNSWFlat - 基于图的索引
print("\n4. IndexHNSWFlat:")
# 定义HNSW的参数
M = 16  # 每个节点的邻居数量
# 创建HNSW索引
hnsw_index = faiss.IndexHNSWFlat(vector_dim, M)
# 执行比较
hnsw_results = compare_indices("IndexHNSWFlat", hnsw_index, vectors, query_vectors)

# 比较不同索引的准确性
# 以IndexFlatL2的结果作为基准
print("\n准确性比较（与IndexFlatL2的结果匹配率）:")

# 计算IndexIVFFlat的匹配率
def calculate_match_rate(indices1, indices2):
    """计算两个搜索结果的匹配率"""
    match_count = 0
    total_count = 0
    for i in range(len(indices1)):
        set1 = set(indices1[i])
        set2 = set(indices2[i])
        match_count += len(set1 & set2)
        total_count += len(set1)
    return match_count / total_count * 100

ivf_match_rate = calculate_match_rate(flat_results["indices"], ivf_results["indices"])
pq_match_rate = calculate_match_rate(flat_results["indices"], pq_results["indices"])
hnsw_match_rate = calculate_match_rate(flat_results["indices"], hnsw_results["indices"])

print(f"IndexIVFFlat 与 IndexFlatL2 的匹配率: {ivf_match_rate:.2f}%")
print(f"IndexIVFPQ 与 IndexFlatL2 的匹配率: {pq_match_rate:.2f}%")
print(f"IndexHNSWFlat 与 IndexFlatL2 的匹配率: {hnsw_match_rate:.2f}%")

# 内存使用情况比较（估计值）
print("\n内存使用估计:")
print(f"IndexFlatL2: 约 {num_vectors * vector_dim * 4 / (1024*1024):.2f} MB (每个float32占4字节)")
print(f"IndexIVFFlat: 约 {num_vectors * vector_dim * 4 / (1024*1024):.2f} MB (与IndexFlatL2相似，但增加了倒排表的开销)")
print(f"IndexIVFPQ: 约 {num_vectors * m * bits / (8*1024*1024):.2f} MB (m={m}, bits={bits})")
print(f"IndexHNSWFlat: 约 {num_vectors * vector_dim * 4 / (1024*1024) + num_vectors * M * 4 / (1024*1024):.2f} MB (向量数据 + 图结构，M={M})")

# 总结
print("\n索引类型总结:")
print("- IndexFlatL2: 精确搜索，速度慢，内存占用大，适用于小规模数据")
print("- IndexIVFFlat: 近似搜索，速度较快，内存占用较大，适用于中等规模数据")
print("- IndexIVFPQ: 近似搜索，速度快，内存占用小，适用于大规模数据")
print("- IndexHNSWFlat: 近似搜索，速度快，内存占用中等，适用于高维数据")
```

## 5. 使用FAISS构建向量知识库

### 理论知识点
使用FAISS构建向量知识库的基本步骤包括：
1. 准备向量数据
2. 选择合适的索引类型
3. 创建和训练索引（如果需要）
4. 添加向量到索引
5. 执行相似性搜索
6. 保存和加载索引

### 实践示例：构建完整的向量知识库

```python
import faiss
import numpy as np
import json
import os

class FAISSVectorKnowledgeBase:
    def __init__(self, vector_dim, index_type="FlatL2", use_gpu=False, storage_dir="d:\\vector_db"):
        """初始化向量知识库"""
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.storage_dir = storage_dir
        self.index = None
        self.metadata = []  # 存储与向量关联的元数据
        self.index_to_id = {}  # 映射FAISS索引到自定义ID
        self.id_to_index = {}  # 映射自定义ID到FAISS索引
        
        # 确保存储目录存在
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        # 创建索引
        self._create_index()
    
    def _create_index(self):
        """创建FAISS索引"""
        # 根据索引类型创建不同的索引
        if self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.vector_dim)
        elif self.index_type == "IVFFlat":
            nlist = 100  # 聚类中心数量
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist, faiss.METRIC_L2)
        elif self.index_type == "IVFPQ":
            nlist = 100  # 聚类中心数量
            m = 8  # 子向量数量
            bits = 8  # 每个子向量的位数
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.vector_dim, nlist, m, bits)
        elif self.index_type == "HNSW":
            M = 16  # 每个节点的邻居数量
            self.index = faiss.IndexHNSWFlat(self.vector_dim, M)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        # 如果支持且请求GPU加速，将索引移动到GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                print("索引已移至GPU")
            except Exception as e:
                print(f"将索引移至GPU失败: {e}")
                print("继续使用CPU索引")
    
    def train(self, vectors):
        """训练索引（对于需要训练的索引类型）"""
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            try:
                self.index.train(vectors)
                print("索引训练完成")
                return True
            except Exception as e:
                print(f"索引训练失败: {e}")
                return False
        return True
    
    def add_vectors(self, vectors, metadatas=None, ids=None):
        """添加向量到知识库"""
        # 确保向量是float32类型
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype='float32')
        else:
            vectors = vectors.astype('float32')
        
        # 检查向量维度是否匹配
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"向量维度不匹配: 期望 {self.vector_dim}, 得到 {vectors.shape[1]}")
        
        # 如果没有提供ID，自动生成
        if ids is None:
            start_id = len(self.index_to_id)
            ids = [f"vec_{start_id + i}" for i in range(len(vectors))]
        
        # 如果没有提供元数据，使用空字典
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        
        # 确保ID、向量和元数据的数量匹配
        if len(vectors) != len(ids) or len(vectors) != len(metadatas):
            raise ValueError("向量、ID和元数据的数量必须匹配")
        
        # 检查ID是否已存在
        for id in ids:
            if id in self.id_to_index:
                raise ValueError(f"ID已存在: {id}")
        
        # 获取当前索引中的向量数量
        current_count = self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.metadata)
        
        try:
            # 添加向量到索引
            self.index.add(vectors)
            
            # 更新映射和元数据
            for i, (id, metadata) in enumerate(zip(ids, metadatas)):
                faiss_index = current_count + i
                self.index_to_id[faiss_index] = id
                self.id_to_index[id] = faiss_index
                self.metadata.append(metadata)
            
            print(f"成功添加 {len(vectors)} 个向量到知识库")
            return True
        except Exception as e:
            print(f"添加向量失败: {e}")
            return False
    
    def search(self, query_vectors, k=5):
        """在知识库中搜索相似向量"""
        # 确保查询向量是float32类型
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype='float32')
        else:
            query_vectors = query_vectors.astype('float32')
        
        # 检查查询向量维度是否匹配
        if query_vectors.shape[1] != self.vector_dim:
            raise ValueError(f"查询向量维度不匹配: 期望 {self.vector_dim}, 得到 {query_vectors.shape[1]}")
        
        try:
            # 执行搜索
            distances, indices = self.index.search(query_vectors, k)
            
            # 处理搜索结果
            results = []
            for i in range(len(query_vectors)):
                query_results = []
                for j in range(len(indices[i])):
                    faiss_index = indices[i][j]
                    # 检查索引是否有效
                    if faiss_index >= 0 and faiss_index in self.index_to_id:
                        vec_id = self.index_to_id[faiss_index]
                        metadata = self.metadata[faiss_index]
                        query_results.append({
                            "id": vec_id,
                            "metadata": metadata,
                            "distance": distances[i][j]
                        })
                results.append(query_results)
            
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def save(self, index_name="vector_index"):
        """保存索引和元数据"""
        try:
            # 创建索引的保存路径
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            metadata_path = os.path.join(self.storage_dir, f"{index_name}_metadata.json")
            
            # 如果索引在GPU上，先移回CPU
            if self.use_gpu:
                self.index = faiss.index_gpu_to_cpu(self.index)
                
            # 保存索引
            faiss.write_index(self.index, index_path)
            
            # 保存元数据和映射
            metadata = {
                "vector_dim": self.vector_dim,
                "index_type": self.index_type,
                "index_to_id": self.index_to_id,
                "id_to_index": self.id_to_index,
                "metadata": self.metadata
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 如果需要，将索引移回GPU
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                except Exception:
                    pass  # 如果失败，继续使用CPU索引
            
            print(f"索引和元数据已保存到 {self.storage_dir}")
            return True
        except Exception as e:
            print(f"保存索引和元数据失败: {e}")
            return False
    
    def load(self, index_name="vector_index"):
        """加载索引和元数据"""
        try:
            # 创建索引的加载路径
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            metadata_path = os.path.join(self.storage_dir, f"{index_name}_metadata.json")
            
            # 检查文件是否存在
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                print(f"索引或元数据文件不存在: {index_path}, {metadata_path}")
                return False
            
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 加载元数据和映射
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # 更新属性
            self.vector_dim = metadata["vector_dim"]
            self.index_type = metadata["index_type"]
            # 将JSON中的字符串键转换为整数
            self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}
            self.id_to_index = metadata["id_to_index"]
            self.metadata = metadata["metadata"]
            
            # 如果需要，将索引移至GPU
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                    print("索引已移至GPU")
                except Exception:
                    print("继续使用CPU索引")
            
            print(f"成功加载索引和元数据，共包含 {len(self.metadata)} 个向量")
            return True
        except Exception as e:
            print(f"加载索引和元数据失败: {e}")
            return False
    
    def delete_vector(self, vector_id):
        """删除向量（注意：FAISS不直接支持删除操作，这里使用标记删除的方式）"""
        if vector_id not in self.id_to_index:
            print(f"未找到ID为 {vector_id} 的向量")
            return False
        
        # 由于FAISS不直接支持删除，我们只能标记向量为已删除
        # 在实际应用中，你可能需要定期重建索引以移除已删除的向量
        faiss_index = self.id_to_index[vector_id]
        
        # 标记元数据中的向量为已删除
        if "deleted" not in self.metadata[faiss_index]:
            self.metadata[faiss_index]["deleted"] = True
            print(f"向量 {vector_id} 已标记为删除")
            return True
        else:
            print(f"向量 {vector_id} 已经是删除状态")
            return False
    
    def get_vector_count(self):
        """获取向量数量"""
        return len(self.metadata)
    
    def get_index_stats(self):
        """获取索引统计信息"""
        stats = {
            "vector_dim": self.vector_dim,
            "index_type": self.index_type,
            "vector_count": self.get_vector_count(),
            "use_gpu": self.use_gpu,
            "storage_dir": self.storage_dir
        }
        return stats

# 测试向量知识库

# 生成示例向量和元数据
def generate_sample_data(num_vectors, vector_dim):
    """生成示例向量和元数据"""
    np.random.seed(42)
    vectors = np.random.random((num_vectors, vector_dim)).astype('float32')
    
    # 生成示例元数据
    metadata = []
    for i in range(num_vectors):
        metadata.append({
            "id": f"doc_{i}",
            "text": f"这是示例文档 {i}",
            "category": f"类别_{i % 5}",
            "timestamp": f"2023-01-{i % 28 + 1}"
        })
    
    # 生成自定义ID
    ids = [f"vec_{i}" for i in range(num_vectors)]
    
    return vectors, metadata, ids

# 主测试函数
def test_vector_knowledge_base():
    # 设置参数
    vector_dim = 128
    num_vectors = 1000
    
    # 创建向量知识库实例
    # 注意：如果你的系统不支持GPU，将use_gpu设置为False
    kb = FAISSVectorKnowledgeBase(
        vector_dim=vector_dim,
        index_type="IVFFlat",  # 选择合适的索引类型
        use_gpu=True,  # 如果你有支持CUDA的GPU（如RTX 3060），可以启用GPU加速
        storage_dir="d:\\vector_db"  # 存储在D盘
    )
    
    # 生成示例数据
    vectors, metadata, ids = generate_sample_data(num_vectors, vector_dim)
    
    # 训练索引（对于需要训练的索引类型）
    if kb.index_type in ["IVFFlat", "IVFPQ"]:
        kb.train(vectors)
    
    # 添加向量到知识库
    kb.add_vectors(vectors, metadata, ids)
    
    # 生成查询向量
    query_vectors = np.random.random((5, vector_dim)).astype('float32')
    
    # 执行搜索
    search_results = kb.search(query_vectors, k=3)
    
    # 打印搜索结果
    print("\n搜索结果示例:")
    for i, results in enumerate(search_results[:2]):  # 只打印前两个查询的结果
        print(f"\n查询 {i+1} 的结果:")
        for j, result in enumerate(results):
            print(f"结果 {j+1}:")
            print(f"  ID: {result['id']}")
            print(f"  距离: {result['distance']}")
            print(f"  元数据: {result['metadata']}")
    
    # 保存索引和元数据
    kb.save("test_index")
    
    # 打印索引统计信息
    print("\n索引统计信息:")
    stats = kb.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 创建一个新的知识库实例并加载索引
    print("\n加载索引测试:")
    new_kb = FAISSVectorKnowledgeBase(
        vector_dim=vector_dim,  # 这个值会被加载的索引覆盖
        use_gpu=True,
        storage_dir="d:\\vector_db"
    )
    new_kb.load("test_index")
    
    # 使用加载的索引执行搜索
    new_search_results = new_kb.search(query_vectors[:1], k=3)
    print("\n使用加载的索引进行搜索的结果:")
    for i, result in enumerate(new_search_results[0]):
        print(f"结果 {i+1}:")
        print(f"  ID: {result['id']}")
        print(f"  距离: {result['distance']}")
        print(f"  元数据: {result['metadata']}")

# 运行测试
if __name__ == "__main__":
    test_vector_knowledge_base()
```

## 6. FAISS高级功能

### 理论知识点
FAISS提供了许多高级功能，如：
- **批处理搜索**：一次性处理多个查询
- **索引组合**：将多个索引组合在一起
- **分布式搜索**：在多台机器上执行搜索
- **自定义距离度量**：使用自定义的距离计算方法

### 实践示例：使用FAISS的高级功能

```python
import faiss
import numpy as np
import time

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 1. 批处理搜索

def test_batch_search():
    """测试批处理搜索功能"""
    print("\n===== 批处理搜索测试 =====")
    
    vector_dim = 128
    num_vectors = 10000
    num_queries = 100
    
    # 生成数据
    vectors = np.random.random((num_vectors, vector_dim)).astype('float32')
    query_vectors = np.random.random((num_queries, vector_dim)).astype('float32')
    
    # 创建索引
    index = faiss.IndexFlatL2(vector_dim)
    index.add(vectors)
    
    # 单个查询（循环）
    start_time = time.time()
    for i in range(num_queries):
        distances, indices = index.search(query_vectors[i:i+1], k=10)
    single_query_time = time.time() - start_time
    print(f"单个查询（循环）时间: {single_query_time:.4f}秒")
    print(f"平均每个查询的时间: {(single_query_time/num_queries)*1000:.2f}毫秒")
    
    # 批处理查询
    start_time = time.time()
    distances, indices = index.search(query_vectors, k=10)
    batch_query_time = time.time() - start_time
    print(f"批处理查询时间: {batch_query_time:.4f}秒")
    print(f"平均每个查询的时间: {(batch_query_time/num_queries)*1000:.2f}毫秒")
    
    # 计算加速比
    speedup = single_query_time / batch_query_time
    print(f"批处理加速比: {speedup:.2f}x")

# 2. 索引组合

def test_index_combinations():
    """测试索引组合功能"""
    print("\n===== 索引组合测试 =====")
    
    vector_dim = 128
    
    # 创建两个独立的索引
    index1 = faiss.IndexFlatL2(vector_dim)
    index2 = faiss.IndexFlatL2(vector_dim)
    
    # 生成数据
    num_vectors1 = 5000
    num_vectors2 = 5000
    vectors1 = np.random.random((num_vectors1, vector_dim)).astype('float32')
    vectors2 = np.random.random((num_vectors2, vector_dim)).astype('float32')
    
    # 添加数据到索引
    index1.add(vectors1)
    index2.add(vectors2)
    
    # 创建索引组合器
    index_combo = faiss.IndexShards(vector_dim)
    index_combo.add_shard(index1)
    index_combo.add_shard(index2)
    
    # 生成查询向量
    query_vector = np.random.random((1, vector_dim)).astype('float32')
    
    # 在组合索引上搜索
    k = 10
    distances, indices = index_combo.search(query_vector, k)
    
    print(f"组合索引中的向量总数: {index_combo.ntotal}")
    print(f"搜索结果数量: {len(indices[0])}")
    print(f"搜索结果索引: {indices[0]}")
    print(f"搜索结果距离: {distances[0]}")

# 3. 使用GPU进行批量索引和搜索

def test_gpu_acceleration():
    """测试GPU加速功能"""
    print("\n===== GPU加速测试 =====")
    
    # 检查是否有可用的GPU
    if faiss.get_num_gpus() == 0:
        print("没有可用的GPU，跳过GPU加速测试")
        return
    
    vector_dim = 128
    num_vectors = 100000  # 更大的数据集以展示GPU的优势
    num_queries = 1000
    
    # 生成数据
    vectors = np.random.random((num_vectors, vector_dim)).astype('float32')
    query_vectors = np.random.random((num_queries, vector_dim)).astype('float32')
    
    # 创建CPU索引
    cpu_index = faiss.IndexFlatL2(vector_dim)
    
    # 测量CPU索引添加时间
    start_time = time.time()
    cpu_index.add(vectors)
    cpu_add_time = time.time() - start_time
    print(f"CPU添加时间: {cpu_add_time:.4f}秒")
    
    # 测量CPU搜索时间
    k = 10
    start_time = time.time()
    cpu_distances, cpu_indices = cpu_index.search(query_vectors, k)
    cpu_search_time = time.time() - start_time
    print(f"CPU搜索时间: {cpu_search_time:.4f}秒")
    
    # 创建GPU索引
    res = faiss.StandardGpuResources()  # 分配GPU资源
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 将CPU索引移至GPU
    
    # 重新创建一个CPU索引用于测试GPU添加性能
    cpu_index2 = faiss.IndexFlatL2(vector_dim)
    gpu_index2 = faiss.index_cpu_to_gpu(res, 0, cpu_index2)  # 空的GPU索引
    
    # 测量GPU索引添加时间
    start_time = time.time()
    gpu_index2.add(vectors)
    gpu_add_time = time.time() - start_time
    print(f"GPU添加时间: {gpu_add_time:.4f}秒")
    
    # 测量GPU搜索时间
    start_time = time.time()
    gpu_distances, gpu_indices = gpu_index.search(query_vectors, k)
    gpu_search_time = time.time() - start_time
    print(f"GPU搜索时间: {gpu_search_time:.4f}秒")
    
    # 计算加速比
    add_speedup = cpu_add_time / gpu_add_time if gpu_add_time > 0 else 0
    search_speedup = cpu_search_time / gpu_search_time if gpu_search_time > 0 else 0
    print(f"添加操作加速比: {add_speedup:.2f}x")
    print(f"搜索操作加速比: {search_speedup:.2f}x")
    
    # 验证结果是否一致（由于浮点精度差异，可能会有轻微不同）
    max_distance_diff = np.max(np.abs(cpu_distances - gpu_distances))
    print(f"CPU和GPU结果的最大距离差异: {max_distance_diff}")

# 4. 自定义距离度量

def test_custom_distance():
    """测试自定义距离度量"""
    print("\n===== 自定义距离度量测试 =====")
    
    vector_dim = 128
    
    # FAISS支持的距离度量
    metrics = {
        "L2 (欧氏距离)": faiss.METRIC_L2,
        "内积": faiss.METRIC_INNER_PRODUCT,
        "L1 (曼哈顿距离)": faiss.METRIC_L1,
        "余弦相似度": faiss.METRIC_COSINE
    }
    
    # 生成数据
    num_vectors = 1000
    vectors = np.random.random((num_vectors, vector_dim)).astype('float32')
    query_vector = np.random.random((1, vector_dim)).astype('float32')
    
    # 对于余弦相似度，我们需要归一化向量
    def normalize_vectors(vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
    
    # 测试不同的距离度量
    for name, metric in metrics.items():
        print(f"\n使用{name}的索引:")
        
        # 为余弦相似度准备数据
        if metric == faiss.METRIC_COSINE:
            normalized_vectors = normalize_vectors(vectors)
            normalized_query = normalize_vectors(query_vector)
            index_vectors = normalized_vectors
            index_query = normalized_query
        else:
            index_vectors = vectors
            index_query = query_vector
        
        # 创建索引
        index = faiss.IndexFlat(vector_dim, metric)
        index.add(index_vectors)
        
        # 执行搜索
        k = 5
        distances, indices = index.search(index_query, k)
        
        print(f"前{min(k, len(indices[0]))}个结果的{name}:")
        for i in range(min(k, len(indices[0]))):
            print(f"  索引 {indices[0][i]}: {distances[0][i]:.6f}")

# 运行所有测试
def run_all_tests():
    # 测试批处理搜索
    test_batch_search()
    
    # 测试索引组合
    test_index_combinations()
    
    # 测试GPU加速
    test_gpu_acceleration()
    
    # 测试自定义距离度量
    test_custom_distance()

# 执行测试
if __name__ == "__main__":
    run_all_tests()
```

## 7. 文本嵌入与FAISS集成

### 理论知识点
在实际应用中，我们通常需要将文本转换为向量，然后使用FAISS进行相似性搜索。常见的文本嵌入模型包括：
- Sentence-BERT
- OpenAI Embeddings
- 各种预训练语言模型的嵌入层

### 实践示例：使用Sentence-BERT生成文本嵌入并与FAISS集成

```python
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json
import os

class TextVectorKnowledgeBase:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", index_type="FlatL2", 
                 use_gpu=False, storage_dir="d:\\text_vector_db"):
        """初始化文本向量知识库"""
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # 获取嵌入向量维度
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 初始化FAISS索引
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.storage_dir = storage_dir
        self.index = None
        self.texts = []  # 存储原始文本
        self.metadata = []  # 存储元数据
        
        # 确保存储目录存在
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        # 创建索引
        self._create_index()
    
    def _create_index(self):
        """创建FAISS索引"""
        # 根据索引类型创建不同的索引
        if self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.vector_dim)
        elif self.index_type == "IVFFlat":
            nlist = 100  # 聚类中心数量
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist, faiss.METRIC_L2)
        elif self.index_type == "IVFPQ":
            nlist = 100  # 聚类中心数量
            m = 8  # 子向量数量
            bits = 8  # 每个子向量的位数
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.vector_dim, nlist, m, bits)
        elif self.index_type == "HNSW":
            M = 16  # 每个节点的邻居数量
            self.index = faiss.IndexHNSWFlat(self.vector_dim, M)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        # 如果支持且请求GPU加速，将索引移动到GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                print("索引已移至GPU")
            except Exception as e:
                print(f"将索引移至GPU失败: {e}")
                print("继续使用CPU索引")
    
    def train(self, texts):
        """训练索引（对于需要训练的索引类型）"""
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            try:
                # 生成文本嵌入
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
                embeddings = np.array(embeddings, dtype='float32')
                
                # 训练索引
                self.index.train(embeddings)
                print("索引训练完成")
                return True
            except Exception as e:
                print(f"索引训练失败: {e}")
                return False
        return True
    
    def add_texts(self, texts, metadatas=None):
        """添加文本到知识库"""
        # 如果没有提供元数据，使用空字典
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # 确保文本和元数据的数量匹配
        if len(texts) != len(metadatas):
            raise ValueError("文本和元数据的数量必须匹配")
        
        try:
            # 生成文本嵌入
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            embeddings = np.array(embeddings, dtype='float32')
            
            # 添加向量到索引
            self.index.add(embeddings)
            
            # 更新文本和元数据
            self.texts.extend(texts)
            self.metadata.extend(metadatas)
            
            print(f"成功添加 {len(texts)} 个文本到知识库")
            return True
        except Exception as e:
            print(f"添加文本失败: {e}")
            return False
    
    def search(self, query_texts, k=5):
        """在知识库中搜索相似文本"""
        try:
            # 生成查询文本的嵌入
            query_embeddings = self.embedding_model.encode(query_texts, convert_to_tensor=False)
            query_embeddings = np.array(query_embeddings, dtype='float32')
            
            # 执行搜索
            distances, indices = self.index.search(query_embeddings, k)
            
            # 处理搜索结果
            results = []
            for i in range(len(query_texts)):
                query_results = []
                for j in range(len(indices[i])):
                    idx = indices[i][j]
                    # 检查索引是否有效
                    if idx >= 0 and idx < len(self.texts):
                        query_results.append({
                            "text": self.texts[idx],
                            "metadata": self.metadata[idx],
                            "distance": distances[i][j]
                        })
                results.append(query_results)
            
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def save(self, index_name="text_index"):
        """保存索引和元数据"""
        try:
            # 创建索引的保存路径
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            metadata_path = os.path.join(self.storage_dir, f"{index_name}_metadata.json")
            
            # 如果索引在GPU上，先移回CPU
            if self.use_gpu:
                self.index = faiss.index_gpu_to_cpu(self.index)
                
            # 保存索引
            faiss.write_index(self.index, index_path)
            
            # 保存元数据
            metadata = {
                "vector_dim": self.vector_dim,
                "index_type": self.index_type,
                "texts": self.texts,
                "metadata": self.metadata
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 如果需要，将索引移回GPU
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                except Exception:
                    pass  # 如果失败，继续使用CPU索引
            
            print(f"索引和元数据已保存到 {self.storage_dir}")
            return True
        except Exception as e:
            print(f"保存索引和元数据失败: {e}")
            return False
    
    def load(self, index_name="text_index"):
        """加载索引和元数据"""
        try:
            # 创建索引的加载路径
            index_path = os.path.join(self.storage_dir, f"{index_name}.index")
            metadata_path = os.path.join(self.storage_dir, f"{index_name}_metadata.json")
            
            # 检查文件是否存在
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                print(f"索引或元数据文件不存在: {index_path}, {metadata_path}")
                return False
            
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # 更新属性
            self.vector_dim = metadata["vector_dim"]
            self.index_type = metadata["index_type"]
            self.texts = metadata["texts"]
            self.metadata = metadata["metadata"]
            
            # 如果需要，将索引移至GPU
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                    print("索引已移至GPU")
                except Exception:
                    print("继续使用CPU索引")
            
            print(f"成功加载索引和元数据，共包含 {len(self.texts)} 个文本")
            return True
        except Exception as e:
            print(f"加载索引和元数据失败: {e}")
            return False
    
    def get_text_count(self):
        """获取文本数量"""
        return len(self.texts)
    
    def get_index_stats(self):
        """获取索引统计信息"""
        stats = {
            "vector_dim": self.vector_dim,
            "index_type": self.index_type,
            "text_count": self.get_text_count(),
            "use_gpu": self.use_gpu,
            "storage_dir": self.storage_dir
        }
        return stats

# 测试文本向量知识库

def test_text_vector_knowledge_base():
    # 创建文本向量知识库实例
    # 注意：如果你有支持CUDA的GPU（如RTX 3060），可以将use_gpu设置为True以获得更好的性能
    kb = TextVectorKnowledgeBase(
        embedding_model_name="all-MiniLM-L6-v2",  # 一个轻量级但高效的嵌入模型
        index_type="IVFFlat",
        use_gpu=True,
        storage_dir="d:\\text_vector_db"
    )
    
    # 示例文本数据
    sample_texts = [
        "Python是一种流行的编程语言，适用于数据分析、人工智能等领域。",
        "机器学习是人工智能的一个分支，让计算机能够从数据中学习。",
        "深度学习使用多层神经网络来模拟人类大脑的工作方式。",
        "数据科学涉及数据分析、可视化和机器学习等技术。",
        "PyTorch是一个开源的机器学习框架，由Facebook开发。",
        "TensorFlow是另一个流行的深度学习框架，由Google开发。",
        "自然语言处理让计算机能够理解和生成人类语言。",
        "计算机视觉让计算机能够理解和解释图像和视频。",
        "推荐系统广泛应用于电商、视频平台等领域。",
        "强化学习是一种机器学习方法，智能体通过与环境交互来学习。"
    ]
    
    # 示例元数据
    sample_metadatas = [
        {"category": "编程语言", "source": "维基百科"},
        {"category": "机器学习", "source": "维基百科"},
        {"category": "深度学习", "source": "维基百科"},
        {"category": "数据科学", "source": "维基百科"},
        {"category": "框架工具", "source": "官方文档"},
        {"category": "框架工具", "source": "官方文档"},
        {"category": "应用领域", "source": "学术论文"},
        {"category": "应用领域", "source": "学术论文"},
        {"category": "应用领域", "source": "行业报告"},
        {"category": "学习方法", "source": "学术论文"}
    ]
    
    # 对于需要训练的索引类型，使用部分数据进行训练
    if kb.index_type in ["IVFFlat", "IVFPQ"]:
        # 使用所有文本进行训练
        kb.train(sample_texts)
    
    # 添加文本到知识库
    kb.add_texts(sample_texts, sample_metadatas)
    
    # 测试搜索
    query_texts = [
        "Python在人工智能中的应用",
        "有哪些流行的深度学习框架",
        "计算机如何理解图像"
    ]
    
    # 执行搜索
    search_results = kb.search(query_texts, k=3)
    
    # 打印搜索结果
    print("\n搜索结果示例:")
    for i, query in enumerate(query_texts):
        print(f"\n查询: {query}")
        results = search_results[i]
        for j, result in enumerate(results):
            print(f"结果 {j+1} (距离: {result['distance']:.4f}):")
            print(f"  文本: {result['text']}")
            print(f"  元数据: {result['metadata']}")
    
    # 保存索引和元数据
    kb.save("sample_text_index")
    
    # 打印索引统计信息
    print("\n索引统计信息:")
    stats = kb.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 创建一个新的知识库实例并加载索引
    print("\n加载索引测试:")
    new_kb = TextVectorKnowledgeBase(
        embedding_model_name="all-MiniLM-L6-v2",
        use_gpu=True,
        storage_dir="d:\\text_vector_db"
    )
    new_kb.load("sample_text_index")
    
    # 使用加载的索引执行搜索
    new_search_results = new_kb.search(["推荐系统的应用场景"], k=3)
    print("\n使用加载的索引进行搜索的结果:")
    for j, result in enumerate(new_search_results[0]):
        print(f"结果 {j+1} (距离: {result['distance']:.4f}):")
        print(f"  文本: {result['text']}")
        print(f"  元数据: {result['metadata']}")

# 执行测试
if __name__ == "__main__":
    test_text_vector_knowledge_base()
```

## 8. 常见问题和解决方案

### 问题1：FAISS占用过多内存
- **症状**：当处理大规模向量数据时，FAISS索引占用过多内存
- **解决方案**：使用乘积量化(IVFPQ)等压缩索引类型，或增加nprobe参数

```python
import faiss
import numpy as np
import os

# 使用IVFPQ索引减少内存使用
def optimize_memory_usage():
    print("\n===== 内存优化示例 =====")
    
    vector_dim = 128
    num_vectors = 1000000  # 100万个向量
    
    # 生成随机数据（仅用于演示，实际应用中使用真实数据）
    print("生成随机向量...")
    vectors = np.random.random((num_vectors, vector_dim)).astype('float32')
    
    # 1. 使用FlatL2索引（对比用）
    print("\n1. 创建FlatL2索引（不压缩）...")
    flat_index = faiss.IndexFlatL2(vector_dim)
    flat_index.add(vectors)
    print(f"FlatL2索引中的向量数量: {flat_index.ntotal}")
    
    # 估计FlatL2索引的内存使用
    flat_memory_mb = num_vectors * vector_dim * 4 / (1024 * 1024)  # 每个float32占4字节
    print(f"FlatL2索引估计内存使用: {flat_memory_mb:.2f} MB")
    
    # 2. 使用IVFPQ索引（压缩）
    print("\n2. 创建IVFPQ索引（压缩）...")
    # 定义参数
    nlist = 1000  # 聚类中心数量
    m = 16  # 子向量数量
    bits = 8  # 每个子向量的位数
    
    # 创建量化器
    quantizer = faiss.IndexFlatL2(vector_dim)
    
    # 创建IVFPQ索引
    pq_index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, bits)
    
    # 训练索引
    print("训练IVFPQ索引...")
    pq_index.train(vectors)
    
    # 添加向量
    print("添加向量到IVFPQ索引...")
    pq_index.add(vectors)
    print(f"IVFPQ索引中的向量数量: {pq_index.ntotal}")
    
    # 估计IVFPQ索引的内存使用
    # 向量数据压缩为 m*bits 位 per vector
    # 加上倒排表和聚类中心的开销
    pq_memory_mb = (num_vectors * m * bits / 8 + nlist * vector_dim * 4) / (1024 * 1024)  # 转换为MB
    print(f"IVFPQ索引估计内存使用: {pq_memory_mb:.2f} MB")
    print(f"内存节省比例: {(1 - pq_memory_mb/flat_memory_mb)*100:.2f}%")
    
    # 3. 创建查询向量
    query_vector = np.random.random((1, vector_dim)).astype('float32')
    
    # 设置搜索参数以平衡速度和精度
    # nprobe参数控制搜索的聚类中心数量，增加nprobe可以提高精度，但会降低速度
    pq_index.nprobe = 10  # 默认值为1，增加到10以提高精度
    
    # 使用两种索引进行搜索
    k = 10
    
    # FlatL2搜索
    flat_distances, flat_indices = flat_index.search(query_vector, k)
    
    # IVFPQ搜索
    pq_distances, pq_indices = pq_index.search(query_vector, k)
    
    # 比较结果
    # 计算结果的交集大小
    flat_set = set(flat_indices[0])
    pq_set = set(pq_indices[0])
    intersection = flat_set & pq_set
    
    print("\n3. 比较搜索结果:")
    print(f"FlatL2和IVFPQ结果的交集大小: {len(intersection)}")
    print(f"匹配率: {len(intersection)/k*100:.2f}%")
    
    # 测试不同的nprobe值对结果的影响
    print("\n4. 测试不同nprobe值的影响:")
    
    nprobe_values = [1, 5, 10, 20, 50]
    for nprobe in nprobe_values:
        # 设置nprobe
        pq_index.nprobe = nprobe
        
        # 执行搜索
        distances, indices = pq_index.search(query_vector, k)
        
        # 计算匹配率
        pq_set = set(indices[0])
        intersection = flat_set & pq_set
        match_rate = len(intersection)/k*100
        
        print(f"nprobe={nprobe}: 匹配率 = {match_rate:.2f}%")

# 执行内存优化示例
if __name__ == "__main__":
    optimize_memory_usage()
```

### 问题2：在Windows系统上安装FAISS失败
- **症状**：在Windows 11系统上安装FAISS时出现错误
- **解决方案**：使用预编译的wheel文件或通过conda安装

```bash
# 尝试通过conda安装FAISS
conda install -c pytorch faiss-cpu

# 或者安装FAISS GPU版本（如果有支持CUDA的GPU）
conda install -c pytorch faiss-gpu

# 如果使用pip，可以尝试指定版本
pip install faiss-cpu==1.7.3
```

### 问题3：FAISS在GPU上运行时出现内存错误
- **症状**：在GPU上运行FAISS时出现CUDA内存不足错误
- **解决方案**：减少批量大小，使用更高效的索引类型，或限制GPU内存使用

```python
import faiss
import numpy as np

# 限制FAISS的GPU内存使用
def limit_gpu_memory_usage():
    # 创建GPU资源管理器
    res = faiss.StandardGpuResources()
    
    # 限制GPU内存使用（例如限制为4GB）
    max_memory_bytes = 4 * 1024 * 1024 * 1024  # 4GB
    res.setTempMemory(max_memory_bytes)
    
    # 创建索引
    vector_dim = 128
    index = faiss.IndexFlatL2(vector_dim)
    
    # 将索引移至GPU，并传入配置好的资源管理器
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    
    print("FAISS索引已创建并移至GPU，内存使用已限制")
    
    return gpu_index

# 执行内存限制示例
if __name__ == "__main__":
    gpu_index = limit_gpu_memory_usage()
```

## 9. 下一步学习建议

1. **探索更高级的FAISS索引类型**：学习如何组合不同的索引类型以获得更好的性能
2. **集成大型语言模型**：将FAISS向量知识库与大型语言模型集成，构建RAG系统
3. **优化生产环境部署**：学习如何在生产环境中高效部署FAISS索引
4. **研究分布式向量搜索**：了解如何在多台机器上分布式部署FAISS索引
5. **学习其他向量数据库**：比较FAISS与Milvus、Qdrant、Pinecone等其他向量数据库的优缺点

通过本教程，你应该已经掌握了如何使用FAISS构建高效的向量知识库。在下一章节中，我们将学习如何实现检索增强生成(RAG)系统，这是将知识库与大语言模型结合的重要技术。