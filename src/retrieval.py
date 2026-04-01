"""
检索模块
负责从向量数据库中检索相关文献
"""

from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from .config import Config


class Retriever:
    """
    检索类
    
    负责从向量数据库中检索相关文献，包括：
    - 加载向量存储
    - 语义检索
    - 结果格式化
    
    Attributes:
        vectorstore: Chroma向量存储
        top_k: 返回的文献数量
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        top_k: int = None,
    ):
        """
        初始化检索器
        
        Args:
            persist_directory: ChromaDB持久化目录，默认使用Config中的配置
            top_k: 返回的文献数量，默认使用Config中的配置
        """
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        self.top_k = top_k or Config.TOP_K
        
        self.embeddings = DashScopeEmbeddings(
            model=Config.EMBEDDING_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
        )
        
        self.vectorstore = None
    
    def load_vectorstore(self) -> Chroma:
        """
        加载向量存储
        
        Returns:
            Chroma向量存储对象
        """
        print(f"正在加载向量存储：{self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        
        print("向量存储加载完成")
        
        return self.vectorstore
    
    def retrieve(self, query: str) -> List[Document]:
        """
        根据查询检索相关文献
        
        Args:
            query: 用户输入的查询主题
            
        Returns:
            检索到的文档列表
            
        Raises:
            ValueError: 如果向量存储未加载
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未加载，请先调用 load_vectorstore()")
        
        print(f"正在检索相关文献，查询主题：{query}")
        print(f"返回Top {self.top_k}篇文献")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )
        
        documents = retriever.invoke(query)
        
        print(f"成功检索到 {len(documents)} 篇文献")
        
        return documents
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        根据查询检索相关文献，并返回相似度分数
        
        Args:
            query: 用户输入的查询主题
            
        Returns:
            包含文档和相似度分数的元组列表
            
        Raises:
            ValueError: 如果向量存储未加载
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未加载，请先调用 load_vectorstore()")
        
        print(f"正在检索相关文献（带分数），查询主题：{query}")
        
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.top_k,
        )
        
        print(f"成功检索到 {len(results)} 篇文献")
        
        return results
    
    def format_results(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        格式化检索结果
        
        Args:
            documents: 检索到的文档列表
            
        Returns:
            格式化后的结果列表
        """
        formatted_results = []
        
        for i, doc in enumerate(documents, 1):
            result = {
                "rank": i,
                "title": doc.metadata.get('title', '未知标题'),
                "author": doc.metadata.get('author', '未知作者'),
                "year": doc.metadata.get('year', '未知年份'),
                "source": doc.metadata.get('source', '未知期刊'),
                "keywords": doc.metadata.get('keywords', ''),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            
            formatted_results.append(result)
        
        return formatted_results
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索文献（便捷方法）
        
        这是主入口方法，执行完整的检索流程：
        1. 加载向量存储（如果未加载）
        2. 检索相关文献
        3. 格式化结果
        
        Args:
            query: 用户输入的查询主题
            
        Returns:
            格式化后的检索结果列表
        """
        if self.vectorstore is None:
            self.load_vectorstore()
        
        documents = self.retrieve(query)
        
        formatted_results = self.format_results(documents)
        
        return formatted_results


def retrieve_documents(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    检索文档的便捷函数
    
    Args:
        query: 用户输入的查询主题
        top_k: 返回的文献数量
        
    Returns:
        格式化后的检索结果列表
    """
    retriever = Retriever(top_k=top_k)
    return retriever.search(query)


if __name__ == "__main__":
    results = retrieve_documents("消费者购买意愿")
    for result in results:
        print(f"\n文献 {result['rank']}:")
        print(f"标题：{result['title']}")
        print(f"作者：{result['author']}")
        print(f"年份：{result['year']}")
        print(f"期刊：{result['source']}")
