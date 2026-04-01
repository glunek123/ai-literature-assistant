"""
知识库构建模块
负责从 Excel 数据构建向量知识库
"""

import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import os

from .config import Config
from .utils import (
    merge_paper_fields,
    format_paper_metadata,
    validate_excel_data,
)


class KnowledgeBase:
    """
    知识库构建类
    
    负责从 Excel 数据构建向量知识库，包括：
    - 读取 Excel 数据
    - 数据预处理
    - 文档分割
    - 向量化
    - 存储
    
    Attributes:
        excel_path: Excel 文件路径
        persist_directory: ChromaDB 持久化目录
        embeddings: Embedding 模型
        vectorstore: 向量存储
    """
    
    def __init__(
        self,
        excel_path: str = None,
        persist_directory: str = None,
    ):
        """
        初始化知识库
        
        Args:
            excel_path: Excel 文件路径，默认使用 Config 中的配置
            persist_directory: ChromaDB 持久化目录，默认使用 Config 中的配置
        """
        self.excel_path = excel_path or Config.EXCEL_FILE_PATH
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        
        Config.ensure_directories()
        
        self.embeddings = DashScopeEmbeddings(
            model=Config.EMBEDDING_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
        )
        
        self.vectorstore = None
    
    def load_excel_data(self) -> pd.DataFrame:
        """
        从 Excel 文件加载数据
        
        Returns:
            包含论文数据的 DataFrame
            
        Raises:
            ValueError: 如果数据验证失败
        """
        print(f"正在加载 Excel 文件：{self.excel_path}")
        
        df = pd.read_excel(self.excel_path, sheet_name='CNKI_1')
        
        print(f"成功加载 {len(df)} 条数据")
        
        if not validate_excel_data(df):
            raise ValueError("Excel 数据验证失败")
        
        return df
    
    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        从 DataFrame 创建 LangChain Document 对象
        
        Args:
            df: 包含论文数据的 DataFrame
            
        Returns:
            Document 对象列表
        """
        print("正在创建文档对象...")
        
        documents = []
        
        for idx, row in df.iterrows():
            content = merge_paper_fields(row)
            
            metadata = format_paper_metadata(row)
            metadata['doc_id'] = str(idx)
            
            doc = Document(
                page_content=content,
                metadata=metadata,
            )
            
            documents.append(doc)
        
        print(f"成功创建 {len(documents)} 个文档对象")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分割后的文档列表
        """
        print("正在分割文档...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.TEXT_SPLITTER_CONFIG['chunk_size'],
            chunk_overlap=Config.TEXT_SPLITTER_CONFIG['chunk_overlap'],
            length_function=Config.TEXT_SPLITTER_CONFIG['length_function'],
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        print(f"文档分割完成，共 {len(split_docs)} 个文档块")
        
        return split_docs
    
    def build_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        构建向量存储（分批处理，避免超过 API 限制）
        
        Args:
            documents: 文档列表
            
        Returns:
            Chroma 向量存储对象
        """
        print("正在构建向量存储...")
        print(f"使用 Embedding 模型：{Config.EMBEDDING_MODEL}")
        
        # 分批处理，每批最多 10 个文档（DashScope API 限制）
        batch_size = 10
        total_docs = len(documents)
        
        print(f"开始分批处理 {total_docs} 个文档，每批 {batch_size} 个")
        
        # 先处理第一批文档
        first_batch = documents[:batch_size]
        self.vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        print(f"✓ 第 1 批处理完成（{len(first_batch)} 个文档）")
        
        # 处理剩余批次
        for i in range(batch_size, total_docs, batch_size):
            batch_num = (i // batch_size) + 1
            batch_docs = documents[i:i + batch_size]
            
            # 添加到现有的 vectorstore
            self.vectorstore.add_documents(batch_docs)
            
            print(f"✓ 第 {batch_num} 批处理完成（{len(batch_docs)} 个文档）")
        
        print(f"向量存储构建完成，持久化到：{self.persist_directory}")
        print(f"总共处理 {total_docs} 个文档")
        
        return self.vectorstore
    
    def build_knowledge_base(self) -> Chroma:
        """
        构建完整的知识库
        
        这是主入口方法，执行完整的知识库构建流程：
        1. 加载 Excel 数据
        2. 创建文档对象
        3. 分割文档
        4. 构建向量存储
        
        Returns:
            构建完成的 Chroma 向量存储对象
        """
        print("=" * 50)
        print("开始构建知识库")
        print("=" * 50)
        
        df = self.load_excel_data()
        
        documents = self.create_documents(df)
        
        split_docs = self.split_documents(documents)
        
        vectorstore = self.build_vectorstore(split_docs)
        
        print("=" * 50)
        print("知识库构建完成！")
        print("=" * 50)
        
        return vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        加载已有的向量存储
        
        Returns:
            Chroma 向量存储对象
            
        Raises:
            ValueError: 如果向量存储不存在
        """
        if not os.path.exists(self.persist_directory):
            raise ValueError(
                f"向量存储不存在：{self.persist_directory}\n"
                "请先运行 build_knowledge_base() 构建知识库"
            )
        
        print(f"正在加载向量存储：{self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        
        print("向量存储加载完成")
        
        return self.vectorstore


def build_knowledge_base():
    """
    构建知识库的便捷函数
    
    Returns:
        Chroma 向量存储对象
    """
    kb = KnowledgeBase()
    return kb.build_knowledge_base()


if __name__ == "__main__":
    build_knowledge_base()
