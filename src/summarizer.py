"""
总结模块
负责使用大语言模型总结文献核心观点
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from .config import Config
from .utils import format_retrieved_documents


class Summarizer:
    """
    总结类
    
    负责使用大语言模型总结文献核心观点，包括：
    - 初始化大模型
    - 构建Prompt
    - 调用大模型生成总结
    
    Attributes:
        llm: 大语言模型
        prompt_template: Prompt模板
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
    ):
        """
        初始化总结器
        
        Args:
            model_name: 大模型名称，默认使用Config中的配置
            temperature: 生成温度，控制随机性
        """
        self.model_name = model_name or Config.CHAT_MODEL
        
        print(f"正在初始化大语言模型：{self.model_name}")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=temperature,
            openai_api_key=Config.DASHSCOPE_API_KEY,
            openai_api_base=Config.DASHSCOPE_BASE_URL,
        )
        
        self.prompt_template = PromptTemplate(
            template=Config.PROMPT_TEMPLATE,
            input_variables=["topic", "documents"],
        )
        
        print("大语言模型初始化完成")
    
    def create_prompt(self, topic: str, documents: List[Document]) -> str:
        """
        创建Prompt
        
        Args:
            topic: 用户输入的主题
            documents: 检索到的文档列表
            
        Returns:
            格式化后的Prompt字符串
        """
        formatted_docs = format_retrieved_documents(documents)
        
        prompt = self.prompt_template.format(
            topic=topic,
            documents=formatted_docs,
        )
        
        return prompt
    
    def summarize(self, topic: str, documents: List[Document]) -> str:
        """
        总结文献核心观点
        
        Args:
            topic: 用户输入的主题
            documents: 检索到的文档列表
            
        Returns:
            大模型生成的总结文本
        """
        print(f"正在总结文献核心观点，主题：{topic}")
        
        prompt = self.create_prompt(topic, documents)
        
        print("正在调用大语言模型...")
        
        response = self.llm.invoke(prompt)
        
        summary = response.content
        
        print("总结完成")
        
        return summary
    
    def summarize_with_metadata(
        self,
        topic: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """
        总结文献核心观点，并返回包含元数据的结果
        
        Args:
            topic: 用户输入的主题
            documents: 检索到的文档列表
            
        Returns:
            包含总结和元数据的字典
        """
        summary = self.summarize(topic, documents)
        
        doc_metadata = [
            {
                "title": doc.metadata.get('title', '未知标题'),
                "author": doc.metadata.get('author', '未知作者'),
                "year": doc.metadata.get('year', '未知年份'),
                "source": doc.metadata.get('source', '未知期刊'),
            }
            for doc in documents
        ]
        
        return {
            "topic": topic,
            "summary": summary,
            "documents": doc_metadata,
            "model": self.model_name,
        }


def summarize_documents(topic: str, documents: List[Document]) -> str:
    """
    总结文档的便捷函数
    
    Args:
        topic: 用户输入的主题
        documents: 检索到的文档列表
        
    Returns:
        大模型生成的总结文本
    """
    summarizer = Summarizer()
    return summarizer.summarize(topic, documents)


if __name__ == "__main__":
    from .retrieval import retrieve_documents
    
    topic = "消费者购买意愿"
    results = retrieve_documents(topic)
    
    documents = [
        Document(
            page_content=result['content'],
            metadata=result['metadata'],
        )
        for result in results
    ]
    
    summary = summarize_documents(topic, documents)
    print("\n" + "=" * 50)
    print("总结结果：")
    print("=" * 50)
    print(summary)
