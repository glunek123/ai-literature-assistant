"""
配置文件模块
存储 API 密钥和基础配置信息
"""

import os
from typing import Dict, Any


class Config:
    """
    配置类，存储所有配置信息
    
    Attributes:
        DASHSCOPE_API_KEY: 阿里云 DashScope API 密钥
        DASHSCOPE_BASE_URL: DashScope API 基础 URL
        EMBEDDING_MODEL: Embedding 模型名称
        CHAT_MODEL: 聊天模型名称
        CHROMA_PERSIST_DIR: ChromaDB 持久化目录
        EXCEL_FILE_PATH: Excel 数据文件路径
        TOP_K: 检索返回的文献数量
    """
    
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "sk-2033afec579e49998b4289759508b739")
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    EMBEDDING_MODEL: str = "text-embedding-v3"
    CHAT_MODEL: str = "qwen-plus"
    
    # 智能检测运行环境，设置正确的路径
    if os.getenv("RAILWAY"):
        # Railway 环境
        CHROMA_PERSIST_DIR: str = "/data/chroma_db"
        EXCEL_FILE_PATH: str = "/app/data/CNKI_1_1.xlsx"
    else:
        # 本地环境
        CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        EXCEL_FILE_PATH: str = os.getenv("EXCEL_FILE_PATH", "./data/CNKI_1_1.xlsx")
    
    TOP_K: int = 3
    
    TEXT_SPLITTER_CONFIG: Dict[str, Any] = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "length_function": len,
    }
    
    PROMPT_TEMPLATE: str = """你是一位学术文献助手，帮助大学生快速了解研究领域。

任务：根据用户输入的主题，从提供的文献中总结核心观点。

要求：
1. 每篇文献总结 3 个核心观点
2. 观点要准确、简洁（≤100 字）
3. 使用学术但易懂的语言
4. 按重要性排序
5. 必须基于提供的文献内容，不得编造或夸大

用户输入的主题：{topic}

相关文献信息：
{documents}

请按照以下格式输出每篇文献的核心观点：

### 文献标题
**作者**：xxx
**年份**：xxxx
**期刊**：xxx

**核心观点**：
1. xxx
2. xxx
3. xxx
"""
    
    @classmethod
    def get_dashscope_config(cls) -> Dict[str, str]:
        """
        获取 DashScope 配置
        
        Returns:
            包含 API 密钥和基础 URL 的字典
        """
        return {
            "api_key": cls.DASHSCOPE_API_KEY,
            "base_url": cls.DASHSCOPE_BASE_URL,
        }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        确保必要的目录存在
        """
        os.makedirs(cls.CHROMA_PERSIST_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.EXCEL_FILE_PATH), exist_ok=True)
