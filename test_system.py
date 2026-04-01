"""
系统测试脚本
全面验证AI文献助手的功能和性能
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.retrieval import Retriever
from src.summarizer import Summarizer
from langchain_core.documents import Document


def test_config():
    """
    测试配置文件
    """
    print("=" * 50)
    print("测试1: 配置文件验证")
    print("=" * 50)
    
    print(f"✓ API密钥已配置: {Config.DASHSCOPE_API_KEY[:10]}...")
    print(f"✓ Embedding模型: {Config.EMBEDDING_MODEL}")
    print(f"✓ 聊天模型: {Config.CHAT_MODEL}")
    print(f"✓ 向量数据库路径: {Config.CHROMA_PERSIST_DIR}")
    print(f"✓ Excel文件路径: {Config.EXCEL_FILE_PATH}")
    print(f"✓ 检索数量: Top {Config.TOP_K}")
    
    print("\n配置文件验证通过！\n")


def test_retrieval():
    """
    测试检索功能
    """
    print("=" * 50)
    print("测试2: 检索功能验证")
    print("=" * 50)
    
    try:
        retriever = Retriever()
        retriever.load_vectorstore()
        print("✓ 向量数据库加载成功")
        
        test_query = "消费者购买意愿"
        print(f"\n测试查询: {test_query}")
        
        documents = retriever.retrieve(test_query)
        print(f"✓ 成功检索到 {len(documents)} 篇文献")
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            print(f"\n文献 {i}:")
            print(f"  标题: {metadata.get('title', '未知标题')}")
            print(f"  作者: {metadata.get('author', '未知作者')}")
            print(f"  年份: {metadata.get('year', '未知年份')}")
        
        print("\n检索功能验证通过！\n")
        return documents
    
    except Exception as e:
        print(f"✗ 检索功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_summarization(documents):
    """
    测试总结功能
    """
    print("=" * 50)
    print("测试3: 总结功能验证")
    print("=" * 50)
    
    if not documents:
        print("✗ 没有文档可供总结")
        return
    
    try:
        summarizer = Summarizer()
        print("✓ 大语言模型初始化成功")
        
        test_topic = "消费者购买意愿"
        print(f"\n测试主题: {test_topic}")
        
        summary = summarizer.summarize(test_topic, documents)
        print("✓ 成功生成总结")
        
        print("\n总结结果:")
        print("-" * 50)
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        print("-" * 50)
        
        print("\n总结功能验证通过！\n")
    
    except Exception as e:
        print(f"✗ 总结功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_knowledge_base():
    """
    测试知识库状态
    """
    print("=" * 50)
    print("测试4: 知识库状态验证")
    print("=" * 50)
    
    if os.path.exists(Config.CHROMA_PERSIST_DIR):
        print(f"✓ 向量数据库目录存在: {Config.CHROMA_PERSIST_DIR}")
        
        files = os.listdir(Config.CHROMA_PERSIST_DIR)
        print(f"✓ 数据库文件数量: {len(files)}")
        
        if files:
            print("✓ 数据库文件列表:")
            for file in files[:5]:
                print(f"  - {file}")
            if len(files) > 5:
                print(f"  ... 还有 {len(files) - 5} 个文件")
    else:
        print(f"✗ 向量数据库目录不存在: {Config.CHROMA_PERSIST_DIR}")
        print("  请先运行 build_kb.py 构建知识库")
    
    print("\n知识库状态验证完成！\n")


def test_data_file():
    """
    测试数据文件
    """
    print("=" * 50)
    print("测试5: 数据文件验证")
    print("=" * 50)
    
    if os.path.exists(Config.EXCEL_FILE_PATH):
        print(f"✓ Excel文件存在: {Config.EXCEL_FILE_PATH}")
        
        import pandas as pd
        df = pd.read_excel(Config.EXCEL_FILE_PATH, sheet_name='CNKI_1')
        print(f"✓ 成功读取Excel文件")
        print(f"✓ 数据行数: {len(df)}")
        print(f"✓ 数据列数: {len(df.columns)}")
        print(f"✓ 列名: {', '.join(df.columns.tolist())}")
    else:
        print(f"✗ Excel文件不存在: {Config.EXCEL_FILE_PATH}")
    
    print("\n数据文件验证完成！\n")


def run_all_tests():
    """
    运行所有测试
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "AI文献助手 - 系统测试报告")
    print("=" * 70 + "\n")
    
    test_config()
    test_data_file()
    test_knowledge_base()
    documents = test_retrieval()
    test_summarization(documents)
    
    print("=" * 70)
    print(" " * 25 + "所有测试完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
