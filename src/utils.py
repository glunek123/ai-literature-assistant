"""
工具函数模块
包含数据清洗等通用函数
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import re


def clean_text(text: str) -> str:
    """
    清洗文本数据
    
    Args:
        text: 待清洗的文本
        
    Returns:
        清洗后的文本
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    return text


def extract_year_from_date(date_str: str) -> str:
    """
    从日期字符串中提取年份
    
    Args:
        date_str: 日期字符串，格式如 "2026-03-06 11:12" 或 "2026-03-06"
        
    Returns:
        年份字符串，如 "2026"
    """
    if pd.isna(date_str) or date_str is None:
        return "未知"
    
    date_str = str(date_str)
    
    match = re.search(r'(\d{4})', date_str)
    if match:
        return match.group(1)
    
    return "未知"


def merge_paper_fields(row: pd.Series) -> str:
    """
    合并论文的多个字段，用于构建文档内容
    
    Args:
        row: 包含论文信息的Series
        
    Returns:
        合并后的文本内容
    """
    title = clean_text(row.get('Title-题名', ''))
    keywords = clean_text(row.get('Keyword-关键词', ''))
    summary = clean_text(row.get('Summary-摘要', ''))
    
    parts = []
    
    if title:
        parts.append(f"标题：{title}")
    
    if keywords:
        parts.append(f"关键词：{keywords}")
    
    if summary:
        parts.append(f"摘要：{summary}")
    
    return "\n".join(parts)


def format_paper_metadata(row: pd.Series) -> Dict[str, Any]:
    """
    格式化论文元数据
    
    Args:
        row: 包含论文信息的Series
        
    Returns:
        包含格式化元数据的字典
    """
    return {
        "title": clean_text(row.get('Title-题名', '未知标题')),
        "author": clean_text(row.get('Author-作者', '未知作者')),
        "organ": clean_text(row.get('Organ-单位', '未知单位')),
        "source": clean_text(row.get('Source-文献来源', '未知期刊')),
        "keywords": clean_text(row.get('Keyword-关键词', '')),
        "year": extract_year_from_date(row.get('PubTime-发表时间', '')),
        "src_database": clean_text(row.get('SrcDatabase-来源库', '')),
    }


def format_retrieved_documents(documents: List[Any]) -> str:
    """
    格式化检索到的文档，用于构建Prompt
    
    Args:
        documents: 检索到的文档列表
        
    Returns:
        格式化后的文档字符串
    """
    formatted_docs = []
    
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        content = doc.page_content
        
        formatted_doc = f"""
文献 {i}：
标题：{metadata.get('title', '未知标题')}
作者：{metadata.get('author', '未知作者')}
年份：{metadata.get('year', '未知年份')}
期刊：{metadata.get('source', '未知期刊')}
关键词：{metadata.get('keywords', '无')}
内容：{content}
"""
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def validate_excel_data(df: pd.DataFrame) -> bool:
    """
    验证Excel数据是否符合要求
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        数据是否有效
    """
    required_columns = [
        'Title-题名',
        'Author-作者',
        'Source-文献来源',
        'Keyword-关键词',
        'Summary-摘要',
        'PubTime-发表时间'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"缺少必要的列：{missing_columns}")
        return False
    
    if df.empty:
        print("数据为空")
        return False
    
    return True
