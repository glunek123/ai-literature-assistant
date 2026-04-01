"""
AI 文献助手 - Streamlit Web 应用
基于 RAG 和大语言模型的文献检索与总结系统
"""

import streamlit as st
import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.knowledge_base import KnowledgeBase
from src.retrieval import Retriever
from src.summarizer import Summarizer
from langchain_core.documents import Document


def init_session_state():
    """
    初始化 Session State
    """
    if 'knowledge_base_built' not in st.session_state:
        st.session_state.knowledge_base_built = False
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None


def build_knowledge_base():
    """
    构建知识库
    """
    with st.spinner('正在构建知识库，请稍候...'):
        try:
            kb = KnowledgeBase()
            kb.build_knowledge_base()
            
            st.session_state.retriever = Retriever()
            st.session_state.retriever.load_vectorstore()
            
            st.session_state.summarizer = Summarizer()
            
            st.session_state.knowledge_base_built = True
            
            st.success('✅ 知识库构建成功！')
        except Exception as e:
            st.error(f'❌ 知识库构建失败：{str(e)}')


def load_knowledge_base():
    """
    加载已有的知识库
    """
    if os.path.exists(Config.CHROMA_PERSIST_DIR):
        try:
            st.session_state.retriever = Retriever()
            st.session_state.retriever.load_vectorstore()
            
            st.session_state.summarizer = Summarizer()
            
            st.session_state.knowledge_base_built = True
            
            return True
        except Exception as e:
            st.warning(f'加载知识库失败：{str(e)}')
            return False
    return False


def search_and_summarize(topic: str) -> Dict[str, Any]:
    """
    检索并总结文献
    
    Args:
        topic: 用户输入的主题
        
    Returns:
        包含检索结果和总结的字典
    """
    if not st.session_state.knowledge_base_built:
        st.error('请先构建知识库')
        return None
    
    with st.spinner('正在检索相关文献...'):
        documents = st.session_state.retriever.retrieve(topic)
    
    if not documents:
        st.warning('未找到相关文献')
        return None
    
    with st.spinner('正在总结核心观点...'):
        summary = st.session_state.summarizer.summarize(topic, documents)
    
    return {
        'topic': topic,
        'documents': documents,
        'summary': summary,
    }


def display_results(result: Dict[str, Any]):
    """
    展示检索和总结结果
    
    Args:
        result: 包含检索结果和总结的字典
    """
    st.markdown('---')
    
    st.markdown('## 📚 文献核心观点总结')
    
    st.markdown(result['summary'])
    
    st.markdown('---')
    
    st.markdown('## 📋 检索到的文献列表')
    
    for i, doc in enumerate(result['documents'], 1):
        metadata = doc.metadata
        
        with st.expander(f"📄 文献 {i}: {metadata.get('title', '未知标题')}", expanded=(i == 1)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**作者**：{metadata.get('author', '未知作者')}")
                st.markdown(f"**年份**：{metadata.get('year', '未知年份')}")
                st.markdown(f"**期刊**：{metadata.get('source', '未知期刊')}")
            
            with col2:
                st.markdown(f"**关键词**：")
                keywords = metadata.get('keywords', '')
                if keywords:
                    keyword_list = keywords.split(';;')
                    for keyword in keyword_list[:5]:
                        if keyword.strip():
                            st.markdown(f"- {keyword.strip()}")
            
            # 修复：移除嵌套 expander，改用 collapsible 区域
            st.markdown('**摘要**：')
            with st.container():
                st.markdown(f'<div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">{doc.page_content}</div>', unsafe_allow_html=True)


def main():
    """
    主函数
    """
    st.set_page_config(
        page_title='AI 文献助手',
        page_icon='📚',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    st.markdown('<h1 class="main-header">📚 AI 文献助手</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">基于 RAG 和大语言模型的智能文献检索与总结系统</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('## ⚙️ 系统设置')
        
        st.markdown('### 📊 知识库状态')
        
        if os.path.exists(Config.CHROMA_PERSIST_DIR):
            st.success('✅ 知识库已存在')
            
            if st.button('🔄 重新构建知识库', type='secondary'):
                build_knowledge_base()
        else:
            st.warning('⚠️ 知识库未构建')
            
            if st.button('🔨 构建知识库', type='primary'):
                build_knowledge_base()
        
        st.markdown('---')
        
        st.markdown('### ℹ️ 系统信息')
        st.markdown(f"""
        - **Embedding 模型**：{Config.EMBEDDING_MODEL}
        - **大语言模型**：{Config.CHAT_MODEL}
        - **检索数量**：Top {Config.TOP_K}
        - **数据来源**：CNKI 文献库
        """)
        
        st.markdown('---')
        
        st.markdown('### 📖 使用说明')
        st.markdown("""
        1. 首次使用请点击"构建知识库"
        2. 在输入框中输入论文主题
        3. 点击"开始检索"按钮
        4. 查看检索到的文献和核心观点总结
        """)
    
    if not st.session_state.knowledge_base_built:
        if not load_knowledge_base():
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('### 👋 欢迎使用 AI 文献助手！')
            st.markdown("""
            本系统可以帮助您快速了解研究领域的核心观点。
            
            **使用步骤**：
            1. 在左侧边栏点击"构建知识库"按钮
            2. 等待知识库构建完成（约需 1-2 分钟）
            3. 在下方输入框中输入您感兴趣的研究主题
            4. 系统将自动检索相关文献并总结核心观点
            
            **示例主题**：
            - 消费者购买意愿
            - 人工智能应用
            - 虚拟代言人
            - 直播带货
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            return
    
    st.markdown('### 🔍 输入研究主题')
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        topic = st.text_input(
            '请输入您想要了解的研究主题',
            placeholder='例如：消费者购买意愿、人工智能应用、虚拟代言人...',
            label_visibility='collapsed',
        )
    
    with col2:
        search_button = st.button('🔍 开始检索', type='primary', use_container_width=True)
    
    if search_button and topic:
        result = search_and_summarize(topic)
        
        if result:
            display_results(result)
    elif search_button and not topic:
        st.warning('请输入研究主题')
    
    st.markdown('---')
    
    st.markdown('### 💡 推荐主题')
    
    example_topics = [
        '消费者购买意愿',
        '虚拟代言人',
        '直播带货',
        '人工智能应用',
        '绿色消费行为',
    ]
    
    cols = st.columns(len(example_topics))
    for i, example_topic in enumerate(example_topics):
        with cols[i]:
            if st.button(example_topic, key=f'example_{i}'):
                result = search_and_summarize(example_topic)
                if result:
                    display_results(result)


if __name__ == '__main__':
    main()
