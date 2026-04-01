# Tasks

## 任务列表

- [x] Task 1: 环境准备和依赖安装
  - [x] SubTask 1.1: 创建Python虚拟环境
  - [x] SubTask 1.2: 创建requirements.txt文件，列出所有依赖包
  - [x] SubTask 1.3: 安装依赖包（langchain、chromadb、streamlit、pandas、openpyxl、dashscope等）

- [x] Task 2: 项目结构搭建
  - [x] SubTask 2.1: 创建src目录和模块文件（__init__.py、config.py、knowledge_base.py、retrieval.py、summarizer.py、utils.py）
  - [x] SubTask 2.2: 创建配置文件config.py，存储API密钥和基础配置
  - [x] SubTask 2.3: 创建工具函数模块utils.py，包含数据清洗等通用函数

- [x] Task 3: 知识库构建模块开发
  - [x] SubTask 3.1: 实现Excel数据读取功能（使用Pandas读取CNKI_1_1.xlsx）
  - [x] SubTask 3.2: 实现数据预处理功能（清洗数据、合并文本字段：标题+关键词+摘要）
  - [x] SubTask 3.3: 实现文档分割功能（使用LangChain的RecursiveCharacterTextSplitter）
  - [x] SubTask 3.4: 实现向量化功能（使用DashScope的text-embedding-v3模型）
  - [x] SubTask 3.5: 实现向量存储功能（使用ChromaDB存储向量）
  - [x] SubTask 3.6: 创建知识库构建脚本，支持一键构建知识库

- [x] Task 4: 检索模块开发
  - [x] SubTask 4.1: 实现向量检索器初始化（从ChromaDB加载向量库）
  - [x] SubTask 4.2: 实现语义检索功能（根据用户输入返回Top 3相关文献）
  - [x] SubTask 4.3: 实现检索结果格式化（提取文献元数据：标题、作者、年份、期刊等）

- [x] Task 5: 总结模块开发
  - [x] SubTask 5.1: 设计Prompt模板（按照用户提供的模板格式）
  - [x] SubTask 5.2: 实现大模型调用功能（使用DashScope的Qwen-Plus模型）
  - [x] SubTask 5.3: 实现核心观点提取功能（确保每篇文献总结3个核心观点）
  - [x] SubTask 5.4: 实现输出格式化功能（按照指定格式输出：文献标题、作者、年份、期刊、核心观点）

- [x] Task 6: Web界面开发
  - [x] SubTask 6.1: 创建Streamlit应用主文件app.py
  - [x] SubTask 6.2: 实现用户输入界面（文本输入框、提交按钮）
  - [x] SubTask 6.3: 实现加载状态显示（处理过程中显示加载动画）
  - [x] SubTask 6.4: 实现结果展示界面（结构化展示文献列表和核心观点总结）
  - [x] SubTask 6.5: 添加样式美化（使用Streamlit的Markdown和CSS功能）

- [x] Task 7: 集成测试和优化
  - [x] SubTask 7.1: 测试知识库构建流程（确保所有论文正确导入）
  - [x] SubTask 7.2: 测试检索功能（验证语义检索的准确性和相关性）
  - [x] SubTask 7.3: 测试总结功能（验证核心观点的质量和准确性）
  - [x] SubTask 7.4: 测试Web界面（验证用户交互流程）
  - [x] SubTask 7.5: 性能优化（优化检索速度和响应时间）

- [x] Task 8: 文档编写
  - [x] SubTask 8.1: 编写README.md（项目介绍、安装步骤、使用说明）
  - [x] SubTask 8.2: 编写代码注释（为所有函数添加详细注释）
  - [x] SubTask 8.3: 编写使用示例（提供示例查询和输出）

## 任务依赖关系

- Task 1（环境准备）是所有任务的基础
- Task 2（项目结构）依赖Task 1
- Task 3（知识库构建）、Task 4（检索模块）、Task 5（总结模块）依赖Task 2，可以并行开发
- Task 6（Web界面）依赖Task 3、Task 4、Task 5
- Task 7（集成测试）依赖Task 6
- Task 8（文档编写）可以在Task 7完成后进行，也可以与开发过程同步进行

## 关键里程碑

1. **里程碑1**：完成环境准备和项目结构搭建（Task 1-2）
2. **里程碑2**：完成知识库构建，可以成功导入所有论文数据（Task 3）
3. **里程碑3**：完成检索和总结功能，可以离线测试（Task 4-5）
4. **里程碑4**：完成Web界面，可以端到端测试（Task 6）
5. **里程碑5**：完成所有测试和文档，项目交付（Task 7-8）
