# Datawhale - LLM Cookbook

## 项目简介

LLM Cookbook 是一个面向开发者的大模型手册，针对国内开发者的实际需求，主打 LLM 全方位入门实践。本项目基于吴恩达老师大模型系列课程内容，对原课程内容进行筛选、翻译、复现和调优。

## 项目链接

- **GitHub仓库**: https://github.com/datawhalechina/llm-cookbook
- **在线阅读地址**: 面向开发者的LLM入门课程（在线版本）
- **PDF下载**: 面向开发者的LLM入门教程（PDF版本）

## 项目特色

### 1. 完全开源免费
- 所有内容完全免费开源
- 支持在线阅读和PDF下载
- 可自由学习和分享

### 2. 针对国内开发者优化
- 基于国内开发者的实际需求设计
- 考虑了中文语境下的LLM应用
- 解决国内访问限制问题

### 3. 系统化的课程体系
- 共包括11门课程
- 分为必修类和选修类
- 循序渐进的学习路径

### 4. 实战导向
- 提供可运行的Notebook代码
- 中英双语版本
- 实践案例丰富

## 课程体系

### 必修类课程（推荐初学者按顺序学习）

#### 1. 面向开发者的 Prompt Engineering
- 基于《ChatGPT Prompt Engineering for Developers》课程
- 学习如何构造Prompt
- 基于OpenAI API实现总结、推断、转换等功能
- **LLM开发的第一步**

#### 2. 搭建基于 ChatGPT 的问答系统
- 基于《Building Systems with the ChatGPT API》课程
- 开发完整的智能问答系统
- 学习大模型开发的新范式
- **大模型开发的实践基础**

#### 3. 使用 LangChain 开发应用程序
- 基于《LangChain for LLM Application Development》课程
- 深入学习LangChain框架
- 开发具备强大能力的应用程序
- **LLM应用开发的核心工具**

#### 4. 使用 LangChain 访问个人数据
- 基于《LangChain Chat with Your Data》课程
- 学习如何访问用户个人数据
- 开发个性化大模型应用
- **个性化LLM应用的关键**

### 选修类课程（掌握必修课后选择学习）

1. **使用 Gradio 搭建生成式 AI 应用**
   - 快速构建生成式AI的用户界面
   - Python接口程序开发

2. **评估改进生成式 AI**
   - 结合wandb工具
   - 系统化的跟踪和调试方法

3. **微调大语言模型**
   - 结合lamini框架
   - 本地基于个人数据微调开源LLM

4. **大模型与语义检索**
   - RAG检索增强生成
   - 多种高级检索技巧

5. **基于 Chroma 的高级检索**
   - Chroma向量数据库应用
   - 提升检索结果准确性

6. **搭建和评估高级 RAG 应用**
   - 构建高质量RAG系统
   - 关键技术和评估框架

7. **LangChain 的 Functions、Tools 和 Agents**
   - LangChain新语法
   - Agent构建方法

8. **Prompt 高级技巧**
   - CoT（Chain of Thought）
   - 自我一致性等多种高级技巧

## 学习指南

### 学习前准备

1. **至少一个 LLM API**
   - 最好是OpenAI API
   - 其他API需要参考相关教程修改代码

2. **Python能力**
   - 基础Python编程知识
   - 了解函数、类等基本概念

3. **开发环境**
   - 能够使用Python Jupyter Notebook
   - 建议使用VS Code或PyCharm

### 学习路径建议

1. **初学者路径**
   - 先完成4门必修课程
   - 掌握LLM开发的基础技能
   - 再根据兴趣选择选修课程

2. **进阶学习**
   - 完成RAG相关课程
   - 学习模型微调技术
   - 探索Agent开发

## 项目结构

```
content/    # 双语版代码，可运行的Notebook（更新频率最高）
docs/       # 必修类课程文字教程版（在线阅读源码）
figures/    # 图片文件
```

## 其他资源

### 视频资源
- 双语字幕视频地址：吴恩达 x OpenAI的Prompt Engineering课程专业翻译版
- 中英双语字幕下载：《ChatGPT提示工程》非官方版
- 视频讲解：面向开发者的Prompt Engineering讲解（数字游民大会）

### 社区支持
- GitHub Issues: 提出问题和建议
- Datawhale社区: 加入讨论和交流

## 项目亮点

1. **权威来源**: 基于吴恩达老师的官方课程
2. **中文化**: 完整的中文版本，便于学习
3. **可运行**: 所有代码经过验证，可直接运行
4. **双语对照**: 中英文双语，适合不同需求
5. **社区驱动**: 由Datawhale开源社区维护

## 适用人群

- 具备基础Python能力的开发者
- 想要入门LLM的开发者
- 需要快速上手LLM应用开发的技术人员
- 对生成式AI感兴趣的初学者

## 项目状态

- 维护状态: 活跃维护中
- 更新频率: 持续更新
- 星标数量: 10k+ stars（参考）
- 开源协议: CC BY-NC-SA 4.0

## 如何贡献

欢迎开发者参与项目：
- 复现新的吴恩达课程
- 改进教程内容
- 优化代码实现
- 提交Issue和PR

---

**最后更新**: 2026-03-12
**维护组织**: Datawhale
**开源协议**: 知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议
