# 机器翻译系统（中文 ↔ 英文）

## 项目简介
课程工程实践项目：  
从头训练一个 **中文→英文** 机器翻译模型，Web服务部署 + Gradio界面，支持双向翻译（加分）。

难度：🌟🌟

## 功能目标
- 输入中文，输出英文
- 输入英文，输出中文（加分）
- Web API：FastAPI
- 前端界面：Gradio
- 模型：Seq2Seq（BiLSTM） / Transformer（可选）

## 技术栈
- 框架：PyTorch
- 后端：FastAPI + Uvicorn
- 前端：Gradio
- 数据：AI Challenger 2017 英中数据集

## 数据集
- 来源：https://github.com/foamliu/Transformer?tab=readme-ov-file
- 内容：中英句对
- 下载后放：`data/raw/`

## 目录结构
machine-translation-project/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│ ├── raw/
│ ├── processed/
│ └── scripts/
├── models/
├── deployment/
├── frontend/
└── docs/
├── proposal/
├── midterm/
└── final/


## 运行方式（后续阶段）
```bash
# 安装依赖
pip install -r requirements.txt

# 启动后端服务
uvicorn deployment.api:app --reload

# 启动 Gradio 界面
python frontend/gradio_app.py

小组分工
数据处理、模型训练：bt
后端开发：Xxx
前端设计：XXX



---

### 3. `data/README.md`（数据集说明）
```markdown
# 数据集说明
## 数据集名称
AI Challenger 2017 中英翻译平行语料

## 数据集来源
官方开源地址：https://github.com/foamliu/Transformer

## 数据存放路径
- 原始数据：`data/raw/`
- 预处理数据：`data/processed/`

## 数据内容
包含大规模中文-英文平行句对，覆盖日常对话、新闻文本、通用领域语句，用于模型训练、验证与测试。

## 数据用途
1. 构建中英词汇表
2. 生成训练集、验证集、测试集
3. 用于 Seq2Seq 模型的端到端训练

