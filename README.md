<p align="center">
  <h1 align="center">BGE Vector Service</h1>
  <p align="center">轻量级中文文本向量（Embedding）API 服务</p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue" />
    <img src="https://img.shields.io/badge/FastAPI-0.104+-green" />
    <img src="https://img.shields.io/badge/Model-bge--base--zh--v1.5-yellow" />
    <img src="https://img.shields.io/badge/License-MIT-informational" />
  </p>
</p>

---

## 项目简介

BGE Vector Service 是一个基于 [BGE (BAAI General Embedding)](https://github.com/FlagOpen/FlagEmbedding) 模型的中文文本向量生成服务。通过简洁的 HTTP 接口，将中文文本转换为高质量的高维向量，可广泛应用于：

- 语义搜索与检索增强（RAG）
- 文本相似度计算
- 文本聚类与分类
- 推荐系统特征提取

**核心特点**

| 特性 | 说明 |
|------|------|
| 开箱即用 | 安装依赖后一行命令启动，无需额外配置 |
| 高性能推理 | 支持 FP16 半精度加速，自动识别 GPU/CPU |
| 配置灵活 | 所有参数支持环境变量覆盖，无需改代码 |
| 接口规范 | RESTful API + 自动生成 Swagger 文档 |
| 轻量部署 | 仅 3 个核心文件，依赖精简 |

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 高性能异步框架，自动生成 OpenAPI 文档 |
| ASGI 服务器 | Uvicorn | 基于 uvloop 的高性能 ASGI 实现 |
| 向量模型 | FlagEmbedding (BGE) | 智源研究院开源的中文向量模型 |
| 数据校验 | Pydantic v2 | 请求/响应自动校验与序列化 |

---

## 快速开始

### 环境要求

- Python 3.9+
- pip（最新版）
-（可选）CUDA 11.7+ ，用于 GPU 加速推理

### 安装与启动

```bash
# 克隆项目
git clone https://github.com/MangolDurian/bge-vector-service.git
cd bge-vector-service

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

首次启动会自动从 HuggingFace 下载模型（约 400MB），后续使用本地缓存。

服务启动后访问：
- API 地址：`http://localhost:7999`
- Swagger 文档：`http://localhost:7999/docs`
- 前端 UI（可选）：`http://localhost:8001`

> 模型下载慢？设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 接口说明

### 生成文本向量

**POST** `/embed`

```bash
curl -X POST http://localhost:7999/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["自然语言处理", "深度学习"]}'
```

响应：

```json
{
  "embeddings": [[0.0123, -0.0345, ...], [0.0543, 0.0789, ...]],
  "dim": 768,
  "count": 2
}
```

### 健康检查

**GET** `/health`

```bash
curl http://localhost:7999/health
```

响应：

```json
{
  "status": "ok",
  "model": "BAAI/bge-base-zh-v1.5"
}
```

### 计算语义相似度

```python
import requests, numpy as np

resp = requests.post("http://localhost:7999/embed",
    json={"texts": ["今天天气真好", "今天阳光明媚"]})
embs = np.array(resp.json()["embeddings"])
print(f"相似度: {np.dot(embs[0], embs[1]):.4f}")
```

---

## 配置项

所有配置均支持环境变量覆盖，无需修改代码：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-zh-v1.5` | 模型名称 |
| `EMBEDDING_USE_FP16` | `true` | 半精度推理 |
| `EMBEDDING_DEVICE` | 自动选择 | 推理设备（cuda / cpu） |
| `EMBEDDING_BATCH_SIZE` | `32` | 编码批大小 |
| `EMBEDDING_HOST` | `0.0.0.0` | 监听地址 |
| `EMBEDDING_PORT` | `7999` | 后端 API 监听端口 |
| `EMBEDDING_MAX_TEXTS` | `128` | 单次最大文本数 |

---

## 前端 UI（可选）

`web/index.html` 是辅助测试页面，可按需用独立端口部署：

```bash
python -m http.server 8001 -d web
```

访问 `http://localhost:8001` 打开页面。页面会自动调用同一主机的后端 API：`http://localhost:7999`。

---

## 可切换模型

通过修改 `EMBEDDING_MODEL_NAME` 环境变量即可切换模型：

| 模型 | 向量维度 | 最大长度 | 大小 | 适用场景 |
|------|---------|---------|------|---------|
| `BAAI/bge-small-zh-v1.5` | 512 | 512 | ~100MB | 轻量快速 |
| `BAAI/bge-base-zh-v1.5` | 768 | 512 | ~400MB | 平衡之选（默认） |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 | ~1.2GB | 最高精度 |
| `BAAI/bge-m3` | 1024 | 8192 | ~1.2GB | 多语言 + 长文本 |

---

## 项目结构

```
bge-vector-service/
├── main.py                                    # 主服务入口，路由与接口定义
├── config.py                                  # 集中配置管理
├── requirements.txt                           # Python 依赖声明
├── .gitignore
└── docs/
    ├── API接口文档.md                          # 接口详细说明
    ├── 使用文档.md                             # 安装与调用指南
    ├── 架构文档.md                             # 系统架构与技术选型
    ├── 升级指南.md                             # 后续升级路径
    └── BGE_Vector_Service.postman_collection.json  # Postman 接口集合
```

---

## 文档导航

| 文档 | 说明 |
|------|------|
| [API 接口文档](docs/API接口文档.md) | 接口请求/响应详细说明 |
| [使用文档](docs/使用文档.md) | 安装、启动、调用示例 |
| [架构文档](docs/架构文档.md) | 系统架构与技术选型 |
| [升级指南](docs/升级指南.md) | Docker 部署、认证、Reranker 等升级方向 |

---

## 致谢

- [FlagEmbedding / BGE](https://github.com/FlagOpen/FlagEmbedding) — 智源研究院开源的文本向量模型
- [FastAPI](https://fastapi.tiangolo.com/) — 高性能 Python Web 框架

## License

[MIT](https://opensource.org/licenses/MIT)
