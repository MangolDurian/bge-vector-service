# API 接口文档

> 中文文本向量（Embedding）服务接口说明
> 基础地址：`http://<host>:8000`

---

## 1. 生成文本向量

**POST** `/embed`

将中文文本列表转换为高维向量，返回结果可用于语义搜索、文本聚类、相似度计算等场景。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `texts` | `string[]` | 是 | 待编码文本列表，最多 128 条 |

#### 请求示例

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["你好世界", "这是一段测试文本"]
  }'
```

### 响应

| 字段 | 类型 | 说明 |
|------|------|------|
| `embeddings` | `float[][]` | 向量列表，与输入一一对应 |
| `dim` | `int` | 向量维度（bge-base-zh-v1.5 为 768） |
| `count` | `int` | 返回向量数量 |

#### 响应示例

```json
{
  "embeddings": [
    [0.0123, -0.0345, 0.0678, ...],
    [0.0543, 0.0789, -0.0123, ...]
  ],
  "dim": 768,
  "count": 2
}
```

### 错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 400 | 请求参数错误（texts 为空或超过上限） |
| 503 | 模型尚未加载完成 |

---

## 2. 健康检查

**GET** `/health`

用于负载均衡器或监控系统探活。

#### 请求示例

```bash
curl http://localhost:8000/health
```

#### 响应示例

```json
{
  "status": "ok",
  "model": "BAAI/bge-base-zh-v1.5"
}
```

| `status` 值 | 含义 |
|-------------|------|
| `ok` | 服务正常，模型已加载 |
| `loading` | 模型正在加载中 |

---

## 3. 交互式 API 文档

服务启动后，浏览器访问以下地址可查看自动生成的 Swagger UI 文档：

- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`

---

## 4. 相似度计算说明

获取向量后，通过**内积（点积）** 计算相似度：

```python
import numpy as np

# 假设 emb_a 和 emb_b 是两个文本的向量
similarity = np.dot(emb_a, emb_b)
```

值越大表示语义越相似，范围大致在 -1 到 1 之间。
