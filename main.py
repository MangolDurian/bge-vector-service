"""
中文文本向量（Embedding）API 服务
=================================
基于 BGE (BAAI General Embedding) 模型，提供中文文本向量生成 HTTP 接口。

功能：
    - POST /embed    ：批量生成文本向量
    - POST /vector   ：单条文本生成向量（兼容 Java NlpService）
    - POST /answer   ：自然语言查询生成向量（兼容 Java NlpService）
    - GET  /health   ：健康检查

启动方式：
    uvicorn main:app --host 0.0.0.0 --port 7999

项目结构：
    embedding-api/
    ├── main.py              # 主服务入口（本文件）
    ├── config.py            # 配置管理
    ├── requirements.txt     # Python 依赖
    └── docs/                # 项目文档
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings

# ---------------------------------------------------------------------------
# 全局变量：模型实例
# 模型在 lifespan 阶段加载，避免每次请求重复加载
# ---------------------------------------------------------------------------
_model = None


def _load_model():
    """加载 Embedding 模型到内存/显存"""
    from FlagEmbedding import FlagAutoModel

    print(f"[启动] 正在加载模型: {settings.MODEL_NAME} ...")
    model = FlagAutoModel.from_finetuned(
        settings.MODEL_NAME,
        query_instruction_for_retrieval=settings.QUERY_INSTRUCTION,
        use_fp16=settings.USE_FP16,
    )
    print("[启动] 模型加载完成")
    return model


# ---------------------------------------------------------------------------
# FastAPI 生命周期管理
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型，关闭时释放资源"""
    global _model
    _model = _load_model()
    yield
    _model = None


# ---------------------------------------------------------------------------
# 创建 FastAPI 应用实例
# ---------------------------------------------------------------------------
app = FastAPI(
    title="中文文本向量服务",
    description="基于 BGE 模型的中文文本 Embedding API，用于将文本转换为高维向量。",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS 跨域支持（允许前端页面从 file:// 或其他端口访问）
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求 / 响应模型定义
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    """向量生成请求体"""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_TEXTS_PER_REQUEST,
        description="待编码的文本列表，每项为一条中文文本",
    )


class EmbedResponse(BaseModel):
    """向量生成响应体"""

    embeddings: list[list[float]] = Field(description="向量列表，与输入文本一一对应")
    dim: int = Field(description="向量维度")
    count: int = Field(description="文本数量")


class ErrorResponse(BaseModel):
    """错误响应体"""

    error: str = Field(description="错误信息")


class KeywordRequest(BaseModel):
    """兼容 Java NlpService 的请求体格式"""

    keyword: str = Field(..., description="待编码的文本内容")


class VectorResponse(BaseModel):
    """兼容 Java NlpService 的响应体格式"""

    vector: list[float] = Field(description="向量数据，与输入文本对应的浮点数列表")


# ---------------------------------------------------------------------------
# 接口实现
# ---------------------------------------------------------------------------
@app.post(
    "/embed",
    response_model=EmbedResponse,
    summary="生成文本向量",
    description="接收中文文本列表，返回对应的向量表示（内积相似度）。",
    responses={400: {"model": ErrorResponse}},
)
def embed(req: EmbedRequest):
    """
    批量生成文本向量。

    请求示例：
        {"texts": ["你好世界", "这是一段测试文本"]}

    返回示例：
        {"embeddings": [[0.01, -0.03, ...], [0.05, 0.07, ...]], "dim": 768, "count": 2}
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    # 调用模型进行编码，返回 numpy 数组，转为 Python list 以便 JSON 序列化
    vectors = _model.encode(req.texts, batch_size=settings.BATCH_SIZE).tolist()

    return EmbedResponse(
        embeddings=vectors,
        dim=len(vectors[0]),
        count=len(vectors),
    )


@app.get("/health", summary="健康检查", description="检查服务是否正常运行")
def health():
    """健康检查接口，可用于负载均衡器或监控系统探活"""
    return {
        "status": "ok" if _model is not None else "loading",
        "model": settings.MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# 兼容 Java NlpService 的接口
# ---------------------------------------------------------------------------
# Java 端调用方式：
#   HttpRequest.post(url).body("{\"keyword\": \"文本\"}").execute()
#   响应解析：JSONObject.getStr("vector") 或 JSONArray.getDouble(i)
#
# /vector 接口：用于字段值向量化（入库时调用）
# /answer 接口：用于自然语言查询向量化（检索时调用）
# ---------------------------------------------------------------------------

@app.post(
    "/vector",
    summary="字段值向量化",
    description="将单个字段值转换为向量，兼容 Java NlpService.word2Vector / calculateVector 接口。",
)
def vector(req: KeywordRequest):
    """
    字段值向量化接口（兼容 Java NlpService）。

    请求示例：
        {"keyword": "北京市2025年吉林1号0.5米分辨率全色遥感影像"}

    返回示例：
        {"vector": [0.0123, -0.0345, ...]}
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    # 将空值或纯空白视为无效输入，返回空向量
    if not req.keyword or not req.keyword.strip():
        return VectorResponse(vector=[])

    # 编码单条文本，取第一个结果
    vector_data = _model.encode([req.keyword], batch_size=settings.BATCH_SIZE).tolist()[0]

    return VectorResponse(vector=vector_data)


@app.post(
    "/answer",
    summary="自然语言查询向量化",
    description="将自然语言查询转换为向量，兼容 Java NlpService 的检索端调用。",
)
def answer(req: KeywordRequest):
    """
    自然语言查询向量化接口（兼容 Java NlpService）。

    请求示例：
        {"keyword": "查询大兴机场最新的北京3号融合影像"}

    返回示例：
        {"vector": [0.0123, -0.0345, ...]}
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    if not req.keyword or not req.keyword.strip():
        return VectorResponse(vector=[])

    vector_data = _model.encode([req.keyword], batch_size=settings.BATCH_SIZE).tolist()[0]

    return VectorResponse(vector=vector_data)


# ---------------------------------------------------------------------------
# 直接运行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
    )
