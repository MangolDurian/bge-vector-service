"""
配置模块
========
集中管理所有可配置参数，方便后续调整和升级。

使用方式：
    from config import settings
    print(settings.MODEL_NAME)
"""

import os


class Settings:
    """全局配置类，所有参数均可通过环境变量覆盖"""

    # ========== 模型配置 ==========
    # 模型名称（HuggingFace 上的模型标识符）
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")

    # 检索场景下为 query 添加的指令前缀，用于提升检索效果
    # 如果只是做文本相似度计算而非检索，可以留空
    QUERY_INSTRUCTION: str = os.getenv(
        "EMBEDDING_QUERY_INSTRUCTION",
        "为这个句子生成表示以用于检索相关文章：",
    )

    # 是否使用 FP16 半精度推理（减少显存占用，推理速度更快）
    USE_FP16: bool = os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true"

    # 推理设备，默认自动选择（优先 GPU，无 GPU 则用 CPU）
    DEVICE: str = os.getenv("EMBEDDING_DEVICE", None)  # None 表示自动选择

    # 批量编码时的 batch size（影响推理速度和内存占用）
    BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # ========== 服务配置 ==========
    # 服务监听地址
    HOST: str = os.getenv("EMBEDDING_HOST", "0.0.0.0")

    # 服务监听端口
    PORT: int = int(os.getenv("EMBEDDING_PORT", "7999"))

    # 单次请求最大文本数量（防止内存溢出）
    MAX_TEXTS_PER_REQUEST: int = int(os.getenv("EMBEDDING_MAX_TEXTS", "128"))

    # 单条文本最大长度（超出会截断）
    MAX_TEXT_LENGTH: int = int(os.getenv("EMBEDDING_MAX_TEXT_LENGTH", "512"))


# 全局单例，其他模块直接 import 使用
settings = Settings()
