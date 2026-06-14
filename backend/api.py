#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD API路由模块
处理图片上传和检测请求
"""

import io
import asyncio
import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import logging

from detector import DetectionService
from fakevlm_service import FakeVLMService

logger = logging.getLogger(__name__)

# 创建路由
detect_router = APIRouter()

# 全局检测服务实例
detector_service: Optional[DetectionService] = None
# 全局 FakeVLM 解释服务实例（第二阶段）
fakevlm_service: Optional[FakeVLMService] = None


def get_detector_service() -> DetectionService:
    """获取检测服务实例（单例）"""
    global detector_service
    if detector_service is None:
        detector_service = DetectionService()
    return detector_service


def get_fakevlm_service() -> FakeVLMService:
    """获取 FakeVLM 解释服务实例（单例，懒加载）"""
    global fakevlm_service
    if fakevlm_service is None:
        fakevlm_service = FakeVLMService()
    return fakevlm_service


@detect_router.post("/detect")
async def detect_image(
    image: UploadFile = File(..., description="待检测的图片文件"),
    threshold: Optional[float] = Form(None, description="检测阈值，默认使用 0.68"),
    explain: bool = Form(True, description="是否调用 FakeVLM 生成理由，默认开启"),
):
    """
    检测上传的图片是否为AI生成

    Args:
        image: 上传的图片文件
        threshold: 可选的检测阈值
        explain: 是否调用第二阶段 FakeVLM 生成自然语言理由（默认开启）

    Returns:
        JSON响应包含：
        - prediction: "Real" 或 "Fake"
        - confidence: 置信度 (0-1)
        - fake_probability: 伪造概率
        - threshold: 本次使用的阈值
        - spatial_overlay: base64编码的空域热力图叠加图
        - frequency_overlay: base64编码的频域热力图叠加图
        - reasoning: FakeVLM 给出的真伪理由（英文）；explain=False 或调用失败时为 null
    """
    try:
        # 读取图片
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # 获取检测服务
        service = get_detector_service()

        # 执行检测
        result = service.detect(img, threshold=threshold)

        # 将overlay图像转为base64
        def img_to_base64(img_array):
            """将numpy图像数组转为base64字符串"""
            from PIL import Image
            import numpy as np
            img_pil = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype('uint8'))
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "fake_probability": result["fake_probability"],
            "threshold": result["threshold"],
            "spatial_overlay": img_to_base64(result["spatial_overlay"]),
        }

        # 只有频域热力图存在时才添加
        if result.get("frequency_overlay") is not None:
            response["frequency_overlay"] = img_to_base64(result["frequency_overlay"])

        # 第二阶段：调用 FakeVLM 生成自然语言理由（基于第一阶段判定）。
        # 默认开启；explain=False 时跳过。模型推理较重，放入线程池避免阻塞事件循环。
        # 任何失败都降级为 reasoning=None，不影响第一阶段的权威判定与热力图返回。
        reasoning = None
        if explain:
            try:
                vlm = get_fakevlm_service()
                reasoning = await asyncio.to_thread(
                    vlm.explain, img, result["prediction"]
                )
            except Exception as e:
                logger.error(f"FakeVLM 生成理由失败，已降级为 null: {e}", exc_info=True)
                reasoning = None
        response["reasoning"] = reasoning

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"检测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@detect_router.get("/info")
async def get_info():
    """获取模型信息"""
    service = get_detector_service()
    return service.get_model_info()
