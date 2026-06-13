#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD API路由模块
处理图片上传和检测请求
"""

import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import logging

from detector import DetectionService

logger = logging.getLogger(__name__)

# 创建路由
detect_router = APIRouter()

# 全局检测服务实例
detector_service: Optional[DetectionService] = None


def get_detector_service() -> DetectionService:
    """获取检测服务实例（单例）"""
    global detector_service
    if detector_service is None:
        detector_service = DetectionService()
    return detector_service


@detect_router.post("/detect")
async def detect_image(
    image: UploadFile = File(..., description="待检测的图片文件"),
    threshold: Optional[float] = Form(None, description="检测阈值，默认使用 0.68")
):
    """
    检测上传的图片是否为AI生成

    Args:
        image: 上传的图片文件
        threshold: 可选的检测阈值

    Returns:
        JSON响应包含：
        - prediction: "Real" 或 "Fake"
        - confidence: 置信度 (0-1)
        - fake_probability: 伪造概率
        - spatial_overlay: base64编码的空域热力图叠加图
        - frequency_overlay: base64编码的频域热力图叠加图
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

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"检测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@detect_router.get("/info")
async def get_info():
    """获取模型信息"""
    service = get_detector_service()
    return service.get_model_info()
