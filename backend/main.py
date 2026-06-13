#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 后端API服务
FastAPI 应用入口，提供图片上传和检测结果API
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from api import detect_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="LAIGCD Detection API",
    description="AI生成内容检测服务 - 上传图片返回检测结果和热力图",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(detect_router, prefix="/api", tags=["detection"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LAIGCD Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/api/detect",
            "health": "/api/health",
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
