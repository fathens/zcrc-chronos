"""
APIサーバーの設定と初期化を行うモジュール
"""
import os
import yaml
import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.api.routes import router
from src.models.predictor import TimeSeriesPredictor

# 設定ファイルのパス
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "app_config.yaml")

def load_config():
    """
    アプリケーション設定を読み込む
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise HTTPException(status_code=500, detail="サーバー設定の読み込みに失敗しました")

# アプリケーション設定の読み込み
config = load_config()

# FastAPIアプリケーションの初期化
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
    docs_url=f"{config['api']['prefix']}/docs",
    redoc_url=f"{config['api']['prefix']}/redoc",
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(router, prefix=config["api"]["prefix"])

# ルートエンドポイント
@app.get("/")
async def root():
    """
    ルートエンドポイント - APIの基本情報を返す
    """
    return {
        "name": config["api"]["title"],
        "version": config["api"]["version"],
        "description": config["api"]["description"],
        "docs": f"{config['api']['prefix']}/docs",
    }

# ヘルスチェックエンドポイント
@app.get(f"{config['api']['prefix']}/health")
async def health_check():
    """
    ヘルスチェックエンドポイント - APIの状態を返す
    """
    return {
        "status": "healthy",
        "api_version": config["api"]["version"],
    }

# 注：ゼロショット予測のエンドポイントはroutes.pyに移動しました
# routes.pyの/predict_zero_shotエンドポイントを使用してください

# サーバー起動関数
def start_server():
    """
    uvicornサーバーを起動する
    """
    uvicorn.run(
        "src.api.server:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["debug"],
        log_level=config["logging"]["level"].lower(),
    )

if __name__ == "__main__":
    # 直接実行された場合はサーバーを起動
    start_server()
