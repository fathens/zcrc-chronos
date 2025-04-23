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

# ゼロショット予測のリクエストモデル
class ZeroShotPredictionRequest(BaseModel):
    """ゼロショット予測リクエストモデル"""
    context: str = Field(..., description="予測のためのコンテキスト情報（テキスト）")
    horizon: Optional[int] = Field(24, description="予測期間（ポイント数）")
    model_name: Optional[str] = Field("chronos_default", description="使用するモデル名")
    model_params: Optional[Dict[str, Any]] = Field(None, description="モデルパラメータ")

    class Config:
        schema_extra = {
            "example": {
                "context": "今後の株価は上昇傾向にあり、安定した成長が見込まれる。",
                "horizon": 24,
                "model_name": "chronos_default"
            }
        }

class ZeroShotPredictionResponse(BaseModel):
    """ゼロショット予測レスポンスモデル"""
    forecast_timestamp: List[datetime.datetime]
    forecast_values: List[float]
    model_name: str
    model_type: str = "zero_shot"
    context: str
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    metrics: Optional[Dict[str, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "forecast_timestamp": ["2023-01-01T03:00:00", "2023-01-01T04:00:00", "2023-01-01T05:00:00"],
                "forecast_values": [10.9, 11.3, 11.7],
                "model_name": "chronos_default",
                "model_type": "zero_shot",
                "context": "今後の株価は上昇傾向にあり、安定した成長が見込まれる。",
                "confidence_intervals": {
                    "lower_95": [10.5, 10.8, 11.0],
                    "upper_95": [11.3, 11.8, 12.4]
                },
                "metrics": {
                    "confidence": 0.7
                }
            }
        }

# ゼロショット予測エンドポイント
@app.post("/predict", response_model=ZeroShotPredictionResponse, tags=["prediction"])
async def zero_shot_predict(request: ZeroShotPredictionRequest):
    """
    ゼロショット予測を実行するエンドポイント

    テキストコンテキストに基づいて時系列データの予測を行います。
    学習データを必要とせず、コンテキスト情報のみから予測を生成します。
    """
    try:
        # 予測モデルの初期化
        predictor = TimeSeriesPredictor(
            model_name=request.model_name,
            model_params=request.model_params
        )

        # ゼロショット予測の実行
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            context=request.context,
            horizon=request.horizon
        )

        # レスポンスを作成
        response = ZeroShotPredictionResponse(
            forecast_timestamp=forecast_timestamps,
            forecast_values=forecast_values,
            model_name=request.model_name,
            model_type=metadata.get('model_type', 'zero_shot'),
            context=request.context,
            confidence_intervals=metadata.get('confidence_intervals'),
            metrics=metadata.get('metrics')
        )

        return response
    except Exception as e:
        logger.error(f"ゼロショット予測処理に失敗しました: {e}")
        raise HTTPException(status_code=500, detail=f"ゼロショット予測処理に失敗しました: {str(e)}")

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
