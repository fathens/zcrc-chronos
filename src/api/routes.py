"""
APIルーティングを定義するモジュール
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Dict, Any, Optional
import datetime
import yaml
import os
from loguru import logger

# 設定ファイルのパス
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "app_config.yaml")
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "model_config.yaml")

# 設定の読み込み
def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise HTTPException(status_code=500, detail="サーバー設定の読み込みに失敗しました")

config = load_config()

# APIルーターの作成
router = APIRouter()

# リクエスト/レスポンスモデル
class TimeSeriesData(BaseModel):
    """時系列データモデル"""
    timestamp: List[datetime.datetime]
    values: List[float]

    class Config:
        schema_extra = {
            "example": {
                "timestamp": ["2023-01-01T00:00:00", "2023-01-01T01:00:00", "2023-01-01T02:00:00"],
                "values": [10.5, 11.2, 10.8]
            }
        }

    @field_validator('values')
    def validate_values_length(cls, v, info):
        """
        valuesの長さがtimestampの長さと一致することを検証
        """
        # 現在のモデルのデータを取得
        data = info.data

        # timestampが存在する場合、長さを比較
        if 'timestamp' in data and len(data['timestamp']) != len(v):
            raise ValueError("timestampとvaluesの長さが一致しません")

        return v

class PredictionRequest(BaseModel):
    """予測リクエストモデル"""
    data: TimeSeriesData
    horizon: int = 24  # 予測期間（デフォルト24時間）
    model_name: Optional[str] = "chronos_default"
    model_params: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "timestamp": ["2023-01-01T00:00:00", "2023-01-01T01:00:00", "2023-01-01T02:00:00"],
                    "values": [10.5, 11.2, 10.8]
                },
                "horizon": 24,
                "model_name": "chronos_default",
                "model_params": {
                    "seasonality_mode": "multiplicative"
                }
            }
        }

class PredictionResponse(BaseModel):
    """予測レスポンスモデル"""
    forecast_timestamp: List[datetime.datetime]
    forecast_values: List[float]
    model_name: str
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    metrics: Optional[Dict[str, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "forecast_timestamp": ["2023-01-01T03:00:00", "2023-01-01T04:00:00", "2023-01-01T05:00:00"],
                "forecast_values": [10.9, 11.3, 11.7],
                "model_name": "chronos_default",
                "confidence_intervals": {
                    "lower_95": [10.5, 10.8, 11.0],
                    "upper_95": [11.3, 11.8, 12.4]
                },
                "metrics": {
                    "mse": 0.15,
                    "mae": 0.12
                }
            }
        }

class ModelInfo(BaseModel):
    """モデル情報モデル"""
    name: str
    version: str
    description: str
    parameters: Dict[str, Any]

# モデル一覧エンドポイント
@router.get("/models", response_model=List[ModelInfo], tags=["models"])
async def get_models():
    """
    利用可能な予測モデルの一覧を取得
    """
    try:
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_config = yaml.safe_load(f)

        # モデル情報のリストを作成
        models = []
        default_model = model_config["default_model"]
        models.append(
            ModelInfo(
                name=default_model["name"],
                version=default_model["version"],
                description=default_model["description"],
                parameters=default_model["chronos"]
            )
        )

        return models
    except Exception as e:
        logger.error(f"モデル情報の取得に失敗しました: {e}")
        raise HTTPException(status_code=500, detail="モデル情報の取得に失敗しました")

# 予測エンドポイント
@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: PredictionRequest):
    """
    時系列データの予測を実行
    """
    try:
        # ここでは実際の予測処理は実装せず、ダミーデータを返す
        # 実際の実装では、chronos-boltを使用して予測を行う

        # ダミーの予測結果を生成
        forecast_timestamp = [
            request.data.timestamp[-1] + datetime.timedelta(hours=i+1)
            for i in range(request.horizon)
        ]

        import random
        last_value = request.data.values[-1]
        forecast_values = [
            last_value + random.uniform(-0.5, 0.5)
            for _ in range(request.horizon)
        ]

        # レスポンスを作成
        response = PredictionResponse(
            forecast_timestamp=forecast_timestamp,
            forecast_values=forecast_values,
            model_name=request.model_name,
            confidence_intervals={
                "lower_95": [v - 0.5 for v in forecast_values],
                "upper_95": [v + 0.5 for v in forecast_values]
            },
            metrics={
                "mse": 0.15,
                "mae": 0.12
            }
        )

        return response
    except Exception as e:
        logger.error(f"予測処理に失敗しました: {e}")
        raise HTTPException(status_code=500, detail=f"予測処理に失敗しました: {str(e)}")
