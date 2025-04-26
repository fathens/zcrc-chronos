"""
APIルーティングを定義するモジュール
"""

import datetime
import os
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from src.models.predictor import TimeSeriesPredictor

# 設定ファイルのパス
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "app_config.yaml",
)
MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "model_config.yaml",
)


# 設定の読み込み
def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise HTTPException(
            status_code=500, detail="サーバー設定の読み込みに失敗しました"
        )


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
                "timestamp": [
                    "2023-01-01T00:00:00",
                    "2023-01-01T01:00:00",
                    "2023-01-01T02:00:00",
                ],
                "values": [10.5, 11.2, 10.8],
            }
        }

    @field_validator("values")
    def validate_values_length(cls, v, info):
        """
        valuesの長さがtimestampの長さと一致することを検証
        """
        # 現在のモデルのデータを取得
        data = info.data

        # timestampが存在する場合、長さを比較
        if "timestamp" in data and len(data["timestamp"]) != len(v):
            raise ValueError("timestampとvaluesの長さが一致しません")

        return v


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
                "forecast_timestamp": [
                    "2023-01-01T03:00:00",
                    "2023-01-01T04:00:00",
                    "2023-01-01T05:00:00",
                ],
                "forecast_values": [10.9, 11.3, 11.7],
                "model_name": "chronos_default",
                "confidence_intervals": {
                    "lower_95": [10.5, 10.8, 11.0],
                    "upper_95": [11.3, 11.8, 12.4],
                },
                "metrics": {"mse": 0.15, "mae": 0.12},
            }
        }


class ZeroShotPredictionRequest(BaseModel):
    """ゼロショット予測リクエストモデル"""

    timestamp: List[datetime.datetime]
    values: List[float]
    horizon: int = 24
    frequency: Optional[str] = "H"
    model_name: Optional[str] = "chronos_default"
    model_params: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "timestamp": [
                    "2023-01-01T00:00:00",
                    "2023-01-01T01:00:00",
                    "2023-01-01T02:00:00",
                ],
                "values": [10.5, 11.2, 10.8],
                "horizon": 72,
                "frequency": "H",
                "model_name": "chronos_default",
                "model_params": {"seasonality_mode": "multiplicative"},
            }
        }

    @field_validator("values")
    def validate_values_length(cls, v, info):
        """
        valuesの長さがtimestampの長さと一致することを検証
        """
        # 現在のモデルのデータを取得
        data = info.data

        # timestampが存在する場合、長さを比較
        if "timestamp" in data and len(data["timestamp"]) != len(v):
            raise ValueError("timestampとvaluesの長さが一致しません")

        return v


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
                parameters=default_model["chronos"],
            )
        )

        return models
    except Exception as e:
        logger.error(f"モデル情報の取得に失敗しました: {e}")
        raise HTTPException(status_code=500, detail="モデル情報の取得に失敗しました")


# ゼロショット予測エンドポイント
@router.post(
    "/predict_zero_shot", response_model=PredictionResponse, tags=["prediction"]
)
async def predict_zero_shot(request: ZeroShotPredictionRequest):
    """
    時系列データに基づくゼロショット予測を実行
    """
    try:
        # 予測モデルの初期化
        predictor = TimeSeriesPredictor(
            model_name=request.model_name, model_params=request.model_params
        )

        # ゼロショット予測の実行
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            timestamp=request.timestamp, values=request.values, horizon=request.horizon
        )

        # レスポンスを作成
        response = PredictionResponse(
            forecast_timestamp=forecast_timestamps,
            forecast_values=forecast_values,
            model_name=request.model_name,
            confidence_intervals=metadata.get("confidence_intervals"),
            metrics=metadata.get("metrics"),
        )

        return response
    except Exception as e:
        logger.error(f"ゼロショット予測処理に失敗しました: {e}")
        raise HTTPException(
            status_code=500, detail=f"ゼロショット予測処理に失敗しました: {str(e)}"
        )
