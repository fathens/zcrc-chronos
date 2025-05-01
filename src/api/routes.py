"""
APIルーティングを定義するモジュール
"""

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
        logger.error("設定ファイルの読み込みに失敗しました: " + str(e))
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": [
                    "2023-01-01T00:00:00",
                    "2023-01-01T01:00:00",
                    "2023-01-01T02:00:00",
                ],
                "values": [10.5, 11.2, 10.8],
            }
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

    model_config = {
        "json_schema_extra": {
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
    }


class ZeroShotPredictionRequest(BaseModel):
    """ゼロショット予測リクエストモデル

    時系列データに基づいて将来の予測を行うためのリクエストモデルです。
    このモデルは履歴データと予測設定を含みます。

    Attributes:
        timestamp (List[datetime.datetime]):
            時系列データの時間情報を表すタイムスタンプのリスト。
            各値に対応する時間点を示します。

        values (List[float]):
            時系列データの実測値のリスト。
            timestampリストと同じ長さである必要があります。

        forecast_until (datetime.datetime):
            予測を行う終了時点。
            この時点までの将来値が予測されます。

        model_name (Optional[str]):
            予測に使用するモデルの名前。デフォルトは "chronos_default"。
            利用可能なモデルは GET /models エンドポイントで確認できます。

        model_params (Optional[Dict[str, Any]]):
            モデルに渡す追加パラメータ。モデルごとに異なるパラメータをサポート。
            例: {"seasonality_mode": "multiplicative"} - 季節性の扱い方を指定
                {"growth": "linear"} - トレンドの成長タイプを指定

    Example:
        ```json
        {
            "timestamp": [
                "2023-01-01T00:00:00",
                "2023-01-01T01:00:00",
                "2023-01-01T02:00:00"
            ],
            "values": [10.5, 11.2, 10.8],
            "forecast_until": "2023-01-04T02:00:00",
            "model_name": "chronos_default",
            "model_params": {"seasonality_mode": "multiplicative"}
        }
        ```
    """

    timestamp: List[datetime.datetime]
    values: List[float]
    forecast_until: datetime.datetime
    model_name: Optional[str] = "chronos_default"
    model_params: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": [
                    "2023-01-01T00:00:00",
                    "2023-01-01T01:00:00",
                    "2023-01-01T02:00:00",
                ],
                "values": [10.5, 11.2, 10.8],
                "forecast_until": "2023-01-04T02:00:00",
                "model_name": "chronos_default",
                "model_params": {"seasonality_mode": "multiplicative"},
            }
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
        logger.error("モデル情報の取得に失敗しました: " + str(e))
        raise HTTPException(status_code=500, detail="モデル情報の取得に失敗しました")


# ゼロショット予測エンドポイント
@router.post("/predict_zero_shot", response_model=PredictionResponse)
async def predict_zero_shot(request: ZeroShotPredictionRequest):
    """
    時系列データに基づくゼロショット予測を実行

    - **timestamp**: 時系列データのタイムスタンプのリスト
    - **values**: 時系列データの値のリスト
    - **forecast_until**: 予測したい時点（datetime形式）
    - **model_name**: 使用する予測モデルの名前
    - **model_params**: モデルに渡す追加パラメータ
    """
    try:
        logger.info("ゼロショット予測APIが呼び出されました")

        # 時系列データの正規化
        normalized_timestamps, normalized_values = normalize_time_series_data(
            request.timestamp, request.values, interpolation_method="auto"
        )

        # 最後のタイムスタンプを取得
        latest_timestamp = max(normalized_timestamps)

        # タイムスタンプの間隔を計算
        if len(normalized_timestamps) >= 2:
            # 実データから間隔を計算
            delta = normalized_timestamps[1] - normalized_timestamps[0]
        else:
            # データが1点しかない場合はエラーを発生させる
            raise HTTPException(
                status_code=400,
                detail="予測には少なくとも2つのデータポイントが必要です",
            )

        # 予測期間の計算
        time_difference = request.forecast_until - latest_timestamp

        # 時間差をdelta単位のポイント数に変換
        if delta.total_seconds() <= 0:
            raise HTTPException(
                status_code=400,
                detail="タイムスタンプの間隔が正しくありません（間隔がゼロまたは負の値）",
            )

        prediction_points = int(time_difference.total_seconds() / delta.total_seconds())

        # 予測ポイント数が0以下の場合はエラー
        if prediction_points <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "予測時点が最新のデータポイント以前です。"
                    "予測時点: "
                    + str(request.forecast_until)
                    + ", 最新のデータポイント: "
                    + str(latest_timestamp)
                ),
            )

        logger.info(
            "予測ポイント数: "
            + str(prediction_points)
            + ", 予測時点: "
            + str(request.forecast_until)
        )

        # 予測モデルの初期化
        predictor = TimeSeriesPredictor(
            model_name=request.model_name, model_params=request.model_params
        )

        # ゼロショット予測の実行
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            timestamp=normalized_timestamps,
            values=normalized_values,
            horizon=prediction_points,
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
        logger.error("ゼロショット予測処理に失敗しました: " + str(e))
        raise HTTPException(
            status_code=500, detail="ゼロショット予測処理に失敗しました: " + str(e)
        )


def normalize_time_series_data(
    timestamps: List[datetime.datetime],
    values: List[float],
    interpolation_method: str = "auto",
) -> Tuple[List[datetime.datetime], List[float]]:
    """
    時系列データを均等な間隔に正規化する関数

    最初と最後のタイムスタンプ間の時間を、データポイントの数に基づいて
    均等に分割し、指定された補間方法によって新しい値を計算します。

    Args:
        timestamps: 元のタイムスタンプのリスト
        values: 元の値のリスト
        interpolation_method: 補間方法。以下の値が使用可能:
            - "auto": データの特性に基づいて自動的に最適な方法を選択（デフォルト）
            - "linear": 線形補間 - 2点間を直線で結ぶ。安定していて予測可能。
            - "time": 時間インデックスを考慮した補間 - 不規則な時間間隔のデータに適している。
            - "cubic": 3次スプライン補間 - より滑らかな曲線だが、オーバーシュートの可能性がある。
            - "nearest": 最近傍補間 - 離散的な値を維持したい場合に有用。
            - "quadratic": 2次スプライン補間 - 線形と3次の中間的な滑らかさ。
            - "spline": スプライン補間 - より高度な滑らかさが必要な場合。
            - "polynomial": 多項式補間 - 少数のデータ点に対して有効だが、不安定になりやすい。

    Returns:
        正規化されたタイムスタンプと値のタプル
    """
    if not timestamps or not values:
        return timestamps, values

    # 有効な補間方法のリスト
    valid_methods = [
        "auto",
        "linear",
        "time",
        "cubic",
        "nearest",
        "quadratic",
        "spline",
        "polynomial",
        "zero",
        "slinear",
        "akima",
        "pchip",
    ]

    # 補間方法の検証
    if interpolation_method not in valid_methods:
        logger.warning(
            "無効な補間方法: " + interpolation_method + ". 'auto'に切り替えます。"
        )
        interpolation_method = "auto"

    # 自動補間方法選択
    if interpolation_method == "auto":
        interpolation_method = _determine_best_interpolation_method(timestamps, values)
        logger.info("自動選択された補間方法: " + interpolation_method)

    # pandasのDataFrameを作成
    df = pd.DataFrame({"timestamp": timestamps, "value": values})

    # タイムスタンプをインデックスに設定
    df.set_index("timestamp", inplace=True)

    # 開始時刻と終了時刻を取得
    start_time = min(timestamps)
    end_time = max(timestamps)

    # 全体の時間範囲を計算（秒単位）
    total_duration = (end_time - start_time).total_seconds()

    # データポイントの数に基づいて均等な間隔を計算
    # 少なくとも元のデータと同じポイント数を維持
    num_points = len(timestamps)

    # 間隔を計算（秒単位）- 少なくとも1秒以上の間隔を確保
    interval_seconds = (
        max(1, total_duration / (num_points - 1)) if num_points > 1 else 1
    )

    # 均等な間隔の新しい時間インデックスを作成
    new_timestamps = []
    current_time = start_time

    while current_time <= end_time:
        new_timestamps.append(current_time)
        current_time += datetime.timedelta(seconds=interval_seconds)

    # 新しいタイムスタンプ数が元のデータポイント数よりも少なくならないように調整
    if len(new_timestamps) < num_points:
        # 間隔を調整して、より細かい時間間隔を作成
        adjusted_interval = total_duration / (num_points + 1)
        new_timestamps = []
        current_time = start_time

        while current_time <= end_time:
            new_timestamps.append(current_time)
            current_time += datetime.timedelta(seconds=adjusted_interval)

    # 補間方法に応じた追加パラメータの設定
    interpolation_kwargs = {}
    if interpolation_method in ["spline", "polynomial"]:
        # スプラインと多項式には追加のパラメータが必要
        interpolation_kwargs["order"] = 3  # デフォルトの次数

    # 補間方法に応じたデータの前処理
    # 一部の方法では、データの前処理が必要になる場合がある
    if interpolation_method == "time" and df.index.inferred_type != "datetime64":
        # timeメソッドはdatetimeインデックスが必要
        logger.warning(
            "'time'補間方法はdatetimeインデックスが必要です。'linear'に切り替えます。"
        )
        interpolation_method = "linear"

    try:
        # 元のデータフレームを新しいタイムスタンプで再インデックス化し、指定された方法で補間
        resampled_df = df.reindex(new_timestamps).interpolate(
            method=interpolation_method, **interpolation_kwargs
        )

        # 結果を返す
        return new_timestamps, resampled_df["value"].tolist()
    except Exception as e:
        # 補間に失敗した場合はエラーをログに記録し、線形補間にフォールバック
        logger.error(
            "補間方法 '"
            + interpolation_method
            + "' でエラーが発生しました: "
            + str(e)
            + ". 線形補間を使用します。"
        )
        resampled_df = df.reindex(new_timestamps).interpolate(method="linear")
        return new_timestamps, resampled_df["value"].tolist()


def _determine_best_interpolation_method(
    timestamps: List[datetime.datetime], values: List[float]
) -> str:
    """
    時系列データの特性を分析して最適な補間方法を判別する関数

    Args:
        timestamps: タイムスタンプのリスト
        values: 値のリスト

    Returns:
        最適な補間方法の文字列
    """
    # データ点が少ない場合は線形補間が最も安全
    if len(timestamps) <= 3:
        return "linear"

    # 時間間隔の規則性を計算
    time_diffs = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    time_diff_array = np.array(time_diffs)

    # 時間間隔の変動係数（標準偏差/平均）を計算
    # 変動係数が大きいほど、時間間隔が不規則
    time_cv = (
        np.std(time_diff_array) / np.mean(time_diff_array)
        if np.mean(time_diff_array) > 0
        else 0
    )

    # 値の変動性を計算
    values_array = np.array(values)
    values_diff = np.diff(values_array)

    # すべての値が同じかどうか確認（最適化のため）
    all_same_values = np.all(values_array == values_array[0])
    if all_same_values:
        # すべての値が同じ場合、linearが最も効率的
        return "linear"

    # 値の変化の急峻さを測定（変化率の標準偏差）
    if len(values_diff) > 1:
        value_volatility = np.std(values_diff)
    else:
        value_volatility = 0

    # データの滑らかさを評価（隣接点間の2次差分の平均絶対値）
    if len(values) >= 3:
        second_diff = np.diff(values_diff)
        smoothness = np.mean(np.abs(second_diff)) if len(second_diff) > 0 else 0
    else:
        smoothness = 0

    # 外れ値の検出
    # 四分位範囲（IQR）を用いた外れ値検出
    q1, q3 = np.percentile(values_array, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in values if x < lower_bound or x > upper_bound]
    has_outliers = len(outliers) > 0

    # 判断ロジック：時間間隔の規則性に基づく選択
    if time_cv > 0.5:  # 時間間隔が非常に不規則
        # 時間考慮補間が最適
        return "time"

    # 判断ロジック：データの滑らかさと変動性に基づく選択
    if smoothness < 0.1 and value_volatility < 0.2:  # データが非常に滑らか
        if has_outliers:
            # 外れ値があるが滑らかなトレンドがある場合、cubic が適切
            return "cubic"
        else:
            # 外れ値がなく非常に滑らかな場合、スプラインが適切
            return "spline" if len(values) > 10 else "cubic"
    elif value_volatility > 1.0 or has_outliers:  # 変動が大きいまたは外れ値がある
        # 変動が大きい場合、線形補間が安定
        return "linear"
    elif 0.1 <= smoothness < 0.5:  # 中程度の滑らかさ
        # 中程度の滑らかさには2次スプラインが適切
        return "quadratic"
    else:
        # その他のケースでは線形補間が最も安全で予測可能
        return "linear"
