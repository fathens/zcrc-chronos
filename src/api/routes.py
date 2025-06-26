"""
APIルーティングを定義するモジュール
"""

import datetime
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
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
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise HTTPException(
            status_code=500, detail="サーバー設定の読み込みに失敗しました"
        )


config = load_config()

# APIルーターの作成
router = APIRouter()

# 非同期タスク管理クラス


class TaskManager:
    """スレッドセーフなタスク管理クラス"""

    def __init__(self, max_tasks: int = 1000, cleanup_interval_hours: int = 24):
        self._tasks: Dict[str, "PredictionResult"] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.RLock()
        self._max_tasks = max_tasks
        self._cleanup_interval_hours = cleanup_interval_hours

    def add_task(self, task_id: str, task: "PredictionResult", future: Future = None):
        """タスクを追加"""
        with self._lock:
            # 最大タスク数の制限
            if len(self._tasks) >= self._max_tasks:
                self._cleanup_old_tasks()

            self._tasks[task_id] = task
            if future:
                self._futures[task_id] = future

    def get_task(self, task_id: str) -> Optional["PredictionResult"]:
        """タスクを取得"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> List["PredictionResult"]:
        """すべてのタスクを取得"""
        with self._lock:
            return list(self._tasks.values())

    def update_task(self, task_id: str, **kwargs):
        """タスクを更新"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.updated_at = datetime.datetime.now(datetime.timezone.utc)

    def cancel_task(self, task_id: str) -> bool:
        """タスクをキャンセル"""
        with self._lock:
            if task_id in self._futures:
                future = self._futures[task_id]
                if future.cancel():
                    self.update_task(
                        task_id,
                        status=PredictionStatus.CANCELLED,
                        message="タスクがキャンセルされました",
                    )
                    return True
                else:
                    # 既に実行中の場合は状態のみ更新
                    self.update_task(
                        task_id,
                        status=PredictionStatus.CANCELLED,
                        message="タスクのキャンセルが要求されました",
                    )
                    return False
            return False

    def _cleanup_old_tasks(self):
        """古いタスクをクリーンアップ"""
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_time = now - datetime.timedelta(hours=self._cleanup_interval_hours)

        tasks_to_remove = []
        for task_id, task in self._tasks.items():
            if task.updated_at < cutoff_time:
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self._tasks[task_id]
            if task_id in self._futures:
                del self._futures[task_id]

    def cleanup_completed_tasks(self):
        """完了したタスクをクリーンアップ"""
        with self._lock:
            tasks_to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in [
                    PredictionStatus.COMPLETED,
                    PredictionStatus.FAILED,
                    PredictionStatus.CANCELLED,
                ]:
                    # 完了から1時間経過したタスクを削除
                    if (
                        datetime.datetime.now(datetime.timezone.utc) - task.updated_at
                    ).total_seconds() > 3600:
                        tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                if task_id in self._futures:
                    del self._futures[task_id]


# グローバルタスクマネージャーのインスタンス
task_manager = TaskManager()

# ThreadPoolExecutor for running prediction tasks
executor = ThreadPoolExecutor(max_workers=4)


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
    """予測レスポンスモデル

    時系列予測の結果を格納するレスポンスモデルです。
    予測値、使用したモデル、信頼区間、評価指標などを含みます。

    Attributes:
        forecast_timestamp (List[datetime.datetime]):
            予測された将来の時間点のリスト。
            予測開始時点から予測終了時点までの時間系列を表します。

        forecast_values (List[float]):
            予測値のリスト。
            forecast_timestampの各時点に対応する予測された値です。

        model_name (str):
            予測に使用されたモデルの名前。
            どのモデルで予測が行われたかを識別します。

        confidence_intervals (Optional[Dict[str, List[float]]]):
            予測の信頼区間。辞書形式で提供され、キーは信頼区間の種類（例: "lower_95", "upper_95"）、
            値は予測値ごとの信頼限界値のリストです。
            例: {"lower_95": [10.5, 10.8, 11.0], "upper_95": [11.3, 11.8, 12.4]}

        metrics (Optional[Dict[str, float]]):
            予測性能の評価指標。辞書形式で提供され、キーは指標の名前、値はその数値です。
            一般的な指標:
            - "mse": 平均二乗誤差 - 予測値と実測値の差の二乗の平均
            - "mae": 平均絶対誤差 - 予測値と実測値の絶対差の平均
            - "rmse": 二乗平均平方根誤差 - MSEの平方根
            - "mape": 平均絶対パーセント誤差 - 相対誤差の平均（%）

    Example:
        ```json
        {
            "forecast_timestamp": [
                "2023-01-01T03:00:00",
                "2023-01-01T04:00:00",
                "2023-01-01T05:00:00"
            ],
            "forecast_values": [10.9, 11.3, 11.7],
            "model_name": "chronos_default",
            "confidence_intervals": {
                "lower_95": [10.5, 10.8, 11.0],
                "upper_95": [11.3, 11.8, 12.4]
            },
            "metrics": {"mse": 0.15, "mae": 0.12}
        }
        ```
    """

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


class PredictionStatus(str, Enum):
    """予測ステータス"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AsyncPredictionRequest(BaseModel):
    """非同期ゼロショット予測リクエストモデル"""

    timestamp: List[datetime.datetime]
    values: List[float]
    forecast_until: datetime.datetime
    model_name: Optional[str] = "chronos_default"
    model_params: Optional[Dict[str, Any]] = None

    @field_validator("values")
    def validate_values_length(cls, v, info):
        data = info.data
        if "timestamp" in data and len(data["timestamp"]) != len(v):
            raise ValueError("timestampとvaluesの長さが一致しません")
        return v


class AsyncPredictionResponse(BaseModel):
    """非同期予測開始レスポンス"""

    task_id: str
    status: PredictionStatus
    message: str


class PredictionResult(BaseModel):
    """予測結果モデル"""

    task_id: str
    status: PredictionStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


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
@router.post("/predict_zero_shot", response_model=PredictionResponse)
async def predict_zero_shot(request: ZeroShotPredictionRequest):
    """
    時系列データに基づくゼロショット予測を実行

    過去の時系列データから将来の値を予測するAPIエンドポイントです。
    「ゼロショット」とは、追加の訓練なしで予測を行うことを意味します。

    Parameters:
        request (ZeroShotPredictionRequest): 予測リクエスト
            - **timestamp**: 時系列データのタイムスタンプのリスト
              過去の時系列データの時間情報を示します。日時形式（ISO 8601）で指定。

            - **values**: 時系列データの値のリスト
              timestampに対応する観測値または実測値。数値の配列で指定。

            - **forecast_until**: 予測したい時点（datetime形式）
              この時点までの将来値が予測されます。最後のタイムスタンプより未来の日時を指定。

            - **model_name**: 使用する予測モデルの名前（オプション、デフォルト: "chronos_default"）
              利用可能なモデルは GET /models エンドポイントで確認できます。

            - **model_params**: モデルに渡す追加パラメータ（オプション）
              モデル固有のパラメータを辞書形式で指定できます。

    Returns:
        PredictionResponse: 予測結果
            - **forecast_timestamp**: 予測された将来時点のリスト
            - **forecast_values**: 予測値のリスト
            - **model_name**: 使用されたモデル名
            - **confidence_intervals**: 信頼区間（提供されている場合）
            - **metrics**: 予測性能の評価指標（提供されている場合）

    Raises:
        HTTPException (400): 以下の場合にエラーを返します
            - データポイントが2点未満の場合
            - タイムスタンプの間隔が不正（ゼロまたは負の値）の場合
            - 予測時点が最新のデータポイント以前の場合
        HTTPException (500): 予測処理中に内部エラーが発生した場合

    Example:
        ```bash
        curl -X 'POST' \\
          'http://localhost:8000/api/v1/predict_zero_shot' \\
          -H 'Content-Type: application/json' \\
          -d '{
            "timestamp": [
              "2023-01-01T00:00:00",
              "2023-01-01T01:00:00"
            ],
            "values": [10.5, 11.2],
            "forecast_until": "2023-01-04T02:00:00",
            "model_name": "chronos_default"
        }'
        ```
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
                    f"予測時点が最新のデータポイント以前です。"
                    f"予測時点: {request.forecast_until}, "
                    f"最新のデータポイント: {latest_timestamp}"
                ),
            )

        logger.info(
            f"予測ポイント数: {prediction_points}, 予測時点: {request.forecast_until}"
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
    except HTTPException:
        # HTTPExceptionはそのまま再発生させる
        raise
    except Exception as e:
        logger.error(f"ゼロショット予測処理に失敗しました: {e}")

        # データポイント不足の場合は400エラーとして処理
        error_message = str(e)
        if (
            "データポイントが不十分" in error_message
            or "少なくとも2つのデータポイントが必要です" in error_message
            or "At least some time series in train_data must have >= 5 observations"
            in error_message
        ):
            raise HTTPException(status_code=400, detail=error_message)

        raise HTTPException(
            status_code=500, detail=f"ゼロショット予測処理に失敗しました: {e}"
        )


def run_prediction_task(task_id: str, request: AsyncPredictionRequest):
    """バックグラウンドで予測を実行する関数"""
    try:
        # タスクステータスを実行中に更新
        task_manager.update_task(
            task_id, status=PredictionStatus.RUNNING, message="予測処理を開始しました"
        )

        logger.info(f"タスク {task_id} の予測処理を開始します")

        # キャンセルチェック
        task = task_manager.get_task(task_id)
        if task and task.status == PredictionStatus.CANCELLED:
            logger.info(f"タスク {task_id} はキャンセルされました")
            return

        # 時系列データの正規化
        normalized_timestamps, normalized_values = normalize_time_series_data(
            request.timestamp, request.values, interpolation_method="auto"
        )

        # キャンセルチェック
        task = task_manager.get_task(task_id)
        if task and task.status == PredictionStatus.CANCELLED:
            logger.info(f"タスク {task_id} はキャンセルされました")
            return

        # 最後のタイムスタンプを取得
        latest_timestamp = max(normalized_timestamps)

        # タイムスタンプの間隔を計算
        if len(normalized_timestamps) >= 2:
            delta = normalized_timestamps[1] - normalized_timestamps[0]
        else:
            raise ValueError("予測には少なくとも2つのデータポイントが必要です")

        # 予測期間の計算
        time_difference = request.forecast_until - latest_timestamp

        if delta.total_seconds() <= 0:
            raise ValueError("タイムスタンプの間隔が正しくありません")

        prediction_points = int(time_difference.total_seconds() / delta.total_seconds())

        if prediction_points <= 0:
            raise ValueError("予測時点が最新のデータポイント以前です")

        logger.info(f"タスク {task_id}: 予測ポイント数: {prediction_points}")

        # 進捗更新: 前処理完了
        task_manager.update_task(
            task_id, progress=0.2, message="データ前処理が完了しました"
        )

        # キャンセルチェック
        task = task_manager.get_task(task_id)
        if task and task.status == PredictionStatus.CANCELLED:
            logger.info(f"タスク {task_id} はキャンセルされました")
            return

        # 予測モデルの初期化
        predictor = TimeSeriesPredictor(
            model_name=request.model_name, model_params=request.model_params
        )

        # 進捗更新: モデル初期化完了
        task_manager.update_task(
            task_id, progress=0.3, message="モデル初期化が完了しました"
        )

        # キャンセルチェック
        task = task_manager.get_task(task_id)
        if task and task.status == PredictionStatus.CANCELLED:
            logger.info(f"タスク {task_id} はキャンセルされました")
            return

        # ゼロショット予測の実行
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            timestamp=normalized_timestamps,
            values=normalized_values,
            horizon=prediction_points,
        )

        # キャンセルチェック
        task = task_manager.get_task(task_id)
        if task and task.status == PredictionStatus.CANCELLED:
            logger.info(f"タスク {task_id} はキャンセルされました")
            return

        # 進捗更新: 予測完了
        task_manager.update_task(
            task_id, progress=0.9, message="予測計算が完了しました"
        )

        # レスポンスを作成
        result = PredictionResponse(
            forecast_timestamp=forecast_timestamps,
            forecast_values=forecast_values,
            model_name=request.model_name,
            confidence_intervals=metadata.get("confidence_intervals"),
            metrics=metadata.get("metrics"),
        )

        # タスク完了
        task_manager.update_task(
            task_id,
            status=PredictionStatus.COMPLETED,
            progress=1.0,
            result=result,
            message="予測処理が正常に完了しました",
        )

        logger.info(f"タスク {task_id} の予測処理が正常に完了しました")

    except Exception as e:
        # エラー処理
        error_msg = f"予測処理中にエラーが発生しました: {str(e)}"
        logger.error(f"タスク {task_id}: {error_msg}")

        task_manager.update_task(
            task_id,
            status=PredictionStatus.FAILED,
            error=error_msg,
            message="予測処理に失敗しました",
        )


# 非同期ゼロショット予測開始エンドポイント
@router.post("/predict_zero_shot_async", response_model=AsyncPredictionResponse)
async def predict_zero_shot_async(request: AsyncPredictionRequest):
    """
    非同期でゼロショット予測を開始する

    長時間実行される予測処理を非同期で開始し、task_idを返します。
    予測の進捗や結果は別のエンドポイントでポーリングして取得できます。
    """
    try:
        # 一意のタスクIDを生成
        task_id = str(uuid.uuid4())

        # タスクを初期化
        now = datetime.datetime.now(datetime.timezone.utc)
        task = PredictionResult(
            task_id=task_id,
            status=PredictionStatus.PENDING,
            progress=0.0,
            message="予測タスクが開始されました",
            created_at=now,
            updated_at=now,
        )

        # Futureオブジェクトを作成してタスクを送信
        future = executor.submit(run_prediction_task, task_id, request)

        # タスクマネージャーにタスクとFutureを追加
        task_manager.add_task(task_id, task, future)

        logger.info(f"非同期予測タスクを開始しました: {task_id}")
        return AsyncPredictionResponse(
            task_id=task_id,
            status=PredictionStatus.PENDING,
            message="予測タスクが正常に開始されました",
        )

    except Exception as e:
        logger.error(f"非同期予測タスクの開始に失敗しました: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"非同期予測タスクの開始に失敗しました: {str(e)}"
        )


# 予測タスクのステータス確認エンドポイント
@router.get("/prediction_status/{task_id}", response_model=PredictionResult)
async def get_prediction_status(task_id: str):
    """
    予測タスクのステータスと結果を取得する

    task_idに基づいて予測の進捗状況、結果、またはエラー情報を返します。
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404, detail=f"タスクが見つかりません: {task_id}"
        )
    return task


# 予測タスクの一覧取得エンドポイント
@router.get("/prediction_tasks", response_model=List[PredictionResult])
async def get_prediction_tasks():
    """
    すべての予測タスクの一覧を取得する
    """
    # 完了したタスクのクリーンアップを実行
    task_manager.cleanup_completed_tasks()
    return task_manager.get_all_tasks()


# 予測タスクのキャンセルエンドポイント
@router.delete("/prediction_cancel/{task_id}")
async def cancel_prediction_task(task_id: str):
    """
    実行中の予測タスクをキャンセルする

    task_idで指定されたタスクの実行をキャンセルします。
    既に実行中のタスクはすぐには停止されませんが、次のチェックポイントでキャンセルされます。
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404, detail=f"タスクが見つかりません: {task_id}"
        )

    if task.status in [
        PredictionStatus.COMPLETED,
        PredictionStatus.FAILED,
        PredictionStatus.CANCELLED,
    ]:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=400,
            content={
                "error": f"タスクは既に{task.status.value}状態です",
                "task_id": task_id,
            },
        )

    cancelled = task_manager.cancel_task(task_id)

    if cancelled:
        message = "タスクが正常にキャンセルされました"
    else:
        message = "タスクのキャンセルが要求されました（実行中のタスクは次のチェックポイントで停止されます）"

    logger.info(f"タスク {task_id} のキャンセルが要求されました")

    return {"task_id": task_id, "message": message, "cancelled": cancelled}


def normalize_time_series_data(
    timestamps: List[datetime.datetime],
    values: List[float],
    interpolation_method: str = "auto",
) -> Tuple[List[datetime.datetime], List[float]]:
    """
    時系列データを均等な間隔に正規化する関数

    本関数は価格データの直線的予測問題を解決するために最適化されています。
    主要な改善点：
    1. 外れ値検出基準の緩和（1.5×IQR → 3.0×IQR）
    2. 価格データに特化した補間方法選択
    3. データポイント倍増の無効化
    4. 変動保持を最優先とする設計

    最初と最後のタイムスタンプ間の時間を、元のデータポイント数を保持しつつ
    均等に分割し、価格変動パターンを保持する補間方法で新しい値を計算します。

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
            f"無効な補間方法: {interpolation_method}. 'auto'に切り替えます。"
        )
        interpolation_method = "auto"

    # 自動補間方法選択
    if interpolation_method == "auto":
        interpolation_method = _determine_best_interpolation_method(timestamps, values)
        logger.info(f"自動選択された補間方法: {interpolation_method}")

    # pandasのDataFrameを作成
    df = pd.DataFrame({"timestamp": timestamps, "value": values})

    # タイムスタンプをインデックスに設定
    df = df.set_index("timestamp")

    # 開始時刻と終了時刻を取得
    start_time = min(timestamps)
    end_time = max(timestamps)

    # 全体の時間範囲を計算（秒単位）
    total_duration = (end_time - start_time).total_seconds()

    # データポイントの数に基づいて均等な間隔を計算
    # 少なくとも元のデータと同じポイント数を維持
    num_points = len(timestamps)

    # タイムスタンプの範囲とデータの個数をログに出力
    logger.info(
        f"タイムスタンプの範囲: {start_time.isoformat()} から {end_time.isoformat()}, "
        f"データ数: {num_points}"
    )

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
        # 時間範囲が0または非常に小さい場合の対処
        if total_duration <= 0:
            # 同一時刻または時間範囲が0の場合、元のタイムスタンプをそのまま使用
            new_timestamps = timestamps.copy()
        else:
            # 間隔を調整して、元のデータポイント数を維持（2倍にしない）
            # 価格データでは元のデータポイント数を保持することが重要
            adjusted_interval = total_duration / max(1, num_points - 1)
            new_timestamps = []
            current_time = start_time

            # 無限ループ防止とデータポイント数制限
            max_points = num_points * 3  # 最大でも3倍まで
            point_count = 0

            while current_time <= end_time and point_count < max_points:
                new_timestamps.append(current_time)
                current_time += datetime.timedelta(seconds=adjusted_interval)
                point_count += 1
                # 安全装置：間隔が非常に小さい場合の停止
                if adjusted_interval < 0.001:  # 1ミリ秒未満
                    break

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
            f"補間方法 '{interpolation_method}' でエラーが発生しました: {e}. "
            f"線形補間を使用します。"
        )
        resampled_df = df.reindex(new_timestamps).interpolate(method="linear")
        return new_timestamps, resampled_df["value"].tolist()


def _determine_best_interpolation_method(
    timestamps: List[datetime.datetime],
    values: List[float],
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
    # 明示的に浮動小数点型を指定してNumPy配列を作成
    time_diff_array = np.array(time_diffs, dtype=np.float64)

    # 時間間隔の変動係数（標準偏差/平均）を計算
    # 変動係数が大きいほど、時間間隔が不規則
    time_cv = (
        np.std(time_diff_array) / np.mean(time_diff_array)
        if np.mean(time_diff_array) > 0
        else 0
    )

    # 値の変動性を計算
    # 明示的に浮動小数点型を指定してNumPy配列を作成
    values_array = np.array(values, dtype=np.float64)
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

    # 外れ値の検出（価格データに適した緩い基準）
    # 価格データでは急激な変動も正常な動きなので、より緩い基準を使用
    q1, q3 = np.quantile(values_array, [0.25, 0.75])
    iqr = q3 - q1
    # 従来の1.5倍から3.0倍に緩和（価格変動を外れ値として扱わない）
    lower_bound = q1 - 3.0 * iqr
    upper_bound = q3 + 3.0 * iqr
    outliers = [x for x in values if x < lower_bound or x > upper_bound]
    has_outliers = len(outliers) > 0

    # 判断ロジック：時間間隔の規則性に基づく選択
    if time_cv > 0.5:  # 時間間隔が非常に不規則
        # 時間考慮補間が最適
        return "time"

    # 判断ロジック：価格データに特化した補間方法選択
    # 価格データでは変動の保持が重要なので、平滑化を避ける

    # 非常に極端な外れ値がある場合のみ線形補間を使用
    if has_outliers and len(outliers) > len(values) * 0.2:  # 20%以上が外れ値
        return "linear"
    # 時系列データの特性に基づく選択
    if value_volatility > 2.0:  # 非常に高い変動性
        # 高い変動性を保持するため線形補間
        return "linear"
    elif len(values) > 20 and smoothness < 0.05:  # 十分なデータ点があり非常に滑らか
        # データ点が多く滑らかな場合のみcubicを使用
        return "cubic"
    elif len(values) >= 10 and 0.05 <= smoothness < 0.3:  # 中程度の滑らかさ
        # 中程度の滑らかさには2次補間
        return "quadratic"
    else:
        # デフォルトは線形補間（変動を最も保持）
        return "linear"
