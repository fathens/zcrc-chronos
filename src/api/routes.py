"""
APIルーティングを定義するモジュール
"""

import datetime
import json
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
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


# モデル設定の読み込み
def load_model_config():
    try:
        with open(MODEL_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"モデル設定ファイルの読み込みに失敗しました: {e}")
        raise HTTPException(
            status_code=500, detail="モデル設定の読み込みに失敗しました"
        )


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
                cancelled = future.cancel()

                # どちらの場合でもステータスをCANCELLEDに更新
                self.update_task(
                    task_id,
                    status=PredictionStatus.CANCELLED,
                    message="タスクがキャンセルされました",
                )
                return cancelled
            elif task_id in self._tasks:
                # futureが登録されていない場合でも、タスクが存在すればキャンセル可能
                self.update_task(
                    task_id,
                    status=PredictionStatus.CANCELLED,
                    message="タスクがキャンセルされました",
                )
                return True
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


# ファイルベースキューシステム
class FileBasedQueue:
    """ファイルベースの永続化キューシステム"""

    def __init__(self, queue_dir: str):
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"

        # ディレクトリ作成
        for dir_path in [self.pending_dir, self.processing_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()

    def enqueue(self, task_id: str, request_data: dict) -> bool:
        """タスクをキューに追加"""
        try:
            with self.lock:
                task_file = self.pending_dir / f"{task_id}.json"
                task_data = {
                    "task_id": task_id,
                    "request": request_data,
                    "enqueued_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "priority": 0,  # 将来の拡張用
                }

                with open(task_file, "w") as f:
                    json.dump(task_data, f, indent=2)

                logger.info(
                    f"タスク {task_id} をファイルキューに追加しました: {task_file.absolute()}"
                )
                return True
        except Exception as e:
            logger.error(f"タスクキューへの追加に失敗: {e}")
            return False

    def dequeue(self) -> Optional[Tuple[str, dict]]:
        """キューから最古のタスクを取得"""
        try:
            with self.lock:
                # pendingディレクトリから最古のファイルを取得
                pending_files = list(self.pending_dir.glob("*.json"))
                logger.debug(
                    f"pendingディレクトリ {self.pending_dir.absolute()} には "
                    f"{len(pending_files)} 個のファイルがあります"
                )
                if not pending_files:
                    return None

                # ファイル作成時刻でソート
                oldest_file = min(pending_files, key=lambda f: f.stat().st_ctime)

                # ファイルを読み込み
                with open(oldest_file, "r") as f:
                    task_data = json.load(f)

                task_id = task_data["task_id"]
                request_data = task_data["request"]

                # ファイルをprocessingディレクトリに移動
                processing_file = self.processing_dir / oldest_file.name
                oldest_file.rename(processing_file)

                logger.info(f"タスク {task_id} をキューから取得しました")
                return task_id, request_data

        except Exception as e:
            logger.error(f"キューからの取得に失敗: {e}")
            return None

    def mark_completed(self, task_id: str, success: bool = True):
        """タスクを完了としてprocessingファイルを削除"""
        try:
            with self.lock:
                processing_file = self.processing_dir / f"{task_id}.json"
                if processing_file.exists():
                    processing_file.unlink()
                    logger.info(f"タスク {task_id} のprocessingファイルを削除しました")
        except Exception as e:
            logger.error(f"タスク完了処理に失敗: {e}")

    def get_queue_size(self) -> int:
        """待機中のタスク数を取得"""
        try:
            return len(list(self.pending_dir.glob("*.json")))
        except Exception:
            return 0

    def get_processing_count(self) -> int:
        """実行中のタスク数を取得"""
        try:
            return len(list(self.processing_dir.glob("*.json")))
        except Exception:
            return 0

    def reset_processing_to_pending(self):
        """processingディレクトリの全ファイルをpendingに戻す（起動時リカバリ用）"""
        try:
            with self.lock:
                processing_files = list(self.processing_dir.glob("*.json"))
                for processing_file in processing_files:
                    pending_file = self.pending_dir / processing_file.name
                    processing_file.rename(pending_file)
                    logger.info(
                        f"ファイル {processing_file.name} をpendingに戻しました"
                    )
                logger.info(
                    f"{len(processing_files)}個のファイルをprocessingからpendingに戻しました"
                )
        except Exception as e:
            logger.error(f"processingファイルのリカバリに失敗: {e}")


# キューイングシステム
class PredictionQueue:
    """予測タスクのキューイングシステム（ファイルベース対応）"""

    def __init__(
        self,
        max_concurrent_tasks: int = 2,
        queue_storage: str = "memory",
        queue_size: int = 10,
        queue_dir: str = None,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.queue_storage = queue_storage
        self.queue_size = queue_size
        self.running_tasks = {}  # task_id -> Future
        self.lock = threading.RLock()

        if queue_storage == "file" and queue_dir:
            logger.info(f"ファイルベースキューを初期化中: queue_dir={queue_dir}")
            self.file_queue = FileBasedQueue(queue_dir)
            logger.info(f"ファイルキューディレクトリ: {self.file_queue.queue_dir}")
            # 起動時リカバリ：processingファイルをpendingに戻す
            self.file_queue.reset_processing_to_pending()
            self.memory_queue = None
        else:
            self.memory_queue = queue.Queue(maxsize=queue_size)
            self.file_queue = None

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info(f"ワーカースレッドを開始しました: {self.worker_thread.name}")

    def submit_task(self, task_id: str, request) -> bool:
        """タスクをキューに追加"""
        try:
            if self.file_queue:
                # ファイルベースキュー（無制限）
                # requestオブジェクトをJSON化可能な辞書に変換
                request_dict = {}
                for key, value in request.__dict__.items():
                    if isinstance(value, datetime.datetime):
                        request_dict[key] = value.isoformat()
                    elif (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], datetime.datetime)
                    ):
                        request_dict[key] = [v.isoformat() for v in value]
                    else:
                        request_dict[key] = value
                result = self.file_queue.enqueue(task_id, request_dict)
                logger.info(f"ファイルキューへのタスク登録結果: {result}")
                return result
            else:
                # メモリベースキュー（サイズ制限あり）
                self.memory_queue.put((task_id, request), block=False)
                logger.info(f"タスク {task_id} をメモリキューに追加しました")
                return True
        except queue.Full:
            logger.warning("タスクキューが満杯です")
            return False
        except Exception as e:
            logger.error(f"タスクキューへの追加に失敗: {e}")
            return False

    def _worker(self):
        """ワーカースレッド - キューからタスクを取り出して実行"""
        logger.info("ワーカースレッドが開始されました")
        while True:
            try:
                # 実行中タスク数をまずチェック
                with self.lock:
                    if len(self.running_tasks) >= self.max_concurrent_tasks:
                        time.sleep(1.0)
                        continue

                # キューからタスクを取得（並列度に余裕がある場合のみ）
                if self.file_queue:
                    # ファイルベースキュー
                    logger.debug("ファイルキューからタスクを取得中...")
                    task_data = self.file_queue.dequeue()
                    if task_data is None:
                        logger.debug("ファイルキューにタスクがありません")
                        time.sleep(1.0)
                        continue
                    task_id, request_dict = task_data
                    # 辞書からRequestオブジェクトを復元
                    from types import SimpleNamespace

                    # datetime文字列を復元
                    for key, value in request_dict.items():
                        if (
                            isinstance(value, str)
                            and key in ["forecast_until"]
                            and "T" in value
                        ):
                            try:
                                request_dict[key] = datetime.datetime.fromisoformat(
                                    value.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass  # 復元に失敗した場合は文字列のまま
                        elif (
                            isinstance(value, list)
                            and value
                            and isinstance(value[0], str)
                            and "T" in value[0]
                        ):
                            try:
                                request_dict[key] = [
                                    datetime.datetime.fromisoformat(
                                        v.replace("Z", "+00:00")
                                    )
                                    for v in value
                                ]
                            except (ValueError, TypeError):
                                pass  # 復元に失敗した場合はリストのまま
                    request = SimpleNamespace(**request_dict)
                else:
                    # メモリベースキュー
                    task_id, request = self.memory_queue.get(timeout=1.0)

                # タスクを実行
                logger.info(f"タスク {task_id} の実行を開始します")

                # タスクステータスを実行中に更新
                task_manager.update_task(
                    task_id,
                    status=PredictionStatus.RUNNING,
                    message="予測処理を実行中です",
                )

                # ThreadPoolExecutorでタスクを実行
                future = executor.submit(run_prediction_task, task_id, request)

                with self.lock:
                    self.running_tasks[task_id] = future

                # タスク完了を監視
                def on_task_complete(captured_task_id):
                    with self.lock:
                        if captured_task_id in self.running_tasks:
                            del self.running_tasks[captured_task_id]
                    # ファイルベースキューの場合は完了マーク
                    if self.file_queue:
                        self.file_queue.mark_completed(captured_task_id, True)
                    # TaskManagerの状態も更新
                    # FAILEDステータスは上書きしない
                    try:
                        current_task = task_manager.get_task(captured_task_id)
                        if (
                            current_task
                            and current_task.status != PredictionStatus.FAILED
                        ):
                            task_manager.update_task(
                                captured_task_id, status=PredictionStatus.COMPLETED
                            )
                            logger.info(
                                f"タスク {captured_task_id} のTaskManager状態を更新しました"
                            )
                        elif (
                            current_task
                            and current_task.status == PredictionStatus.FAILED
                        ):
                            logger.info(
                                f"タスク {captured_task_id} はFAILED状態のため、COMPLETEDに更新しません"
                            )
                    except Exception as e:
                        logger.error(
                            f"タスク {captured_task_id} のTaskManager状態更新に失敗: {e}"
                        )
                    logger.info(f"タスク {captured_task_id} が完了しました")

                future.add_done_callback(lambda f, tid=task_id: on_task_complete(tid))

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ワーカースレッドでエラーが発生しました: {e}")


# グローバルタスクマネージャーのインスタンス
task_manager = TaskManager()

# 設定を読み込んで並列処理を設定
try:
    config = load_config()
    prediction_config = config.get("prediction", {})
    data_config = config.get("data", {})

    max_concurrent_tasks = prediction_config.get("max_concurrent_tasks", 2)
    queue_storage = prediction_config.get("queue_storage", "memory")
    queue_size = prediction_config.get("queue_size", 10)
    queue_dir = data_config.get("queue_dir", "data/queue")

    # queue_size = -1 の場合は無制限とみなし、ファイルベースを強制
    if queue_size == -1:
        queue_storage = "file"
        queue_size = 0  # ファイルベースでは使用しない

except Exception as e:
    logger.warning(f"設定ファイルの読み込みに失敗、デフォルト値を使用: {e}")
    max_concurrent_tasks = 2
    queue_storage = "memory"
    queue_size = 10
    queue_dir = "data/queue"

# キューイングシステムのインスタンス
prediction_queue = PredictionQueue(
    max_concurrent_tasks=max_concurrent_tasks,
    queue_storage=queue_storage,
    queue_size=queue_size,
    queue_dir=queue_dir,
)

logger.info(
    f"キューイングシステムを初期化しました: storage={queue_storage}, "
    f"max_concurrent={max_concurrent_tasks}, queue_size={queue_size}"
)

# ThreadPoolExecutor for running prediction tasks
executor = ThreadPoolExecutor(
    max_workers=max_concurrent_tasks * 2
)  # キューイング用に少し多めに設定


def validate_model_name(model_name: str) -> bool:
    """モデル名が有効かどうかを検証する"""
    config = load_model_config()

    # available_models配列から検索
    if "available_models" in config:
        for model in config["available_models"]:
            if model.get("name") == model_name:
                return True

    # default_modelとの完全一致チェック
    if "default_model" in config and config["default_model"].get("name") == model_name:
        return True

    return False


def get_available_model_names() -> List[str]:
    """利用可能なモデル名のリストを取得する"""
    config = load_model_config()
    available_models = []

    if "available_models" in config:
        available_models = [model.get("name") for model in config["available_models"]]

    if "default_model" in config:
        available_models.append(config["default_model"].get("name"))

    return available_models


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
        model_config = load_model_config()
        models = []

        # デフォルトモデルを追加
        if "default_model" in model_config:
            default_model = model_config["default_model"]
            models.append(
                ModelInfo(
                    name=default_model["name"],
                    version=default_model["version"],
                    description=default_model["description"],
                    parameters=default_model["chronos"],
                )
            )

        # 利用可能なモデルを追加
        if "available_models" in model_config:
            for model in model_config["available_models"]:
                models.append(
                    ModelInfo(
                        name=model["name"],
                        version=model["version"],
                        description=model["description"],
                        parameters=model["chronos"],
                    )
                )

        return models
    except Exception as e:
        logger.error(f"モデル情報の取得に失敗しました: {e}")
        raise HTTPException(status_code=500, detail="モデル情報の取得に失敗しました")


# 重複した関数定義を削除


def run_prediction_task(task_id: str, request: AsyncPredictionRequest):
    """バックグラウンドで予測を実行する関数"""
    try:
        # タスクステータスを実行中に更新
        task_manager.update_task(
            task_id, status=PredictionStatus.RUNNING, message="予測処理を開始しました"
        )

        logger.info(f"タスク {task_id} の予測処理を開始します")

        # モデル名の検証
        if not validate_model_name(request.model_name):
            available_models = get_available_model_names()
            error_msg = (
                f"モデル '{request.model_name}' が見つかりません。"
                f"利用可能なモデル: {available_models}"
            )
            logger.error(error_msg)
            task_manager.update_task(
                task_id, status=PredictionStatus.FAILED, message=error_msg
            )
            return

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
            model_name=request.model_name,
            model_params=request.model_params,
            enable_hierarchical_training=True,  # カラム名修正により再有効化
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

        # タスクマネージャーにタスクを追加
        task_manager.add_task(task_id, task)

        # キューイングシステムにタスクを送信
        if not prediction_queue.submit_task(task_id, request):
            task_manager.update_task(
                task_id,
                status=PredictionStatus.FAILED,
                message="タスクキューが満杯です。しばらく時間をおいて再試行してください。",
            )
            raise HTTPException(
                status_code=503,
                detail="サーバーが混雑しています。しばらく時間をおいて再試行してください。",
            )

        # タスクをキューに入れた状態に更新
        task_manager.update_task(
            task_id,
            status=PredictionStatus.PENDING,
            message="予測タスクがキューに登録されました",
        )

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
    1. 価格変動パターンの完全保持を最優先
    2. 元のデータポイントを基準とした最小限の補間
    3. スパイクや急激な変動の保持
    4. データポイント数を増やさない設計

    価格データでは急激な変動（スパイク）も重要な情報なので、
    可能な限り元のタイムスタンプ間隔を保持し、必要最小限の正規化のみを行います。

    Args:
        timestamps: 元のタイムスタンプのリスト
        values: 元の値のリスト
        interpolation_method: 補間方法（価格データでは使用しません）

    Returns:
        正規化されたタイムスタンプと値のタプル
    """
    if not timestamps or not values:
        return timestamps, values

    # タイムスタンプの範囲とデータの個数をログに出力
    start_time = min(timestamps)
    end_time = max(timestamps)
    num_points = len(timestamps)

    logger.info(
        f"タイムスタンプの範囲: {start_time.isoformat()} から {end_time.isoformat()}, "
        f"データ数: {num_points}"
    )

    # 価格データの変動保持を最優先する設計
    # 時間間隔の規則性をチェック
    if num_points <= 1:
        return timestamps, values

    time_diffs = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]

    # 時間間隔の変動係数を計算
    time_diff_array = np.array(time_diffs, dtype=np.float64)
    mean_interval = np.mean(time_diff_array)
    std_interval = np.std(time_diff_array)
    cv = std_interval / mean_interval if mean_interval > 0 else 0

    # 時間間隔が十分に規則的（CV < 0.1）な場合は正規化をスキップ
    if cv < 0.1:
        logger.info(f"時間間隔が規則的（CV={cv:.3f}）なので正規化をスキップします")
        return timestamps, values

    # 時間間隔が不規則な場合は最小限の正規化を実行
    # ただし、価格変動を保持するために元のデータポイント数を維持
    logger.info(f"時間間隔が不規則（CV={cv:.3f}）なので最小限の正規化を実行します")

    # 開始時刻と終了時刻の間を元のデータポイント数で等分
    total_duration = (end_time - start_time).total_seconds()

    if total_duration <= 0:
        # 時間範囲が0の場合は元のデータをそのまま返す
        return timestamps, values

    # 均等な間隔を計算
    interval_seconds = total_duration / max(1, num_points - 1)

    # 新しいタイムスタンプを生成（元のデータポイント数と同じ）
    new_timestamps = []
    for i in range(num_points):
        new_time = start_time + datetime.timedelta(seconds=interval_seconds * i)
        new_timestamps.append(new_time)

    # 最後のタイムスタンプを終了時刻に合わせる
    if len(new_timestamps) > 0:
        new_timestamps[-1] = end_time

    # pandasを使用して最小限の補間を実行
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    df = df.set_index("timestamp")

    try:
        # 元のデータポイントが最も重要なので、nearest interpolationを使用
        # これにより急激な変動（スパイク）を保持
        resampled_df = df.reindex(new_timestamps, method="nearest")

        # 結果を返す
        normalized_values = resampled_df["value"].tolist()

        # 変動保持の確認
        original_range = max(values) - min(values)
        normalized_range = max(normalized_values) - min(normalized_values)
        retention_rate = (
            (normalized_range / original_range * 100) if original_range > 0 else 100
        )

        logger.info(
            f"正規化完了: 変動保持率={retention_rate:.1f}% "
            f"(元: {original_range:.2f} → 正規化後: {normalized_range:.2f})"
        )

        return new_timestamps, normalized_values

    except Exception as e:
        # エラーが発生した場合は元のデータをそのまま返す
        logger.error(
            f"正規化処理でエラーが発生しました: {e}. 元のデータをそのまま使用します。"
        )
        return timestamps, values


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
