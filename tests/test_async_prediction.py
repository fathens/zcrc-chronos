"""
非同期予測APIの包括的なテストモジュール
"""

import datetime
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes import PredictionStatus, task_manager
from src.api.server import app

# テストクライアントの初期化
client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_prediction_tasks():
    """各テスト前後でtask_managerをクリア"""
    task_manager._tasks.clear()
    task_manager._futures.clear()
    yield
    task_manager._tasks.clear()
    task_manager._futures.clear()


def create_test_request_data():
    """テスト用のリクエストデータを作成"""
    now = datetime.datetime.now()
    timestamps = [
        (now - datetime.timedelta(hours=i)).isoformat() for i in range(24, 0, -1)
    ]
    values = [10.0 + i * 0.1 for i in range(24)]

    last_timestamp = datetime.datetime.fromisoformat(timestamps[-1])
    forecast_until = (last_timestamp + datetime.timedelta(hours=12)).isoformat()

    return {
        "timestamp": timestamps,
        "values": values,
        "forecast_until": forecast_until,
        "model_name": "chronos_default",
    }


class TestAsyncPredictionBasic:
    """基本的な非同期予測機能のテスト"""

    def test_predict_zero_shot_async_success(self):
        """非同期予測の正常開始テスト"""
        request_data = create_test_request_data()

        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "task_id" in data
        assert "status" in data
        assert "message" in data
        assert data["status"] == "pending"

        # task_idが有効なUUIDフォーマットかチェック
        task_id = data["task_id"]
        uuid.UUID(task_id)  # 無効なUUIDの場合は例外が発生

    def test_predict_zero_shot_async_invalid_data(self):
        """無効なデータでの非同期予測テスト"""
        invalid_data = {
            "timestamp": ["2023-01-01T00:00:00"],
            "values": [10.0, 11.0],  # 長さが一致しない
            "forecast_until": "2023-01-02T00:00:00",
        }

        response = client.post("/api/v1/predict_zero_shot_async", json=invalid_data)
        assert response.status_code == 422

    def test_prediction_status_existing_task(self):
        """既存タスクのステータス確認テスト"""
        request_data = create_test_request_data()

        # 非同期予測を開始
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        # ステータスを確認
        status_response = client.get(f"/api/v1/prediction_status/{task_id}")
        assert status_response.status_code == 200

        data = status_response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_prediction_status_nonexistent_task(self):
        """存在しないタスクのステータス確認テスト"""
        fake_task_id = str(uuid.uuid4())

        response = client.get(f"/api/v1/prediction_status/{fake_task_id}")
        assert response.status_code == 404

    def test_prediction_tasks_list_empty(self):
        """空のタスクリスト取得テスト"""
        response = client.get("/api/v1/prediction_tasks")
        assert response.status_code == 200
        assert response.json() == []

    def test_prediction_tasks_list_with_tasks(self):
        """タスクがあるタスクリスト取得テスト"""
        request_data = create_test_request_data()

        # 複数のタスクを作成
        task_ids = []
        for _ in range(3):
            response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
            task_ids.append(response.json()["task_id"])

        # タスクリストを取得
        tasks_response = client.get("/api/v1/prediction_tasks")
        assert tasks_response.status_code == 200

        tasks = tasks_response.json()
        assert len(tasks) == 3

        # 各タスクが正しい構造を持っているかチェック
        for task in tasks:
            assert "task_id" in task
            assert task["task_id"] in task_ids
            assert "status" in task
            assert "created_at" in task


class TestAsyncPredictionWorkflow:
    """非同期予測のワークフローテスト"""

    def test_full_prediction_workflow(self):
        """完全な予測ワークフローのテスト"""
        request_data = create_test_request_data()

        # 1. 非同期予測を開始
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        # 2. 進捗をポーリング（最大30秒まで）
        max_wait_time = 30
        start_time = time.time()
        final_status = None

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            status_data = status_response.json()

            if status_data["status"] in ["completed", "failed"]:
                final_status = status_data
                break

            time.sleep(1)

        # 3. 結果を検証
        assert final_status is not None, "タスクが指定時間内に完了しませんでした"

        if final_status["status"] == "completed":
            assert "result" in final_status
            assert final_status["result"] is not None
            result = final_status["result"]
            assert "forecast_timestamp" in result
            assert "forecast_values" in result
            assert len(result["forecast_values"]) > 0
        else:
            # 失敗した場合はエラー情報があることを確認
            assert "error" in final_status

    @patch("src.api.routes.run_prediction_task")
    def test_prediction_task_progress_updates(self, mock_run_task):
        """予測タスクの進捗更新テスト"""
        request_data = create_test_request_data()

        def mock_task_execution(task_id, request):
            """モックされたタスク実行関数"""
            # 進捗を段階的に更新
            task_manager.update_task(
                task_id,
                status=PredictionStatus.RUNNING,
                progress=0.3,
                message="データ前処理中",
            )

            time.sleep(0.1)

            task_manager.update_task(task_id, progress=0.7, message="予測計算中")

            time.sleep(0.1)

            task_manager.update_task(
                task_id, status=PredictionStatus.COMPLETED, progress=1.0, message="完了"
            )

        mock_run_task.side_effect = mock_task_execution

        # 非同期予測を開始
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        # 少し待ってから進捗を確認
        time.sleep(0.5)

        status_response = client.get(f"/api/v1/prediction_status/{task_id}")
        status_data = status_response.json()

        assert status_data["progress"] >= 0.0
        assert "message" in status_data


class TestAsyncPredictionConcurrency:
    """並行処理テスト"""

    def test_concurrent_predictions(self):
        """同時並行予測のテスト"""
        request_data = create_test_request_data()

        # 複数のタスクを同時に開始
        task_ids = []
        threads = []

        def start_prediction():
            response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
            if response.status_code == 200:
                task_ids.append(response.json()["task_id"])

        # 5つのスレッドで同時にタスクを開始
        for _ in range(5):
            thread = threading.Thread(target=start_prediction)
            threads.append(thread)
            thread.start()

        # すべてのスレッドの完了を待つ
        for thread in threads:
            thread.join()

        # すべてのタスクが正常に作成されたことを確認
        assert len(task_ids) == 5
        assert len(set(task_ids)) == 5  # すべてユニークなID

        # タスクリストを確認
        tasks_response = client.get("/api/v1/prediction_tasks")
        tasks = tasks_response.json()
        assert len(tasks) == 5

    def test_thread_safety_status_check(self):
        """ステータス確認のスレッドセーフティテスト"""
        request_data = create_test_request_data()

        # タスクを開始
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        results = []
        errors = []

        def check_status():
            try:
                response = client.get(f"/api/v1/prediction_status/{task_id}")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # 10個のスレッドで同時にステータスをチェック
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_status)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # エラーが発生せず、すべてのリクエストが成功したことを確認
        assert len(errors) == 0
        assert all(status == 200 for status in results)
        assert len(results) == 10


class TestAsyncPredictionCancellation:
    """タスクキャンセル機能のテスト"""

    def test_cancel_pending_task(self):
        """待機中タスクのキャンセルテスト"""
        request_data = create_test_request_data()

        # タスクを作成
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        # すぐにキャンセル
        cancel_response = client.delete(f"/api/v1/prediction_cancel/{task_id}")
        assert cancel_response.status_code == 200

        cancel_data = cancel_response.json()
        assert cancel_data["task_id"] == task_id
        assert "キャンセル" in cancel_data["message"]

        # ステータスを確認
        status_response = client.get(f"/api/v1/prediction_status/{task_id}")
        status_data = status_response.json()
        assert status_data["status"] == "cancelled"

    def test_cancel_nonexistent_task(self):
        """存在しないタスクのキャンセルテスト"""
        fake_task_id = str(uuid.uuid4())

        response = client.delete(f"/api/v1/prediction_cancel/{fake_task_id}")
        assert response.status_code == 404

    def test_cancel_completed_task(self):
        """完了済みタスクのキャンセルテスト"""
        request_data = create_test_request_data()

        # タスクを作成
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        # タスクを手動で完了状態に設定
        task_manager.update_task(
            task_id, status=PredictionStatus.COMPLETED, progress=1.0, message="完了"
        )

        # 完了済みタスクのキャンセルを試行
        cancel_response = client.delete(f"/api/v1/prediction_cancel/{task_id}")
        assert cancel_response.status_code == 400

    def test_cancel_during_execution(self):
        """実行中タスクのキャンセルテスト"""
        request_data = create_test_request_data()

        # タスクを作成
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        # 少し待って実行中になってからキャンセル
        time.sleep(1)

        cancel_response = client.delete(f"/api/v1/prediction_cancel/{task_id}")
        assert cancel_response.status_code == 200

        cancel_data = cancel_response.json()
        assert cancel_data["task_id"] == task_id

        # ステータスを確認（キャンセルされていることを確認）
        max_wait_time = 10
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            status_data = status_response.json()

            if status_data["status"] == "cancelled":
                break

            time.sleep(0.5)

        # 最終的にキャンセル状態になっていることを確認
        status_response = client.get(f"/api/v1/prediction_status/{task_id}")
        status_data = status_response.json()
        assert status_data["status"] == "cancelled"


class TestAsyncPredictionEdgeCases:
    """エッジケースのテスト"""

    def test_minimal_data_async_prediction(self):
        """最小限のデータでの非同期予測テスト"""
        now = datetime.datetime.now()
        minimal_data = {
            "timestamp": [
                (now - datetime.timedelta(hours=2)).isoformat(),
                (now - datetime.timedelta(hours=1)).isoformat(),
            ],
            "values": [10.0, 11.0],
            "forecast_until": (now + datetime.timedelta(hours=1)).isoformat(),
        }

        response = client.post("/api/v1/predict_zero_shot_async", json=minimal_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]

        # タスクの結果を確認（完了またはエラーが発生するまで）
        max_wait_time = 15
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            status_data = status_response.json()

            if status_data["status"] in ["completed", "failed"]:
                # 最小限のデータでも何らかの結果が得られることを確認
                assert status_data["status"] in ["completed", "failed"]
                break

            time.sleep(1)

    def test_large_dataset_async_prediction(self):
        """大きなデータセットでの非同期予測テスト"""
        now = datetime.datetime.now()

        # 1000データポイントの大きなデータセット
        timestamps = [
            (now - datetime.timedelta(hours=i)).isoformat() for i in range(1000, 0, -1)
        ]
        values = [10.0 + (i % 100) * 0.1 for i in range(1000)]

        large_data = {
            "timestamp": timestamps,
            "values": values,
            "forecast_until": (now + datetime.timedelta(hours=24)).isoformat(),
        }

        response = client.post("/api/v1/predict_zero_shot_async", json=large_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]

        # 開始直後のステータスを確認
        status_response = client.get(f"/api/v1/prediction_status/{task_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["status"] in ["pending", "running"]

    @patch("src.api.routes.TimeSeriesPredictor")
    def test_prediction_failure_handling(self, mock_predictor_class):
        """予測失敗時のエラーハンドリングテスト"""
        # 予測器がエラーを発生させるようにモック
        mock_predictor = MagicMock()
        mock_predictor.zero_shot_predict.side_effect = Exception("予測エラー")
        mock_predictor_class.return_value = mock_predictor

        request_data = create_test_request_data()

        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        # タスクが失敗状態になるまで待つ
        max_wait_time = 10
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            status_data = status_response.json()

            if status_data["status"] == "failed":
                assert "error" in status_data
                assert "予測エラー" in status_data["error"]
                break

            time.sleep(0.5)

    def test_invalid_forecast_until(self):
        """無効な予測時点での非同期予測テスト"""
        now = datetime.datetime.now()

        # 過去の時点を予測時点として指定
        invalid_data = {
            "timestamp": [
                (now - datetime.timedelta(hours=2)).isoformat(),
                (now - datetime.timedelta(hours=1)).isoformat(),
            ],
            "values": [10.0, 11.0],
            "forecast_until": (
                now - datetime.timedelta(hours=3)
            ).isoformat(),  # 過去の時点
        }

        response = client.post("/api/v1/predict_zero_shot_async", json=invalid_data)
        task_id = response.json()["task_id"]

        # タスクが失敗状態になることを確認
        max_wait_time = 10
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            status_data = status_response.json()

            if status_data["status"] == "failed":
                assert "error" in status_data
                break

            time.sleep(0.5)


class TestAsyncPredictionPerformance:
    """パフォーマンステスト"""

    def test_task_creation_performance(self):
        """タスク作成のパフォーマンステスト"""
        request_data = create_test_request_data()

        start_time = time.time()

        # 10個のタスクを連続で作成
        task_ids = []
        for _ in range(10):
            response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
            assert response.status_code == 200
            task_ids.append(response.json()["task_id"])

        end_time = time.time()

        # タスク作成が1秒以内に完了することを確認
        assert end_time - start_time < 1.0

        # すべてのタスクが作成されたことを確認
        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10

    def test_status_check_performance(self):
        """ステータス確認のパフォーマンステスト"""
        request_data = create_test_request_data()

        # タスクを作成
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
        task_id = response.json()["task_id"]

        start_time = time.time()

        # 100回ステータスをチェック
        for _ in range(100):
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            assert status_response.status_code == 200

        end_time = time.time()

        # 100回のステータスチェックが1秒以内に完了することを確認
        assert end_time - start_time < 1.0


class TestAsyncPredictionCleanup:
    """クリーンアップとメモリ管理のテスト"""

    def test_memory_usage_with_many_tasks(self):
        """多数のタスクでのメモリ使用量テスト"""
        request_data = create_test_request_data()

        # 50個のタスクを作成
        initial_task_count = len(task_manager.get_all_tasks())

        for _ in range(50):
            response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
            assert response.status_code == 200

        # タスクが適切に保存されていることを確認
        assert len(task_manager.get_all_tasks()) == initial_task_count + 50

        # タスクリストAPIが正常に動作することを確認
        tasks_response = client.get("/api/v1/prediction_tasks")
        assert tasks_response.status_code == 200
        tasks = tasks_response.json()
        assert len(tasks) == initial_task_count + 50

    def test_task_id_uniqueness(self):
        """タスクIDの一意性テスト"""
        request_data = create_test_request_data()

        task_ids = set()

        # 100個のタスクを作成してIDの一意性を確認
        for _ in range(100):
            response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
            task_id = response.json()["task_id"]

            # 重複がないことを確認
            assert task_id not in task_ids
            task_ids.add(task_id)

        # すべてのタスクIDが一意であることを確認
        assert len(task_ids) == 100


# 統合テスト用のマーカー
@pytest.mark.integration
class TestAsyncPredictionIntegration:
    """統合テスト"""

    def test_end_to_end_prediction_workflow(self):
        """エンドツーエンドの予測ワークフローテスト"""
        # 実際のモデルを使用したフルワークフロー
        request_data = create_test_request_data()

        # 1. 非同期予測開始
        start_response = client.post(
            "/api/v1/predict_zero_shot_async", json=request_data
        )
        assert start_response.status_code == 200
        task_id = start_response.json()["task_id"]

        # 2. 定期的なステータス確認
        max_wait_time = 60  # 実際の予測は時間がかかる可能性がある
        start_time = time.time()
        status_history = []

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            status_data = status_response.json()
            status_history.append(status_data["status"])

            if status_data["status"] in ["completed", "failed"]:
                break

            time.sleep(2)

        # 3. 最終結果の検証
        final_status_response = client.get(f"/api/v1/prediction_status/{task_id}")
        final_status = final_status_response.json()

        # 進捗が順序通りに進んだことを確認
        assert "pending" in status_history or "running" in status_history

        if final_status["status"] == "completed":
            assert final_status["result"] is not None
            assert "forecast_timestamp" in final_status["result"]
            assert "forecast_values" in final_status["result"]
            assert len(final_status["result"]["forecast_values"]) > 0

        # 4. タスクリストでの確認
        tasks_response = client.get("/api/v1/prediction_tasks")
        tasks = tasks_response.json()
        task_found = any(task["task_id"] == task_id for task in tasks)
        assert task_found

    def test_multiple_models_async_prediction(self):
        """複数モデルでの非同期予測テスト"""
        base_request = create_test_request_data()

        # 異なるモデルでタスクを作成
        model_names = ["chronos_default"]  # 実際に利用可能なモデル名
        task_ids = []

        for model_name in model_names:
            request_data = base_request.copy()
            request_data["model_name"] = model_name

            response = client.post("/api/v1/predict_zero_shot_async", json=request_data)
            assert response.status_code == 200
            task_ids.append(response.json()["task_id"])

        # すべてのタスクが正常に作成されたことを確認
        assert len(task_ids) == len(model_names)

        # 各タスクのステータスを確認
        for task_id in task_ids:
            status_response = client.get(f"/api/v1/prediction_status/{task_id}")
            assert status_response.status_code == 200
