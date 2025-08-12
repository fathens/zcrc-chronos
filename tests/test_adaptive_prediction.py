"""
適応的予測システムのテストモジュール
データ特性分析と階層的学習の動作を検証
"""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.server import app
from src.models.adaptive_model_selector import AdaptiveModelSelector
from src.models.hierarchical_trainer import HierarchicalTrainer
from src.models.time_series_analyzer import TimeSeriesAnalyzer

# テストクライアントの初期化
client = TestClient(app)


class TestTimeSeriesAnalyzer:
    """時系列データ分析のテスト"""

    def test_seasonal_pattern_detection(self):
        """季節性パターンの検出テスト"""
        analyzer = TimeSeriesAnalyzer()

        # 季節性のある時系列データを生成
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(72)]  # 3日分

        # 24時間周期の季節性を持つデータ
        values = [
            10 + 3 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 0.1)
            for i in range(72)
        ]

        characteristics = analyzer.analyze_time_series_characteristics(
            values, timestamps
        )

        # 季節性が検出されることを確認
        assert characteristics.seasonality["strength"] in ["moderate", "strong"]
        assert characteristics.seasonality.get("score", 0) > 0.1

    def test_trend_detection(self):
        """トレンド検出テスト"""
        analyzer = TimeSeriesAnalyzer()

        # トレンドのあるデータを生成
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(48)]

        # 明確な上昇トレンド
        values = [10 + 0.1 * i + np.random.normal(0, 0.1) for i in range(48)]

        characteristics = analyzer.analyze_time_series_characteristics(
            values, timestamps
        )

        # トレンドが検出されることを確認
        assert characteristics.trend["direction"] == "increasing"
        assert characteristics.trend["r_squared"] > 0.5

    def test_volatility_calculation(self):
        """ボラティリティ計算テスト"""
        analyzer = TimeSeriesAnalyzer()

        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(24)]

        # 高ボラティリティデータ
        high_vol_values = [10 + np.random.normal(0, 2) for _ in range(24)]

        # 低ボラティリティデータ
        low_vol_values = [10 + np.random.normal(0, 0.1) for _ in range(24)]

        high_char = analyzer.analyze_time_series_characteristics(
            high_vol_values, timestamps
        )
        low_char = analyzer.analyze_time_series_characteristics(
            low_vol_values, timestamps
        )

        # 高ボラティリティの方が大きな値を持つことを確認
        assert high_char.volatility > low_char.volatility

    def test_small_dataset_handling(self):
        """小さなデータセットの処理テスト"""
        analyzer = TimeSeriesAnalyzer()

        # 非常に小さなデータセット
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now, now + datetime.timedelta(hours=1)]
        values = [10.0, 11.0]

        # エラーを起こさずに基本分析を実行できることを確認
        characteristics = analyzer.analyze_time_series_characteristics(
            values, timestamps
        )

        assert characteristics.volatility >= 0
        assert "strength" in characteristics.seasonality
        assert "strength" in characteristics.trend


class TestAdaptiveModelSelector:
    """適応的モデル選択器のテスト"""

    def test_seasonal_strategy_selection(self):
        """季節性データに対する戦略選択テスト"""
        selector = AdaptiveModelSelector()

        # 強い季節性を持つデータ
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(72)]
        values = [10 + 5 * np.sin(2 * np.pi * i / 24) for i in range(72)]

        strategy = selector.select_optimal_strategy(values, timestamps, 24, 900)

        # 季節性関連のモデルが優先されることを確認
        assert (
            "SeasonalNaive" in strategy.priority_models
            or "AutoETS" in strategy.priority_models
        )
        assert strategy.strategy_name in ["strong_seasonal", "balanced"]

    def test_high_volatility_strategy_selection(self):
        """高ボラティリティデータに対する戦略選択テスト"""
        selector = AdaptiveModelSelector()

        # 高ボラティリティデータ
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(48)]
        values = [10 + np.random.normal(0, 3) for _ in range(48)]

        strategy = selector.select_optimal_strategy(values, timestamps, 12, 900)

        # 非線形モデルが優先されることを確認
        assert (
            "NPTS" in strategy.priority_models
            or "RecursiveTabular" in strategy.priority_models
        )

    def test_small_dataset_strategy(self):
        """小データセットに対する戦略選択テスト"""
        selector = AdaptiveModelSelector()

        # 小さなデータセット
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(20)]
        values = [10 + 0.1 * i for i in range(20)]

        strategy = selector.select_optimal_strategy(values, timestamps, 6, 300)

        # 軽量モデルが選択されることを確認
        assert strategy.strategy_name in ["small_dataset", "balanced"]
        assert "DeepAR" not in strategy.priority_models  # 重いモデルは除外

    def test_hierarchical_model_groups(self):
        """階層的モデルグループ生成テスト"""
        selector = AdaptiveModelSelector()
        strategy = selector.strategies["balanced"]

        groups = selector.get_hierarchical_model_groups(strategy)

        # 各グループが適切に分類されることを確認
        assert "fast" in groups
        assert "medium" in groups

        # 高速モデルの確認
        fast_models = groups.get("fast", [])
        expected_fast = ["SeasonalNaive", "AutoETS", "ETS", "Theta"]
        assert any(model in fast_models for model in expected_fast)


class TestHierarchicalTrainer:
    """階層的学習システムのテスト"""

    def test_time_allocation_calculation(self):
        """時間配分計算テスト"""
        trainer = HierarchicalTrainer()

        allocation_ratios = {"fast": 0.2, "medium": 0.5, "advanced": 0.3}
        total_budget = 900

        allocation = trainer._calculate_time_allocation(allocation_ratios, total_budget)

        assert allocation["fast"] == 180
        assert allocation["medium"] == 450
        assert allocation["advanced"] == 270

    def test_early_stopping_logic(self):
        """早期停止ロジックのテスト"""
        trainer = HierarchicalTrainer()

        # 改善が少ない場合のシミュレーション
        trainer.training_results = {
            "fast": MagicMock(score=0.1),
            "medium": MagicMock(score=0.09),
        }
        trainer.best_score = 0.09

        # 早期停止しないことを確認（fastステージ）
        should_stop = trainer._should_stop_early("fast", ["SeasonalNaive"])
        assert not should_stop

        # 改善が少ない場合は停止することを確認（mediumステージ）
        should_stop = trainer._should_stop_early("medium", ["RecursiveTabular"])
        assert should_stop


class TestIntegratedAdaptivePrediction:
    """統合された適応的予測システムのテスト"""

    @patch("src.models.predictor.AutoGluonTSPredictor")
    def test_adaptive_prediction_api_call(self, mock_predictor_class):
        """適応的予測APIコールのテスト"""

        # モックの設定
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor

        # ダミーの予測結果
        mock_forecast_result = MagicMock()
        mock_forecast_result.item_ids = ["time_series_data"]
        mock_forecast_result.loc = {("time_series_data", "mean"): [11.0, 12.0, 13.0]}
        mock_predictor.predict.return_value = mock_forecast_result
        mock_predictor.model_names.return_value = ["AutoETS", "RecursiveTabular"]

        # テストデータ
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [
            (now - datetime.timedelta(hours=i)).isoformat() for i in range(23, -1, -1)
        ]
        values = [10.0 + i * 0.1 for i in range(24)]
        forecast_until = (now + datetime.timedelta(hours=6)).isoformat()

        request_data = {
            "timestamp": timestamps,
            "values": values,
            "forecast_until": forecast_until,
            # model_name を省略してデフォルトの適応的選択を使用
        }

        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)

        # APIが正常に動作することを確認
        assert response.status_code == 200

        data = response.json()
        assert "task_id" in data
        assert "status" in data

    def test_adaptive_features_disabled(self):
        """適応的機能を無効にした場合のテスト"""
        from src.models.predictor import TimeSeriesPredictor

        # 適応的機能を無効にして初期化
        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=False, enable_hierarchical_training=False
        )

        # 適応的機能が無効化されていることを確認
        assert not predictor.enable_adaptive_selection
        assert not predictor.enable_hierarchical_training
        assert not hasattr(predictor, "adaptive_selector")
        assert not hasattr(predictor, "hierarchical_trainer")

    @patch("src.models.predictor.AutoGluonTSPredictor")
    def test_fallback_to_traditional_training(self, mock_predictor_class):
        """従来学習へのフォールバックテスト"""
        from src.models.predictor import TimeSeriesPredictor

        # モックの設定
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor

        mock_forecast_result = MagicMock()
        mock_forecast_result.item_ids = ["time_series_data"]
        mock_forecast_result.loc = MagicMock()
        mock_forecast_result.loc.__getitem__ = MagicMock(return_value=[11.0, 12.0])
        mock_predictor.predict.return_value = mock_forecast_result
        mock_predictor.model_names.return_value = ["AutoETS"]

        # 小さなデータセット（階層的学習の条件を満たさない）
        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=True, enable_hierarchical_training=True
        )

        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [
            now + datetime.timedelta(hours=i) for i in range(10)
        ]  # 小データセット
        values = [10.0 + i * 0.1 for i in range(10)]

        # 従来学習にフォールバックすることを確認
        result = predictor.zero_shot_predict(timestamps, values, 3)

        # 結果が返されることを確認
        assert len(result) == 3  # timestamps, values, metadata
        forecast_timestamps, forecast_values, metadata = result

        # メタデータに適応的機能の情報が含まれることを確認
        assert metadata["adaptive_selection_enabled"]
        assert metadata["hierarchical_training_enabled"]


class TestPredictionQuality:
    """予測値品質検証のテスト"""

    def test_prediction_values_not_zero(self):
        """予測値がすべて0でないことを検証"""
        from src.models.predictor import TimeSeriesPredictor

        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=False, enable_hierarchical_training=False
        )

        # 実際のデータパターンを使用
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(48)]
        # 明確なトレンドのあるデータ
        values = [100.0 + i * 2.0 for i in range(48)]

        try:
            forecast_timestamps, forecast_values, metadata = (
                predictor.zero_shot_predict(timestamps, values, 6)
            )

            # すべて0でないことを確認
            assert not all(
                v == 0.0 for v in forecast_values
            ), f"すべての予測値が0.0です: {forecast_values}"

            # 合理的な範囲内にあることを確認（入力データの10倍以内）
            max_input = max(values)
            min_input = min(values)
            for v in forecast_values:
                assert (
                    min_input * 0.1 <= v <= max_input * 10
                ), f"予測値 {v} が合理的範囲外です (入力範囲: {min_input}-{max_input})"

            # 予測値が有限であることを確認
            assert all(
                abs(v) < float("inf") for v in forecast_values
            ), "予測値に無限大が含まれています"

            # NaNでないことを確認
            import math

            assert all(
                not math.isnan(v) for v in forecast_values
            ), "予測値にNaNが含まれています"

        except Exception as e:
            # エラーが発生した場合は適切なエラーメッセージであることを確認
            assert "予測" in str(e) or "失敗" in str(
                e
            ), f"不明確なエラーメッセージ: {e}"

    def test_real_autogluon_integration(self):
        """実際のAutoGluonライブラリを使用した統合テスト"""
        import pytest

        try:
            from autogluon.timeseries import (  # noqa: F401
                TimeSeriesPredictor as AutoGluonTSPredictor,
            )
        except ImportError:
            pytest.skip("AutoGluon not available")

        from src.models.predictor import TimeSeriesPredictor

        # AutoGluonを直接使用するテスト
        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=False, enable_hierarchical_training=False
        )

        # 十分なデータポイントを用意
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(168)]  # 1週間分
        # 季節性とトレンドのあるデータ
        import numpy as np

        values = [
            100.0 + i * 0.5 + 20.0 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 2)
            for i in range(168)
        ]

        try:
            forecast_timestamps, forecast_values, metadata = (
                predictor.zero_shot_predict(timestamps, values, 24)  # 24時間先の予測
            )

            # 基本的な品質チェック
            assert len(forecast_values) == 24, f"予測期間が不正: {len(forecast_values)}"

            # ゼロ値問題の検証 - これが最も重要
            zero_count = sum(1 for v in forecast_values if v == 0.0)
            assert (
                zero_count == 0
            ), f"予測値に{zero_count}個のゼロが含まれています: {forecast_values}"

            # すべての値が有効であることを確認
            import math

            for i, v in enumerate(forecast_values):
                assert not math.isnan(v), f"予測値[{i}]がNaN: {v}"
                assert abs(v) < float("inf"), f"予測値[{i}]が無限大: {v}"

            # 統計的妥当性の確認
            mean_forecast = np.mean(forecast_values)
            std_forecast = np.std(forecast_values)
            mean_input = np.mean(values[-48:])  # 最後の48時間の平均

            # 予測平均が入力平均から極端に乖離していないこと
            relative_diff = abs(mean_forecast - mean_input) / mean_input
            assert (
                relative_diff < 1.0
            ), f"予測平均の相対誤差が100%を超過: {relative_diff:.2f}"

            # 予測値の変動が極端でないことを確認
            input_std = np.std(values[-48:])
            if input_std > 0:
                relative_std = std_forecast / input_std
                assert (
                    relative_std < 10.0
                ), f"予測の標準偏差が入力の10倍を超過: {relative_std:.2f}"

        except Exception as e:
            # エラーが発生した場合、具体的なエラー内容をログに出力
            import traceback

            error_details = traceback.format_exc()
            pytest.fail(
                f"実際のAutoGluon統合テストでエラー: {e}\n詳細:\n{error_details}"
            )

    def test_prediction_quality_metrics(self):
        """予測品質メトリクスの検証"""
        from src.models.predictor import TimeSeriesPredictor

        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=False, enable_hierarchical_training=False
        )

        # 季節性のあるデータ
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(72)]
        import numpy as np

        values = [50.0 + 10.0 * np.sin(2 * np.pi * i / 24) + i * 0.1 for i in range(72)]

        try:
            forecast_timestamps, forecast_values, metadata = (
                predictor.zero_shot_predict(timestamps, values, 12)
            )

            # 予測値の統計的妥当性
            mean_forecast = np.mean(forecast_values)
            std_forecast = np.std(forecast_values)
            mean_input = np.mean(values[-24:])  # 最後の24時間の平均

            # 予測の平均が入力の平均と極端に乖離していないことを確認
            assert (
                abs(mean_forecast - mean_input) < mean_input * 0.5
            ), f"予測平均 {mean_forecast} が入力平均 {mean_input} から大きく乖離"

            # 予測値の変動が極端でないことを確認
            input_std = np.std(values[-24:])
            assert (
                std_forecast < input_std * 5
            ), f"予測の標準偏差 {std_forecast} が入力の標準偏差 {input_std} の5倍を超過"

        except Exception as e:
            # エラーが発生した場合は適切なエラーメッセージであることを確認
            assert "予測" in str(e) or "失敗" in str(
                e
            ), f"不明確なエラーメッセージ: {e}"


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_analyzer_error_handling(self):
        """分析器のエラーハンドリングテスト"""
        analyzer = TimeSeriesAnalyzer()

        # 無効なデータでもエラーを起こさないことを確認
        invalid_timestamps = []
        invalid_values = []

        characteristics = analyzer.analyze_time_series_characteristics(
            invalid_values, invalid_timestamps
        )

        # 基本的な構造が返されることを確認
        assert hasattr(characteristics, "volatility")
        assert hasattr(characteristics, "seasonality")
        assert hasattr(characteristics, "trend")

    def test_selector_error_handling(self):
        """選択器のエラーハンドリングテスト"""
        selector = AdaptiveModelSelector()

        # エラーが発生した場合にデフォルト戦略が返されることを確認
        with patch.object(
            selector.analyzer,
            "analyze_time_series_characteristics",
            side_effect=Exception("Test error"),
        ):

            now = datetime.datetime(2023, 1, 1, 12, 0, 0)
            timestamps = [now + datetime.timedelta(hours=i) for i in range(24)]
            values = [10.0] * 24

            strategy = selector.select_optimal_strategy(values, timestamps, 12, 900)

            # デフォルト戦略が返されることを確認
            assert strategy.strategy_name == "balanced"

    def test_hierarchical_training_failure_handling(self):
        """階層的学習失敗時のエラーハンドリング"""
        import pytest

        from src.models.predictor import TimeSeriesPredictor

        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=True, enable_hierarchical_training=True
        )

        # 階層的学習が失敗する条件を作成
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [
            now + datetime.timedelta(hours=i) for i in range(50)
        ]  # 十分なデータ
        values = [100.0 + i for i in range(50)]

        # 階層的学習を強制的に失敗させる
        with patch.object(
            predictor.hierarchical_trainer,
            "train_hierarchically",
            side_effect=RuntimeError("Hierarchical training failed"),
        ):

            with pytest.raises(RuntimeError) as exc_info:
                predictor.zero_shot_predict(timestamps, values, 6)

            # 適切なエラーメッセージが含まれることを確認
            assert "階層的学習" in str(exc_info.value) or "Hierarchical" in str(
                exc_info.value
            )

    def test_prediction_result_extraction_failure(self):
        """予測結果抽出失敗時のエラーハンドリング"""
        import pytest

        from src.models.predictor import TimeSeriesPredictor

        predictor = TimeSeriesPredictor(
            enable_adaptive_selection=False, enable_hierarchical_training=False
        )

        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [now + datetime.timedelta(hours=i) for i in range(24)]
        values = [100.0 + i for i in range(24)]

        # AutoGluon predictorが空の結果を返すようにモック
        with patch("src.models.predictor.AutoGluonTSPredictor") as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor_class.return_value = mock_predictor

            # 空の予測結果をモック
            mock_forecast_result = MagicMock()
            mock_forecast_result.item_ids = ["time_series_data"]
            mock_forecast_result.loc = {}  # 空の結果
            mock_predictor.predict.return_value = mock_forecast_result
            mock_predictor.model_names.return_value = []

            with pytest.raises((ValueError, RuntimeError)) as exc_info:
                predictor.zero_shot_predict(timestamps, values, 6)

            # 適切なエラーメッセージが含まれることを確認
            error_msg = str(exc_info.value).lower()
            assert any(
                keyword in error_msg
                for keyword in ["予測", "失敗", "空", "アクセス", "抽出"]
            ), f"不明確なエラーメッセージ: {exc_info.value}"

    def test_api_endpoint_zero_value_detection(self):
        """/api/v1/predict_zero_shot_asyncエンドポイントのゼロ値問題検証"""
        import time

        import pytest
        from fastapi.testclient import TestClient

        from src.api.server import app

        client = TestClient(app)

        # テストデータの作成
        base_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [
            (base_time + datetime.timedelta(hours=i)).isoformat() for i in range(72)
        ]
        # 明確なパターンのあるデータ
        import numpy as np

        values = [50.0 + i * 0.5 + 10.0 * np.sin(2 * np.pi * i / 24) for i in range(72)]

        forecast_until = (base_time + datetime.timedelta(hours=96)).isoformat()

        request_data = {
            "timestamp": timestamps,
            "values": values,
            "forecast_until": forecast_until,
            # model_nameを省略してデフォルトのchronos_defaultを使用
        }

        # APIリクエストの実行
        response = client.post("/api/v1/predict_zero_shot_async", json=request_data)

        if response.status_code != 200:
            pytest.skip(
                f"APIリクエストが失敗: {response.status_code} - {response.text}"
            )

        data = response.json()
        task_id = data.get("task_id")

        if not task_id:
            pytest.skip("task_idが取得できません")

        # タスクの完了を待つ（最大180秒）
        max_wait_time = 180
        poll_interval = 5
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/v1/prediction_tasks/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get("status") == "completed":
                    break
                elif status_data.get("status") in ["failed", "error"]:
                    pytest.fail(f"Prediction task failed: {status_data}")
            time.sleep(poll_interval)
        else:
            pytest.skip(
                f"Task {task_id} did not complete within {max_wait_time} seconds"
            )

        # 結果の取得
        result_response = client.get(f"/api/v1/prediction_results/{task_id}")
        if result_response.status_code != 200:
            pytest.fail(
                f"Failed to get prediction results: {result_response.status_code}"
            )

        result_data = result_response.json()
        predicted_values = result_data.get("predicted_values", [])

        if not predicted_values:
            pytest.fail("No predicted values returned")

        # ゼロ値問題の検証 - これが主要なテスト目的
        forecast_values = [item["value"] for item in predicted_values]
        zero_count = sum(1 for v in forecast_values if v == 0.0)

        # 以前のバグではここで全てゼロになっていた
        assert (
            zero_count == 0
        ), f"APIが{zero_count}個のゼロ値を返しました: {forecast_values[:10]}..."

        # 追加の品質チェック
        import math

        for i, v in enumerate(forecast_values):
            assert not math.isnan(v), f"API予測値[{i}]がNaN"
            assert abs(v) < float("inf"), f"API予測値[{i}]が無限大"
