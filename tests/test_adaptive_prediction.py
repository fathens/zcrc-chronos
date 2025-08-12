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
