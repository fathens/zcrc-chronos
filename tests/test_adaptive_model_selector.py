"""
AdaptiveModelSelectorの単体テスト
"""

import datetime
import unittest
from unittest.mock import MagicMock, patch

from src.models.adaptive_model_selector import (
    AdaptiveModelSelector,
    ModelSelectionStrategy,
)
from src.models.time_series_analyzer import TimeSeriesCharacteristics


class TestAdaptiveModelSelector(unittest.TestCase):
    """AdaptiveModelSelectorのテストクラス"""

    def setUp(self):
        """テストの前処理"""
        self.selector = AdaptiveModelSelector()

    def test_init_with_default_config(self):
        """デフォルト設定での初期化テスト"""
        selector = AdaptiveModelSelector()
        self.assertEqual(selector.small_dataset_threshold, 100)
        self.assertEqual(selector.large_dataset_threshold, 1000)
        self.assertIsNotNone(selector.strategies)

    def test_init_with_custom_config(self):
        """カスタム設定での初期化テスト"""
        config = {
            "small_dataset_threshold": 50,
            "large_dataset_threshold": 500,
        }
        selector = AdaptiveModelSelector(config=config)
        self.assertEqual(selector.small_dataset_threshold, 50)
        self.assertEqual(selector.large_dataset_threshold, 500)

    def test_initialize_strategies(self):
        """戦略の初期化テスト"""
        strategies = self.selector._initialize_strategies()

        # 必要な戦略が存在することを確認
        expected_strategies = [
            "strong_seasonal",
            "strong_trend",
            "high_volatility",
            "small_dataset",
            "large_dataset",
            "irregular",
            "balanced",
        ]

        for strategy_name in expected_strategies:
            self.assertIn(strategy_name, strategies)
            strategy = strategies[strategy_name]
            self.assertIsInstance(strategy, ModelSelectionStrategy)
            self.assertIsNotNone(strategy.priority_models)
            self.assertIsNotNone(strategy.excluded_models)
            self.assertIsNotNone(strategy.time_allocation)

    def test_determine_strategy_small_dataset(self):
        """小データセットの戦略決定テスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        strategy = self.selector._determine_strategy(
            characteristics=characteristics,
            data_size=50,  # 小データセット
            horizon=24,
            time_budget=900,
        )

        self.assertEqual(strategy.strategy_name, "small_dataset")

    def test_determine_strategy_large_dataset(self):
        """大データセットの戦略決定テスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        strategy = self.selector._determine_strategy(
            characteristics=characteristics,
            data_size=1500,  # 大データセット
            horizon=24,
            time_budget=900,
        )

        self.assertEqual(strategy.strategy_name, "large_dataset")

    def test_determine_strategy_balanced(self):
        """バランス型戦略の決定テスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        strategy = self.selector._determine_strategy(
            characteristics=characteristics,
            data_size=500,  # 中間サイズ
            horizon=24,
            time_budget=900,
        )

        self.assertEqual(strategy.strategy_name, "balanced")

    def test_determine_strategy_strong_seasonal(self):
        """強い季節性パターンの戦略決定テスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "strong", "score": 0.8},  # 強い季節性
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        strategy = self.selector._determine_strategy(
            characteristics=characteristics,
            data_size=500,
            horizon=24,
            time_budget=900,
        )

        # 強い季節性が検出された場合
        self.assertIn(strategy.strategy_name, ["strong_seasonal", "balanced"])

    def test_determine_strategy_strong_trend(self):
        """強いトレンドパターンの戦略決定テスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "strong", "r_squared": 0.9},  # 強いトレンド
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": False, "adf_pvalue": 0.5},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        strategy = self.selector._determine_strategy(
            characteristics=characteristics,
            data_size=500,
            horizon=24,
            time_budget=900,
        )

        # 強いトレンドが検出された場合
        self.assertIn(strategy.strategy_name, ["strong_trend", "balanced"])

    def test_determine_strategy_high_volatility(self):
        """高ボラティリティパターンの戦略決定テスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.8, "level": "high"},  # 高ボラティリティ
            stationarity={"is_stationary": False, "adf_pvalue": 0.5},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        strategy = self.selector._determine_strategy(
            characteristics=characteristics,
            data_size=500,
            horizon=24,
            time_budget=900,
        )

        # 高ボラティリティが検出された場合
        self.assertIn(strategy.strategy_name, ["high_volatility", "balanced"])

    @patch("src.models.adaptive_model_selector.TimeSeriesAnalyzer")
    def test_select_optimal_strategy(self, mock_analyzer_class):
        """最適戦略選択の統合テスト"""
        # モックの設定
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer

        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.9},
        )
        mock_analyzer.analyze_time_series_characteristics.return_value = characteristics

        # 新しいセレクターを作成（モックを使用）
        selector = AdaptiveModelSelector()

        # データの準備
        values = [10.0 + i * 0.1 for i in range(100)]
        timestamps = [
            datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
            for i in range(100)
        ]

        # 戦略選択の実行
        strategy = selector.select_optimal_strategy(
            values=values,
            timestamps=timestamps,
            horizon=24,
            time_budget=900,
        )

        # 結果の検証
        self.assertIsInstance(strategy, ModelSelectionStrategy)
        self.assertIsNotNone(strategy.strategy_name)
        self.assertIsNotNone(strategy.priority_models)
        self.assertIsNotNone(strategy.excluded_models)

    def test_select_optimal_strategy_with_error(self):
        """エラー時のフォールバック動作テスト"""
        # 分析器をモックしてエラーを発生させる
        with patch.object(
            self.selector.analyzer,
            "analyze_time_series_characteristics",
            side_effect=Exception("Test error"),
        ):
            values = [10.0 + i * 0.1 for i in range(100)]
            timestamps = [
                datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
                for i in range(100)
            ]

            # エラーが発生してもデフォルト戦略が返されることを確認
            strategy = self.selector.select_optimal_strategy(
                values=values,
                timestamps=timestamps,
                horizon=24,
                time_budget=900,
            )

            self.assertEqual(strategy.strategy_name, "balanced")

    def test_custom_threshold_application(self):
        """カスタム閾値の適用テスト"""
        # カスタム閾値での初期化
        config = {
            "small_dataset_threshold": 200,
            "large_dataset_threshold": 800,
        }
        selector = AdaptiveModelSelector(config=config)

        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "weak", "r_squared": 0.1},
            seasonality={"strength": "weak", "score": 0.1},
            volatility={"cv": 0.1, "level": "low"},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.9},
        )

        # 150ポイント：カスタム閾値では小データセット
        strategy = selector._determine_strategy(
            characteristics=characteristics,
            data_size=150,
            horizon=24,
            time_budget=900,
        )
        self.assertEqual(strategy.strategy_name, "small_dataset")

        # 900ポイント：カスタム閾値では大データセット
        strategy = selector._determine_strategy(
            characteristics=characteristics,
            data_size=900,
            horizon=24,
            time_budget=900,
        )
        self.assertEqual(strategy.strategy_name, "large_dataset")

        # 500ポイント：カスタム閾値ではバランス型
        strategy = selector._determine_strategy(
            characteristics=characteristics,
            data_size=500,
            horizon=24,
            time_budget=900,
        )
        self.assertEqual(strategy.strategy_name, "balanced")


if __name__ == "__main__":
    unittest.main()
