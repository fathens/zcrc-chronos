"""
HierarchicalTrainerの単体テスト
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from src.models.hierarchical_trainer import HierarchicalTrainer


class TestHierarchicalTrainer(unittest.TestCase):
    """HierarchicalTrainerのテストクラス"""

    def setUp(self):
        """テストの前処理"""
        self.trainer = HierarchicalTrainer()

    def test_init_with_default_config(self):
        """デフォルト設定での初期化テスト"""
        trainer = HierarchicalTrainer()
        self.assertEqual(trainer.max_workers, 2)
        self.assertEqual(trainer.early_stopping_threshold, 0.02)
        self.assertEqual(trainer.min_score_for_stop, 0.01)
        self.assertIsNotNone(trainer.selector)
        self.assertEqual(trainer.best_score, float("inf"))

    def test_init_with_custom_config(self):
        """カスタム設定での初期化テスト"""
        config = {
            "early_stopping_threshold": 0.05,
            "min_score_for_stop": 0.02,
            "adaptive_selection": {
                "small_dataset_threshold": 50,
                "large_dataset_threshold": 500,
            },
        }
        trainer = HierarchicalTrainer(max_workers=4, config=config)
        self.assertEqual(trainer.max_workers, 4)
        self.assertEqual(trainer.early_stopping_threshold, 0.05)
        self.assertEqual(trainer.min_score_for_stop, 0.02)

    def test_calculate_time_allocation(self):
        """時間配分計算のテスト"""
        allocation_ratios = {"fast": 0.2, "medium": 0.5, "advanced": 0.3}
        total_budget = 1000

        result = self.trainer._calculate_time_allocation(
            allocation_ratios, total_budget
        )

        self.assertEqual(result["fast"], 200)
        self.assertEqual(result["medium"], 500)
        self.assertEqual(result["advanced"], 300)

    def test_should_stop_early_fast_stage(self):
        """fastステージでの早期停止判定テスト"""
        # fastステージでは停止しない
        should_stop = self.trainer._should_stop_early("fast", ["model1", "model2"])
        self.assertFalse(should_stop)

    def test_should_stop_early_good_score(self):
        """良好なスコアでの早期停止判定テスト"""
        # 良いスコアを設定
        self.trainer.best_score = 0.005  # 0.5%のエラー

        should_stop = self.trainer._should_stop_early("medium", ["model1", "model2"])
        self.assertTrue(should_stop)

    def test_should_stop_early_insufficient_improvement(self):
        """改善不足での早期停止判定テスト"""
        # 複数の結果をモック
        mock_result1 = MagicMock()
        mock_result1.score = 0.10
        mock_result2 = MagicMock()
        mock_result2.score = 0.09  # わずかな改善

        self.trainer.training_results["stage1"] = mock_result1
        self.trainer.training_results["stage2"] = mock_result2

        should_stop = self.trainer._should_stop_early("advanced", ["model3"])
        # 改善率が閾値未満の場合は停止
        self.assertTrue(should_stop)

    def test_should_stop_early_sufficient_improvement(self):
        """十分な改善での継続判定テスト"""
        # 複数の結果をモック（大きな改善）
        mock_result1 = MagicMock()
        mock_result1.score = 0.20
        mock_result2 = MagicMock()
        mock_result2.score = 0.10  # 50%の改善

        self.trainer.training_results["stage1"] = mock_result1
        self.trainer.training_results["stage2"] = mock_result2
        self.trainer.best_score = 0.10

        should_stop = self.trainer._should_stop_early("advanced", ["model3"])
        # 大きな改善がある場合は継続
        self.assertFalse(should_stop)

    def test_summarize_results_empty(self):
        """結果が空の場合のサマリーテスト"""
        summary = self.trainer._summarize_results()
        self.assertEqual(summary["total_stages"], 0)
        self.assertEqual(summary["stages"], [])

    def test_summarize_results_with_data(self):
        """結果がある場合のサマリーテスト"""
        # テストデータを追加
        mock_result1 = MagicMock()
        mock_result1.stage = "fast"
        mock_result1.models = ["model1", "model2"]
        mock_result1.score = 0.15
        mock_result1.time_spent = 10.0

        mock_result2 = MagicMock()
        mock_result2.stage = "medium"
        mock_result2.models = ["model3"]
        mock_result2.score = 0.10
        mock_result2.time_spent = 20.0

        self.trainer.training_results["stage1"] = mock_result1
        self.trainer.training_results["stage2"] = mock_result2

        summary = self.trainer._summarize_results()

        self.assertEqual(summary["total_stages"], 2)
        self.assertEqual(len(summary["stages"]), 2)

    @patch("src.models.hierarchical_trainer.AutoGluonTSPredictor")
    def test_train_stage_success(self, mock_predictor_class):
        """ステージ学習の成功テスト"""
        # モックの設定
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor

        # 予測結果のモック
        mock_result = MagicMock()
        mock_predictor.predict.return_value = mock_result

        # データの準備
        df = pd.DataFrame(
            {
                "item_id": ["item1"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
                "target": [10.0 + i for i in range(10)],
            }
        )
        time_series_data = TimeSeriesDataFrame(df)

        # ステージ学習の実行
        result = self.trainer._train_stage(
            predictor_class=mock_predictor_class,
            predictor_kwargs={"save_path": "/tmp/test"},
            time_series_data=time_series_data,
            models=["AutoETS", "ETS"],
            excluded_models=["Naive"],
            time_budget=60,
            horizon=5,
            stage="test",
        )

        # 結果の検証
        self.assertEqual(result, mock_result)
        mock_predictor.fit.assert_called_once()
        mock_predictor.predict.assert_called_once()

    def test_train_stage_zero_budget(self):
        """時間予算ゼロでのステージ学習テスト"""
        result = self.trainer._train_stage(
            predictor_class=MagicMock,
            predictor_kwargs={},
            time_series_data=None,
            models=["model1"],
            excluded_models=[],
            time_budget=0,  # ゼロ予算
            horizon=5,
            stage="test",
        )

        self.assertIsNone(result)

    @patch("src.models.hierarchical_trainer.AutoGluonTSPredictor")
    def test_train_stage_with_error(self, mock_predictor_class):
        """ステージ学習のエラーハンドリングテスト"""
        # エラーを発生させる
        mock_predictor_class.side_effect = Exception("Test error")

        # データの準備
        df = pd.DataFrame(
            {
                "item_id": ["item1"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
                "target": [10.0 + i for i in range(10)],
            }
        )
        time_series_data = TimeSeriesDataFrame(df)

        # ステージ学習の実行（エラーが発生してもNoneが返される）
        result = self.trainer._train_stage(
            predictor_class=mock_predictor_class,
            predictor_kwargs={"save_path": "/tmp/test"},
            time_series_data=time_series_data,
            models=["AutoETS"],
            excluded_models=["Naive"],
            time_budget=60,
            horizon=5,
            stage="test",
        )

        self.assertIsNone(result)

    def test_update_best_score(self):
        """ベストスコア更新のテスト"""
        # 初期状態
        self.assertEqual(self.trainer.best_score, float("inf"))

        # より良いスコアで更新
        self.trainer._update_best_score(0.5)
        self.assertEqual(self.trainer.best_score, 0.5)

        # より悪いスコアでは更新されない
        self.trainer._update_best_score(1.0)
        self.assertEqual(self.trainer.best_score, 0.5)

        # さらに良いスコアで更新
        self.trainer._update_best_score(0.3)
        self.assertEqual(self.trainer.best_score, 0.3)

    def test_thread_safety(self):
        """スレッドセーフティのテスト"""
        # 複数スレッドからベストスコアを更新
        scores = []

        def update_score(score):
            self.trainer._update_best_score(score)
            scores.append(self.trainer.best_score)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_score, args=(i * 0.1,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 最小値が正しく設定されていることを確認
        self.assertEqual(self.trainer.best_score, 0.0)

    def test_custom_early_stopping_threshold(self):
        """カスタム早期停止閾値のテスト"""
        config = {
            "early_stopping_threshold": 0.10,  # 10%改善が必要
            "min_score_for_stop": 0.005,  # 0.5%未満で停止
        }
        trainer = HierarchicalTrainer(config=config)

        # スコアが閾値より良い場合
        trainer.best_score = 0.003
        should_stop = trainer._should_stop_early("medium", ["model1"])
        self.assertTrue(should_stop)

        # スコアが閾値より悪い場合
        trainer.best_score = 0.01
        should_stop = trainer._should_stop_early("medium", ["model1"])
        self.assertFalse(should_stop)


if __name__ == "__main__":
    unittest.main()
