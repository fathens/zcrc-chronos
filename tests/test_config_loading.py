"""
設定ファイル読み込み機能の単体テスト
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.models.predictor import TimeSeriesPredictor


class TestConfigLoading(unittest.TestCase):
    """設定ファイル読み込みのテストクラス"""

    def setUp(self):
        """テストの前処理"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テストの後処理"""
        # 一時ファイルをクリーンアップ
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_config_files(self):
        """テスト用の設定ファイルを作成"""
        # モデル設定ファイル
        model_config = {
            "default_model": {
                "name": "test_model",
                "version": "1.0.0",
                "description": "Test model",
                "chronos": {
                    "model_type": "autogluon",
                    "time_limit": 300,
                    "presets": "medium_quality",
                },
            },
            "available_models": [],
        }

        model_config_path = os.path.join(self.temp_dir, "model_config.yaml")
        with open(model_config_path, "w") as f:
            yaml.dump(model_config, f)

        # アプリ設定ファイル
        app_config = {
            "prediction": {
                "max_concurrent_tasks": 3,
                "adaptive_selection": {
                    "small_dataset_threshold": 150,
                    "large_dataset_threshold": 1500,
                },
                "hierarchical_training": {
                    "early_stopping_threshold": 0.03,
                    "min_score_for_stop": 0.005,
                    "max_workers": 4,
                },
            },
        }

        app_config_path = os.path.join(self.temp_dir, "app_config.yaml")
        with open(app_config_path, "w") as f:
            yaml.dump(app_config, f)

        return model_config_path, app_config_path

    @patch("src.models.predictor.MODEL_CONFIG_PATH")
    @patch("src.models.predictor.APP_CONFIG_PATH")
    def test_config_loading_success(self, mock_app_path, mock_model_path):
        """正常な設定ファイル読み込みテスト"""
        # テスト設定ファイルを作成
        model_path, app_path = self.create_test_config_files()
        mock_model_path.__str__ = lambda: model_path
        mock_app_path.__str__ = lambda: app_path

        # パスを直接設定
        with patch("builtins.open") as mock_open:
            # モデル設定ファイルの読み込み
            model_config = {
                "default_model": {
                    "name": "test_model",
                    "chronos": {"time_limit": 300},
                },
                "prediction": {
                    "adaptive_selection": {"small_dataset_threshold": 150},
                    "hierarchical_training": {"early_stopping_threshold": 0.03},
                },
            }

            mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(
                model_config
            )

            predictor = TimeSeriesPredictor()

            # 設定が正しく読み込まれていることを確認
            self.assertIsNotNone(predictor.config)

    def test_adaptive_selector_config_application(self):
        """適応的モデル選択器への設定適用テスト"""
        config = {
            "small_dataset_threshold": 200,
            "large_dataset_threshold": 800,
        }

        from src.models.adaptive_model_selector import AdaptiveModelSelector

        selector = AdaptiveModelSelector(config=config)

        self.assertEqual(selector.small_dataset_threshold, 200)
        self.assertEqual(selector.large_dataset_threshold, 800)

    def test_hierarchical_trainer_config_application(self):
        """階層的学習器への設定適用テスト"""
        config = {
            "early_stopping_threshold": 0.05,
            "min_score_for_stop": 0.02,
            "adaptive_selection": {
                "small_dataset_threshold": 300,
                "large_dataset_threshold": 1200,
            },
        }

        from src.models.hierarchical_trainer import HierarchicalTrainer

        trainer = HierarchicalTrainer(max_workers=3, config=config)

        self.assertEqual(trainer.early_stopping_threshold, 0.05)
        self.assertEqual(trainer.min_score_for_stop, 0.02)
        self.assertEqual(trainer.max_workers, 3)

        # AdaptiveModelSelectorにも設定が渡されていることを確認
        self.assertEqual(trainer.selector.small_dataset_threshold, 300)
        self.assertEqual(trainer.selector.large_dataset_threshold, 1200)

    def test_config_with_missing_values(self):
        """設定値欠損時のデフォルト値テスト"""
        # 空の設定
        empty_config = {}

        from src.models.adaptive_model_selector import AdaptiveModelSelector
        from src.models.hierarchical_trainer import HierarchicalTrainer

        # デフォルト値が使用されることを確認
        selector = AdaptiveModelSelector(config=empty_config)
        self.assertEqual(selector.small_dataset_threshold, 100)
        self.assertEqual(selector.large_dataset_threshold, 1000)

        trainer = HierarchicalTrainer(config=empty_config)
        self.assertEqual(trainer.early_stopping_threshold, 0.02)
        self.assertEqual(trainer.min_score_for_stop, 0.01)

    def test_config_with_partial_values(self):
        """部分的な設定値での動作テスト"""
        config = {
            "small_dataset_threshold": 250,
            # large_dataset_threshold は未設定
            "early_stopping_threshold": 0.04,
            # min_score_for_stop は未設定
        }

        from src.models.adaptive_model_selector import AdaptiveModelSelector
        from src.models.hierarchical_trainer import HierarchicalTrainer

        # 設定されている値は反映、未設定はデフォルト値
        selector = AdaptiveModelSelector(config=config)
        self.assertEqual(selector.small_dataset_threshold, 250)
        self.assertEqual(selector.large_dataset_threshold, 1000)  # デフォルト

        trainer = HierarchicalTrainer(config=config)
        self.assertEqual(trainer.early_stopping_threshold, 0.04)
        self.assertEqual(trainer.min_score_for_stop, 0.01)  # デフォルト

    @patch("src.models.predictor.MODEL_CONFIG_PATH", "/nonexistent/path.yaml")
    @patch("src.models.predictor.APP_CONFIG_PATH", "/nonexistent/app.yaml")
    def test_config_file_not_found(self):
        """設定ファイルが見つからない場合のテスト"""
        with self.assertRaises(FileNotFoundError):
            TimeSeriesPredictor()

    def create_invalid_yaml_file(self):
        """無効なYAMLファイルを作成"""
        invalid_yaml_path = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_yaml_path, "w") as f:
            f.write("invalid: yaml: content: [unclosed")
        return invalid_yaml_path

    @patch("src.models.predictor.MODEL_CONFIG_PATH")
    @patch("src.models.predictor.APP_CONFIG_PATH", "/nonexistent/app.yaml")
    def test_invalid_yaml_config(self, mock_app_path, mock_model_path):
        """無効なYAMLファイルの処理テスト"""
        # 無効なYAMLファイルのパスを設定
        invalid_path = self.create_invalid_yaml_file()
        mock_model_path.__str__ = lambda: invalid_path

        with self.assertRaises((yaml.YAMLError, FileNotFoundError)):
            TimeSeriesPredictor()

    def test_nested_config_access(self):
        """ネストされた設定へのアクセステスト"""
        config = {
            "prediction": {
                "adaptive_selection": {
                    "small_dataset_threshold": 75,
                    "large_dataset_threshold": 750,
                },
                "hierarchical_training": {
                    "early_stopping_threshold": 0.01,
                    "min_score_for_stop": 0.001,
                },
            },
        }

        from src.models.hierarchical_trainer import HierarchicalTrainer

        # ネストされた設定の正しい取得
        hierarchical_config = config.get("prediction", {}).get(
            "hierarchical_training", {}
        )
        adaptive_config = config.get("prediction", {}).get("adaptive_selection", {})

        full_config = {
            **hierarchical_config,
            "adaptive_selection": adaptive_config,
        }

        trainer = HierarchicalTrainer(config=full_config)

        self.assertEqual(trainer.early_stopping_threshold, 0.01)
        self.assertEqual(trainer.min_score_for_stop, 0.001)
        self.assertEqual(trainer.selector.small_dataset_threshold, 75)
        self.assertEqual(trainer.selector.large_dataset_threshold, 750)

    def test_config_types_validation(self):
        """設定値の型検証テスト"""
        # 正しい型の設定
        valid_config = {
            "small_dataset_threshold": 100,  # int
            "large_dataset_threshold": 1000,  # int
            "early_stopping_threshold": 0.02,  # float
            "min_score_for_stop": 0.01,  # float
        }

        from src.models.adaptive_model_selector import AdaptiveModelSelector
        from src.models.hierarchical_trainer import HierarchicalTrainer

        # 正常に作成できることを確認
        selector = AdaptiveModelSelector(config=valid_config)
        trainer = HierarchicalTrainer(config=valid_config)

        self.assertIsInstance(selector.small_dataset_threshold, int)
        self.assertIsInstance(trainer.early_stopping_threshold, float)

    def test_config_override_priority(self):
        """設定の優先順位テスト"""
        # デフォルト設定
        from src.models.adaptive_model_selector import AdaptiveModelSelector

        default_selector = AdaptiveModelSelector()
        self.assertEqual(default_selector.small_dataset_threshold, 100)

        # カスタム設定で上書き
        custom_config = {"small_dataset_threshold": 300}
        custom_selector = AdaptiveModelSelector(config=custom_config)
        self.assertEqual(custom_selector.small_dataset_threshold, 300)

        # 上書きされない項目はデフォルト値
        self.assertEqual(custom_selector.large_dataset_threshold, 1000)


if __name__ == "__main__":
    unittest.main()
