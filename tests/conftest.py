"""
pytestの設定とフィクスチャを定義するモジュール
"""
import os
import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import sys
from types import ModuleType

@pytest.fixture
def test_config():
    """
    テスト用の設定を提供するフィクスチャ
    """
    return {
        "api": {
            "title": "zcrc-chronos API",
            "description": "時系列予測API",
            "version": "0.1.0",
            "prefix": "/api/v1",
            "cors_origins": ["*"]
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": True,
            "workers": 1,
            "reload": True
        },
        "logging": {
            "level": "INFO",
            "format": "{time} | {level} | {message}",
            "file": "logs/app.log",
            "rotation": "500 MB",
            "retention": "10 days"
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "models_dir": "data/models",
            "default_model": "default_model"
        },
        "security": {
            "enable_auth": False,
            "api_key_header": "X-API-Key",
            "allowed_api_keys": []
        }
    }

@pytest.fixture
def test_model_config():
    """
    テスト用のモデル設定を提供するフィクスチャ
    """
    return {
        "default_model": {
            "name": "chronos_default",
            "version": "1.0.0",
            "description": "デフォルトの時系列予測モデル",
            "chronos": {
                "model_type": "prophet",
                "forecast_horizon": 24,
                "frequency": "H",
                "seasonality_mode": "multiplicative"
            }
        },
        "preprocessing": {
            "impute_missing": True,
            "imputation_method": "linear",
            "normalize": True,
            "normalization_method": "minmax",
            "outlier_detection": True,
            "outlier_method": "iqr"
        },
        "evaluation": {
            "metrics": ["mse", "mae"]
        }
    }

@pytest.fixture
def temp_data_dir():
    """
    テスト用の一時データディレクトリを提供するフィクスチャ
    """
    temp_dir = tempfile.mkdtemp()

    # サブディレクトリを作成
    os.makedirs(os.path.join(temp_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)

    yield temp_dir

    # テスト後にディレクトリを削除
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config_files(test_config, test_model_config, monkeypatch):
    """
    設定ファイルをモックするフィクスチャ
    """
    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp()
    config_dir = os.path.join(temp_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # 設定ファイルを作成
    app_config_path = os.path.join(config_dir, "app_config.yaml")
    model_config_path = os.path.join(config_dir, "model_config.yaml")

    with open(app_config_path, "w") as f:
        yaml.dump(test_config, f)

    with open(model_config_path, "w") as f:
        yaml.dump(test_model_config, f)

    # 環境変数をモック
    monkeypatch.setenv("CONFIG_DIR", config_dir)

    yield {
        "app_config_path": app_config_path,
        "model_config_path": model_config_path,
        "config_dir": config_dir
    }

    # テスト後にディレクトリを削除
    shutil.rmtree(temp_dir)

@pytest.fixture(autouse=True)
def mock_chronos_bolt(monkeypatch):
    """
    chronos-boltライブラリをモックするフィクスチャ
    """
    # モックモジュールの作成
    mock_chronos_bolt = ModuleType("chronos_bolt")

    # ZeroShotPredictorクラスのモック
    class MockZeroShotPredictor:
        def __init__(self, params=None):
            self.params = params or {}

        def predict(self, context=None, horizon=24):
            # 予測結果のモック
            class MockResult:
                def __init__(self):
                    self.values = [10.0 + i * 0.1 for i in range(horizon)]
                    self.confidence_intervals = {
                        'lower_95': [v - 1.0 for v in self.values],
                        'upper_95': [v + 1.0 for v in self.values]
                    }
                    self.metrics = {
                        'mse': 0.15,
                        'mae': 0.12
                    }

            return MockResult()

    # AutoGluonPredictorクラスのモック
    class MockAutoGluonPredictor:
        def __init__(self, params=None):
            self.params = params or {}

        def predict(self, df, horizon=24):
            # 予測結果のモック
            class MockResult:
                def __init__(self):
                    self.values = [10.0 + i * 0.1 for i in range(horizon)]
                    self.confidence_intervals = {
                        'lower_95': [v - 1.0 for v in self.values],
                        'upper_95': [v + 1.0 for v in self.values]
                    }
                    self.metrics = {
                        'mse': 0.15,
                        'mae': 0.12
                    }

            return MockResult()

    # モックモジュールにクラスを追加
    mock_chronos_bolt.ZeroShotPredictor = MockZeroShotPredictor
    mock_chronos_bolt.AutoGluonPredictor = MockAutoGluonPredictor

    # sys.modulesにモックモジュールを追加
    monkeypatch.setitem(sys.modules, "chronos_bolt", mock_chronos_bolt)
