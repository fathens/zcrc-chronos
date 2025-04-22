"""
pytestの設定とフィクスチャを定義するモジュール
"""
import os
import pytest
import yaml
import tempfile
import shutil
from pathlib import Path

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