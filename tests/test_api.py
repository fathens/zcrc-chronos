"""
APIエンドポイントのテストモジュール
"""
import pytest
from fastapi.testclient import TestClient
import json
import datetime
from src.api.server import app
from src.api.routes import ZeroShotPredictionRequest

# テストクライアントの初期化
client = TestClient(app)

def test_root_endpoint():
    """
    ルートエンドポイントのテスト
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "description" in data
    assert "docs" in data

def test_health_endpoint():
    """
    ヘルスチェックエンドポイントのテスト
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_version" in data

def test_models_endpoint():
    """
    モデル一覧エンドポイントのテスト
    """
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    if len(models) > 0:
        model = models[0]
        assert "name" in model
        assert "version" in model
        assert "description" in model
        assert "parameters" in model

def test_predict_endpoint():
    """
    予測エンドポイントのテスト
    """
    # テスト用のリクエストデータ
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    timestamps_str = [ts.isoformat() for ts in timestamps]

    request_data = {
        "data": {
            "timestamp": timestamps_str,
            "values": [10.0 + i * 0.1 for i in range(24)]
        },
        "horizon": 12,
        "model_name": "chronos_default"
    }

    response = client.post("/api/v1/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()

    # レスポンスの検証
    assert "forecast_timestamp" in data
    assert "forecast_values" in data
    assert "model_name" in data
    assert "confidence_intervals" in data
    assert "metrics" in data

    # 予測値の数が指定したhorizonと一致することを確認
    assert len(data["forecast_timestamp"]) == request_data["horizon"]
    assert len(data["forecast_values"]) == request_data["horizon"]

def test_predict_endpoint_with_forecast_until():
    """
    forecast_untilパラメータを使用した予測エンドポイントのテスト
    """
    # テスト用のリクエストデータ
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    timestamps_str = [ts.isoformat() for ts in timestamps]

    # 予測終了時刻を設定（最後のタイムスタンプから12時間後）
    forecast_until = now + datetime.timedelta(hours=12)

    request_data = {
        "data": {
            "timestamp": timestamps_str,
            "values": [10.0 + i * 0.1 for i in range(24)]
        },
        "forecast_until": forecast_until.isoformat(),
        "model_name": "chronos_default"
    }

    response = client.post("/api/v1/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()

    # レスポンスの検証
    assert "forecast_timestamp" in data
    assert "forecast_values" in data
    assert "model_name" in data
    assert "confidence_intervals" in data
    assert "metrics" in data

    # 予測値が存在することを確認
    assert len(data["forecast_timestamp"]) > 0
    assert len(data["forecast_values"]) > 0

    # 最後の予測時刻がforecast_until以降であることを確認
    last_forecast_time = datetime.datetime.fromisoformat(data["forecast_timestamp"][-1])
    assert last_forecast_time >= forecast_until

def test_predict_endpoint_invalid_data():
    """
    予測エンドポイントの無効なデータに対するテスト
    """
    # タイムスタンプと値の長さが一致しない無効なデータ
    invalid_data = {
        "data": {
            "timestamp": ["2023-01-01T00:00:00", "2023-01-01T01:00:00"],
            "values": [10.0]  # 値が1つしかない
        },
        "horizon": 12
    }

    response = client.post("/api/v1/predict", json=invalid_data)
    # バリデーションエラーが発生することを期待
    assert response.status_code == 422

def test_zero_shot_predict_endpoint():
    """
    ゼロショット予測エンドポイントのテスト
    """
    # テスト用のリクエストデータ
    request_data = {
        "context": "今後の株価は上昇傾向にあり、安定した成長が見込まれる。",
        "horizon": 12,
        "model_name": "chronos_default"
    }

    response = client.post("/api/v1/predict_zero_shot", json=request_data)
    assert response.status_code == 200
    data = response.json()

    # レスポンスの検証
    assert "forecast_timestamp" in data
    assert "forecast_values" in data
    assert "model_name" in data
    assert "confidence_intervals" in data
    assert "metrics" in data

    # 予測値の数が指定したhorizonと一致することを確認
    assert len(data["forecast_timestamp"]) == request_data["horizon"]
    assert len(data["forecast_values"]) == request_data["horizon"]

    # コンテキストの検証は削除（PredictionResponseにはcontextフィールドがないため）

def test_zero_shot_predict_endpoint_invalid_data():
    """
    ゼロショット予測エンドポイントの無効なデータに対するテスト
    """
    # contextが欠けている無効なデータ
    invalid_data = {
        "horizon": 12,
        "model_name": "chronos_default"
    }

    response = client.post("/api/v1/predict_zero_shot", json=invalid_data)
    # バリデーションエラーが発生することを期待
    assert response.status_code == 422

def test_predict_with_autogluon_endpoint():
    """
    AutoGluon予測エンドポイントのテスト
    """
    # テスト用のリクエストデータ
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    timestamps_str = [ts.isoformat() for ts in timestamps]

    request_data = {
        "data": {
            "timestamp": timestamps_str,
            "values": [10.0 + i * 0.1 for i in range(24)]
        },
        "horizon": 12,
        "model_name": "chronos_default",
        "model_params": {
            "autogluon_params": {
                "presets": "base"
            }
        }
    }

    response = client.post("/api/v1/predict_with_autogluon", json=request_data)
    assert response.status_code == 200
    data = response.json()

    # レスポンスの検証
    assert "forecast_timestamp" in data
    assert "forecast_values" in data
    assert "model_name" in data
    assert "confidence_intervals" in data
    assert "metrics" in data

    # 予測値の数が指定したhorizonと一致することを確認
    assert len(data["forecast_timestamp"]) == request_data["horizon"]
    assert len(data["forecast_values"]) == request_data["horizon"]

def test_predict_with_autogluon_endpoint_invalid_data():
    """
    AutoGluon予測エンドポイントの無効なデータに対するテスト
    """
    # タイムスタンプと値の長さが一致しない無効なデータ
    invalid_data = {
        "data": {
            "timestamp": ["2023-01-01T00:00:00", "2023-01-01T01:00:00"],
            "values": [10.0]  # 値が1つしかない
        },
        "horizon": 12
    }

    response = client.post("/api/v1/predict_with_autogluon", json=invalid_data)
    # バリデーションエラーが発生することを期待
    assert response.status_code == 422
