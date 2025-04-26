"""
APIエンドポイントのテストモジュール
"""

import datetime

from fastapi.testclient import TestClient

from src.api.server import app

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


def test_zero_shot_predict_endpoint():
    """
    ゼロショット予測エンドポイントのテスト
    """
    # テスト用のリクエストデータ
    now = datetime.datetime.now()
    # 過去24時間分のダミー時系列データを生成
    timestamps = [
        (now - datetime.timedelta(hours=i)).isoformat() for i in range(24, 0, -1)
    ]
    values = [10.0 + i * 0.1 for i in range(24)]

    request_data = {
        "timestamp": timestamps,
        "values": values,
        "horizon": 12,
        "model_name": "chronos_default",
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


def test_zero_shot_predict_endpoint_invalid_data():
    """
    ゼロショット予測エンドポイントの無効なデータに対するテスト
    """
    # timestampとvaluesの長さが一致しない無効なデータ
    now = datetime.datetime.now()
    timestamps = [
        (now - datetime.timedelta(hours=i)).isoformat() for i in range(5, 0, -1)
    ]
    values = [10.0, 11.0, 12.0]  # 長さが一致しない

    invalid_data = {
        "timestamp": timestamps,
        "values": values,
        "horizon": 12,
        "model_name": "chronos_default",
    }

    response = client.post("/api/v1/predict_zero_shot", json=invalid_data)
    # バリデーションエラーが発生することを期待
    assert response.status_code == 422
