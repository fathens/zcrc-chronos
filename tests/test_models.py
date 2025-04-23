"""
時系列予測モデルのテストモジュール
"""
import os
import pytest
import datetime
import pandas as pd
import numpy as np
from src.models.predictor import TimeSeriesPredictor

def test_predictor_initialization():
    """
    TimeSeriesPredictorの初期化テスト
    """
    # デフォルトパラメータでの初期化
    predictor = TimeSeriesPredictor()
    assert predictor.model_name == "chronos_default"
    assert predictor.model_params == {}
    assert predictor.model is None

    # カスタムパラメータでの初期化
    custom_params = {"seasonality_mode": "multiplicative"}
    predictor = TimeSeriesPredictor(model_name="custom_model", model_params=custom_params)
    assert predictor.model_name == "custom_model"
    assert predictor.model_params == custom_params
    assert predictor.model is None

def test_prepare_data():
    """
    データ準備メソッドのテスト
    """
    predictor = TimeSeriesPredictor()

    # テストデータの作成
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    values = [10.0 + i * 0.1 for i in range(24)]

    # データ準備メソッドの呼び出し
    df = predictor._prepare_data(timestamps, values)

    # 結果の検証
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert 'value' in df.columns
    assert df.index.name == 'timestamp'

def test_fit_predict():
    """
    モデルの学習と予測のテスト
    """
    predictor = TimeSeriesPredictor()

    # テストデータの作成
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    values = [10.0 + i * 0.1 for i in range(24)]

    # モデルの学習
    predictor.fit(timestamps, values)
    assert predictor.model is not None

    # 予測の実行
    horizon = 12
    forecast_timestamps, forecast_values, metadata = predictor.predict(horizon=horizon)

    # 結果の検証
    assert len(forecast_timestamps) == horizon
    assert len(forecast_values) == horizon
    assert isinstance(metadata, dict)
    assert "model_name" in metadata
    assert "confidence_intervals" in metadata
    assert "metrics" in metadata

def test_predict_with_autogluon():
    """
    AutoGluonを使用した予測のテスト
    """
    predictor = TimeSeriesPredictor()

    # テストデータの作成
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    values = [10.0 + i * 0.1 for i in range(24)]

    # AutoGluon予測の実行
    try:
        horizon = 12
        forecast_timestamps, forecast_values, metadata = predictor.predict_with_autogluon(
            timestamps, values, horizon=horizon
        )

        # 結果の検証
        assert len(forecast_timestamps) == horizon
        assert len(forecast_values) == horizon
        assert isinstance(metadata, dict)
        assert "model_name" in metadata
        assert "model_type" in metadata
        assert metadata["model_type"] == "autogluon"
        assert "confidence_intervals" in metadata
        assert "metrics" in metadata
    except ImportError:
        pytest.skip("chronos-boltのAutoGluon機能が利用できません")

def test_save_load_model(tmp_path):
    """
    モデルの保存と読み込みのテスト
    """
    # 一時ディレクトリのパスを作成
    model_path = os.path.join(tmp_path, "test_model.pkl")

    # モデルの学習と保存
    predictor = TimeSeriesPredictor()
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    values = [10.0 + i * 0.1 for i in range(24)]
    predictor.fit(timestamps, values)
    predictor.save_model(model_path)

    # モデルファイルが存在することを確認
    assert os.path.exists(model_path)

    # モデルの読み込み
    loaded_predictor = TimeSeriesPredictor.load_model(model_path)
    assert loaded_predictor.model is not None

    # 読み込んだモデルで予測を実行
    horizon = 12
    forecast_timestamps, forecast_values, metadata = loaded_predictor.predict(horizon=horizon)

    # 結果の検証
    assert len(forecast_timestamps) == horizon
    assert len(forecast_values) == horizon
