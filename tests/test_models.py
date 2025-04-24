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

def test_zero_shot_predict():
    """
    Chronos-Bolt を使用したゼロショット予測のテスト
    """
    predictor = TimeSeriesPredictor()

    # コンテキスト情報
    context = "過去24時間の電力消費量データに基づいて、今後12時間の予測を行う"

    # ゼロショット予測の実行
    try:
        horizon = 12
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            context=context, horizon=horizon
        )

        # 結果の検証
        assert len(forecast_timestamps) == horizon
        assert len(forecast_values) == horizon
        assert isinstance(metadata, dict)
        assert "model_name" in metadata
        assert "model_type" in metadata
        assert metadata["model_type"] == "chronos_bolt"
        assert "preset" in metadata
        assert "context" in metadata
        assert "confidence_intervals" in metadata
    except ImportError:
        pytest.skip("AutoGluon-TimeSeriesのChronos-Bolt機能が利用できません")

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

    # 読み込んだモデルの検証（予測メソッドの呼び出しはしない）
    assert loaded_predictor.model_name == predictor.model_name
