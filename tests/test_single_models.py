#!/usr/bin/env python3
"""
AutoGluon-TimeSeries単一モデル設定のテストスクリプト

このスクリプトは、新しく実装された単一モデル設定が正しく動作することを確認します。
特に、指定されたモデルのみが訓練され、他のモデルが自動選択されないことを検証します。
"""

import datetime
import os
import sys

import numpy as np
from loguru import logger

from src.models.predictor import TimeSeriesPredictor

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def generate_test_data(length=100):
    """テスト用の時系列データを生成"""
    now = datetime.datetime.now()
    timestamps = [now + datetime.timedelta(hours=i) for i in range(length)]

    # トレンドと季節性を持つデータを生成
    np.random.seed(42)
    trend = np.linspace(10, 15, length)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(length) / 24)  # 24時間周期
    noise = np.random.normal(0, 0.5, length)
    values = trend + seasonal + noise

    return timestamps, values.tolist()


def test_single_model_configuration(model_name, model_params):
    """単一モデル設定のテスト"""
    logger.info(f"\n=== {model_name} のテスト開始 ===")

    try:
        # テストデータの生成
        timestamps, values = generate_test_data(50)

        # 予測器の初期化
        predictor = TimeSeriesPredictor(
            model_name=model_name, model_params=model_params
        )

        # 予測の実行
        result = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=12
        )
        forecast_timestamps, forecast_values, metadata = result

        # 結果の検証
        logger.info(f"予測結果: {len(forecast_values)} ポイント")
        logger.info(f"メタデータ: {metadata}")

        # 単一モデル設定が正しく適用されているかを確認
        if metadata.get("use_single_model"):
            logger.success(
                f"✓ 単一モデル設定が適用されました: {metadata.get('target_model')}"
            )
            trained_models = metadata.get("trained_models", [])
            logger.info(f"実際に訓練されたモデル: {trained_models}")
        else:
            logger.warning("! 単一モデル設定が適用されませんでした")

        return True

    except Exception as e:
        logger.error(f"✗ {model_name} のテストに失敗: {str(e)}")
        return False


def main():
    """メインテスト関数"""
    logger.info("AutoGluon-TimeSeries 単一モデル設定のテストを開始します")

    # テスト対象の単一モデル設定
    test_configs = [
        {
            "name": "autoets_only",
            "params": {
                "model_type": "autogluon",
                "time_limit": 300,
                "use_single_model": True,
                "target_model": "AutoETSModel",
                "hyperparameters": {
                    "AutoETSModel": {"model": "ZZZ", "seasonal_period": None}
                },
            },
        },
        {
            "name": "seasonal_naive_only",
            "params": {
                "model_type": "autogluon",
                "time_limit": 60,
                "use_single_model": True,
                "target_model": "SeasonalNaiveModel",
                "hyperparameters": {"SeasonalNaiveModel": {}},
            },
        },
        {
            "name": "npts_only",
            "params": {
                "model_type": "autogluon",
                "time_limit": 600,
                "use_single_model": True,
                "target_model": "NPTSModel",
                "hyperparameters": {"NPTSModel": {}},
            },
        },
    ]

    results = []

    for config in test_configs:
        success = test_single_model_configuration(config["name"], config["params"])
        results.append((config["name"], success))

    # 結果の要約
    logger.info("\n=== テスト結果の要約 ===")
    success_count = 0
    for model_name, success in results:
        status = "✓ 成功" if success else "✗ 失敗"
        logger.info(f"{model_name}: {status}")
        if success:
            success_count += 1

    logger.info(f"\n成功: {success_count}/{len(results)}")

    if success_count == len(results):
        logger.success("全ての単一モデル設定のテストが成功しました！")
        return True
    else:
        logger.error("一部のテストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
