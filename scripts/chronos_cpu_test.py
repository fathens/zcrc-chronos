#!/usr/bin/env python3
"""
CPU環境でのChronos実行テストスクリプト
"""

import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor


def test_chronos_cpu():
    print("🧪 Testing Chronos with CPU-friendly settings")
    print("=" * 50)

    # テストデータ作成
    train_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=20, freq="h"),
            "target": [100 + i + (i % 3) for i in range(20)],
            "item_id": ["test"] * 20,
        }
    )

    print(f"📊 Created test data: {len(train_data)} rows")

    # CPU対応Chronos設定の試行パターン
    chronos_configs = [
        {"name": "basic_chronos", "hyperparameters": {"Chronos": {}}},
        {
            "name": "chronos_with_device",
            "hyperparameters": {"Chronos": {"device": "cpu"}},
        },
        {
            "name": "chronos_tiny_model",
            "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-tiny"}},
        },
        {
            "name": "chronos_mini_model",
            "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-mini"}},
        },
        {
            "name": "chronos_small_model",
            "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-small"}},
        },
    ]

    for config in chronos_configs:
        print(f"\n🔍 Testing: {config['name']}")
        try:
            predictor = TimeSeriesPredictor(
                prediction_length=3, target="target", eval_metric="MAPE", verbosity=0
            )

            # 単一Chronosモデルでfit試行
            predictor.fit(
                train_data=train_data,
                hyperparameters=config["hyperparameters"],
                time_limit=60,  # 短時間制限
                presets=None,
                num_cpus=1,
            )

            print(f"  ✅ {config['name']}: Fit successful!")

            # 予測テスト
            predictions = predictor.predict(train_data.tail(1))
            print(f"  ✅ {config['name']}: Predict successful!")
            print(f"  📈 Prediction shape: {predictions.shape}")

            break  # 成功したら終了

        except Exception as e:
            error_msg = str(e)
            if "GPU" in error_msg or "CUDA" in error_msg:
                print(f"  ❌ {config['name']}: GPU requirement - {error_msg[:100]}...")
            elif "model_path" in error_msg:
                print(f"  ❌ {config['name']}: Model path issue - {error_msg[:100]}...")
            else:
                print(f"  ❌ {config['name']}: Other error - {error_msg[:100]}...")


def test_alternative_zero_shot():
    print("\n\n🔄 Testing alternative zero-shot approaches")
    print("=" * 50)

    # テストデータ
    train_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=20, freq="h"),
            "target": [100 + i + (i % 3) for i in range(20)],
            "item_id": ["test"] * 20,
        }
    )

    # 代替Zero-Shot風モデル（軽量で事前訓練済み風）
    alternatives = [
        {
            "name": "SeasonalNaive (Zero-shot like)",
            "hyperparameters": {"SeasonalNaive": {}},
        },
        {"name": "Naive (Simple zero-shot)", "hyperparameters": {"Naive": {}}},
        {"name": "Average (Zero-shot baseline)", "hyperparameters": {"Average": {}}},
    ]

    for alt in alternatives:
        print(f"\n🔍 Testing: {alt['name']}")
        try:
            predictor = TimeSeriesPredictor(
                prediction_length=3, target="target", verbosity=0
            )

            predictor.fit(
                train_data=train_data,
                hyperparameters=alt["hyperparameters"],
                time_limit=30,
            )

            predictions = predictor.predict(train_data.tail(1))
            print(f"  ✅ {alt['name']}: Success! Shape: {predictions.shape}")

        except Exception as e:
            print(f"  ❌ {alt['name']}: Failed - {str(e)[:100]}...")


if __name__ == "__main__":
    test_chronos_cpu()
    test_alternative_zero_shot()
