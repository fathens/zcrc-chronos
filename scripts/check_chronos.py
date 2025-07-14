#!/usr/bin/env python3
"""
Chronosモデルの設定とCPU実行可能性を調査するスクリプト
"""

import inspect

from autogluon.timeseries.models.chronos import ChronosModel


def main():
    print("🔍 Chronos model investigation")
    print("=" * 50)

    # パラメータ調査
    print("\n📋 Chronos model __init__ parameters:")
    sig = inspect.signature(ChronosModel.__init__)
    for param_name, param in sig.parameters.items():
        if param_name != "self":
            print(f"  - {param_name}: {param.default}")

    # 属性調査
    print("\n📋 Chronos model attributes:")
    attrs = [attr for attr in dir(ChronosModel) if not attr.startswith("_")]
    for attr in attrs[:15]:  # 最初の15個
        print(f"  - {attr}")

    # デバイス関連設定調査
    print("\n🔍 Searching for device/GPU related settings:")
    try:
        source = inspect.getsource(ChronosModel)
        keywords = ["device", "cpu", "cuda", "gpu", "torch"]
        found_keywords = []

        for keyword in keywords:
            if keyword.lower() in source.lower():
                found_keywords.append(keyword)

        if found_keywords:
            print(f"  ✅ Found keywords: {', '.join(found_keywords)}")
        else:
            print("  ❌ No device-related keywords found")

    except Exception as e:
        print(f"  ❌ Error getting source: {e}")

    # 利用可能なハイパーパラメータ調査
    print("\n📋 Chronos hyperparameters investigation:")
    try:
        from autogluon.timeseries import TimeSeriesPredictor

        # 空のpredictorを作成してhyperparametersヘルプを取得
        predictor = TimeSeriesPredictor(prediction_length=1, target="target")

        # Chronosのデフォルト設定を確認
        chronos_defaults = predictor._get_model_defaults().get("Chronos", {})
        print(f"  Default Chronos config: {chronos_defaults}")

    except Exception as e:
        print(f"  ❌ Error getting hyperparameters: {e}")

    # 直接Chronosインスタンス作成テスト
    print("\n🧪 Testing Chronos instantiation:")
    try:
        # 最小限のパラメータでChronosを作成
        model = ChronosModel(
            freq="H",
            prediction_length=1,
            quantile_levels=[0.5],
            model_path="amazon/chronos-t5-tiny",  # より小さなモデルを試す
        )
        print("  ✅ ChronosModel instantiation successful")

        # デバイス設定があるか確認
        if hasattr(model, "device"):
            print(f"  📱 Model device: {model.device}")
        if hasattr(model, "_device"):
            print(f"  📱 Model _device: {model._device}")

    except Exception as e:
        print(f"  ❌ ChronosModel instantiation failed: {e}")


if __name__ == "__main__":
    main()
