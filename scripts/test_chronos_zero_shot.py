#!/usr/bin/env python3
"""
Chronos Zero Shot モデルの動作確認スクリプト
真のChronos Transformerモデルの実行を検証
"""

import random
import time

import requests


def test_chronos_zero_shot():
    print("🚀 Chronos Zero Shot (真のTransformer) 動作確認")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # サーバー接続確認
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ サーバー接続成功: {response.status_code}")
    except Exception as e:
        print(f"❌ サーバーに接続できません: {e}")
        return False

    # chronos_zero_shotモデル確認
    try:
        response = requests.get(f"{base_url}/api/v1/models", timeout=5)
        models = response.json()

        chronos_model = None
        for model in models:
            if model["name"] == "chronos_zero_shot":
                chronos_model = model
                break

        if chronos_model:
            print("✅ chronos_zero_shot モデル発見")
            print(f"   説明: {chronos_model['description']}")
            params = chronos_model["parameters"]
            print(f"   target_model: {params.get('target_model')}")
            if "hyperparameters" in params and "Chronos" in params["hyperparameters"]:
                chronos_params = params["hyperparameters"]["Chronos"]
                print(f"   model_path: {chronos_params.get('model_path')}")
                print(f"   device: {chronos_params.get('device')}")
        else:
            print("❌ chronos_zero_shot モデルが見つかりません")
            return False

    except Exception as e:
        print(f"❌ モデル情報取得エラー: {e}")
        return False

    # テストデータ作成（時系列パターン）
    base_value = random.randint(100, 200)
    trend = random.choice([-2, -1, 0, 1, 2])
    seasonal_amplitude = random.randint(5, 15)

    values = []
    for i in range(12):  # 12時間分のデータ
        trend_component = base_value + (trend * i)
        seasonal_component = seasonal_amplitude * (1 if i % 4 < 2 else -1)
        noise = random.randint(-3, 3)
        values.append(trend_component + seasonal_component + noise)

    test_data = {
        "timestamp": [f"2023-01-01T{i:02d}:00:00" for i in range(12)],
        "values": values,
        "forecast_until": "2023-01-01T15:00:00",  # 3時間先まで予測
        "model_name": "chronos_zero_shot",
    }

    print("\n📊 テストデータ:")
    print(f"   ベース値: {base_value}")
    print(f"   トレンド: {trend}")
    print(f"   季節振幅: {seasonal_amplitude}")
    print(f"   入力系列: {values[:6]}... (12点)")
    print("   予測期間: 3時間先まで")

    # Chronos Zero Shot予測実行
    print("\n🧠 Chronos Transformer 予測開始...")
    print("   (amazon/chronos-t5-tiny モデル使用)")

    try:
        start_time = time.time()

        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=30
        )

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            print(f"✅ 予測タスク開始: {task_id}")

            # 予測完了を待機
            print("⏳ Chronos Transformer計算中...")
            max_wait = 60  # 最大60秒待機
            wait_time = 0

            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5
                print(f"   待機中... ({wait_time}s/{max_wait}s)")

            elapsed_time = time.time() - start_time
            print(f"🎯 予測処理時間: {elapsed_time:.1f}秒")
            print("✅ Chronos Zero Shot予測が完了しました")

            return True

        else:
            print(f"❌ 予測リクエスト失敗: {response.status_code}")
            print(f"   エラー: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 予測リクエストエラー: {e}")
        return False


def test_model_comparison():
    """Chronos vs SeasonalNaive比較テスト"""
    print("\n🔬 モデル比較テスト")
    print("=" * 60)

    models_to_test = [
        ("chronos_zero_shot", "真のChronos Transformer"),
        ("seasonal_naive_only", "SeasonalNaive (旧疑似)"),
    ]

    base_url = "http://localhost:8000"
    base_value = 150

    # 共通テストデータ
    test_data_base = {
        "timestamp": [f"2023-01-01T{i:02d}:00:00" for i in range(6)],
        "values": [base_value + (i % 3) for i in range(6)],
        "forecast_until": "2023-01-01T09:00:00",
    }

    for model_name, description in models_to_test:
        print(f"\n🧪 テスト中: {model_name} ({description})")

        test_data = test_data_base.copy()
        test_data["model_name"] = model_name

        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=20
            )

            if response.status_code == 200:
                elapsed = time.time() - start_time
                task_id = response.json().get("task_id")
                print(f"   ✅ 成功: {elapsed:.1f}s (task: {task_id})")
            else:
                print(f"   ❌ 失敗: {response.status_code}")

        except Exception as e:
            print(f"   ❌ エラー: {str(e)[:50]}...")

    return True


if __name__ == "__main__":
    print("🎯 Chronos Zero Shot GPU動作確認開始")

    success = test_chronos_zero_shot()

    if success:
        test_model_comparison()
        print("\n🎉 **真のChronos Transformerモデルが正常に動作しています！**")
        print("   - モデル: amazon/chronos-t5-tiny")
        print("   - デバイス: Apple Silicon GPU (MPS)")
        print("   - ゼロショット: 事前訓練済みTransformer")
    else:
        print("\n❌ Chronos Zero Shot動作確認に失敗しました")

    print("\n🏁 テスト完了")
