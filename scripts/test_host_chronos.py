#!/usr/bin/env python3
"""
ホスト環境でのChronos GPU動作確認スクリプト
"""

import random
import time

import requests


def test_chronos_zero_shot():
    print("🧪 Chronos Zero Shot GPU動作確認")
    print("=" * 50)

    base_url = "http://localhost:8000"

    # ヘルスチェック
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ ヘルスチェック: {response.status_code}")
    except Exception as e:
        print(f"❌ サーバーに接続できません: {e}")
        return

    # 利用可能モデル確認
    try:
        response = requests.get(f"{base_url}/api/v1/models", timeout=5)
        models = response.json()
        print(f"📋 利用可能モデル数: {len(models)}")
        if "chronos_zero_shot" in models:
            print("✅ chronos_zero_shot モデルが利用可能")
        else:
            print("❌ chronos_zero_shot モデルが見つかりません")
            print(f"   利用可能: {models}")
    except Exception as e:
        print(f"❌ モデル一覧取得エラー: {e}")
        return

    # テストデータ作成
    base_value = random.randint(100, 200)
    test_data = {
        "timestamp": [
            "2023-01-01T00:00:00",
            "2023-01-01T01:00:00",
            "2023-01-01T02:00:00",
            "2023-01-01T03:00:00",
            "2023-01-01T04:00:00",
            "2023-01-01T05:00:00",
        ],
        "values": [
            base_value,
            base_value + 1,
            base_value - 1,
            base_value + 2,
            base_value + 1,
            base_value + 3,
        ],
        "forecast_until": "2023-01-01T08:00:00",
        "model_name": "chronos_zero_shot",
    }

    print(f"\n🔍 テストデータ (base: {base_value}):")
    print(f"   入力値: {test_data['values']}")

    # 非同期予測リクエスト
    print("\n🚀 Chronos Zero Shot 予測開始...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=10
        )

        if response.status_code == 200:
            task_id = response.json().get("task_id")
            print(f"✅ 予測タスク開始: {task_id}")

            # 結果待機
            print("⏳ 予測完了を待機中...")
            time.sleep(10)

            # 結果確認 (ここではログで確認)
            print("📊 予測完了 (詳細はサーバーログを確認)")

        else:
            print(f"❌ 予測リクエスト失敗: {response.status_code}")
            print(f"   エラー: {response.text}")

    except Exception as e:
        print(f"❌ 予測リクエストエラー: {e}")


def test_device_usage():
    print("\n🎯 GPU (MPS) 使用状況確認")
    print("=" * 50)

    import torch

    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"MPS 利用可能: {torch.backends.mps.is_available()}")
    print(f"MPS ビルド済み: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        # MPS デバイステスト
        try:
            test_tensor = torch.randn(1000, 1000, device="mps")
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✅ MPS テンソル演算成功: {result.device}")
            print(f"   テンソルサイズ: {result.shape}")
        except Exception as e:
            print(f"❌ MPS テンソル演算失敗: {e}")
    else:
        print("⚠️  MPS が利用できません")


if __name__ == "__main__":
    test_device_usage()
    test_chronos_zero_shot()
    print("\n🏁 テスト完了")
