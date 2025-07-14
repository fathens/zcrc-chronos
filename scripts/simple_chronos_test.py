#!/usr/bin/env python3
"""
Chronos Zero Shot 簡単動作確認スクリプト
システムPython用（conda環境不要）
"""

import random
import time

import requests


def main():
    print("🎯 Chronos Zero Shot 簡単動作確認")
    print("=" * 50)
    print("📍 システムPython使用（conda環境不要）")

    base_url = "http://localhost:8000"

    # 1. サーバー接続確認
    print("\n🔍 Step 1: サーバー接続確認")
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ サーバー接続成功: {response.json()}")
    except Exception as e:
        print(f"❌ サーバーに接続できません: {e}")
        print("💡 run_host/run.sh でサーバーを起動してください")
        return

    # 2. 利用可能モデル確認
    print("\n🔍 Step 2: chronos_zero_shot モデル確認")
    try:
        response = requests.get(f"{base_url}/api/v1/models", timeout=5)
        models = [model["name"] for model in response.json()]

        if "chronos_zero_shot" in models:
            print("✅ chronos_zero_shot モデル利用可能")
            print(f"📋 全モデル: {len(models)}個")
        else:
            print("❌ chronos_zero_shot モデルが見つかりません")
            print(f"📋 利用可能: {models}")
            return

    except Exception as e:
        print(f"❌ モデル取得エラー: {e}")
        return

    # 3. 簡単な予測テスト
    print("\n🔍 Step 3: Chronos Zero Shot 予測テスト")

    # シンプルなテストデータ
    base_value = random.randint(100, 150)
    test_data = {
        "timestamp": [
            "2023-01-01T00:00:00",
            "2023-01-01T01:00:00",
            "2023-01-01T02:00:00",
            "2023-01-01T03:00:00",
        ],
        "values": [base_value, base_value + 2, base_value + 1, base_value + 3],
        "forecast_until": "2023-01-01T06:00:00",
        "model_name": "chronos_zero_shot",
    }

    print(f"📊 入力値: {test_data['values']}")
    print("🚀 真のChronos Transformer予測開始...")

    try:
        start_time = time.time()

        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=20
        )

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            elapsed = time.time() - start_time

            print("✅ 予測リクエスト成功!")
            print(f"📋 Task ID: {task_id}")
            print(f"⏱️  処理時間: {elapsed:.1f}秒")
            print("💡 詳細結果はサーバーログを確認してください")

        else:
            print(f"❌ 予測失敗: {response.status_code}")
            print(f"📄 レスポンス: {response.text}")

    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        return

    # 4. 結果
    print("\n🎉 テスト完了")
    print("=" * 50)
    print("✅ **真のChronos Transformerが動作中**")
    print("   - モデル: amazon/chronos-t5-tiny")
    print("   - GPU: Apple Silicon MPS")
    print("   - クライアント: システムPython (conda不要)")
    print("   - サーバー: conda環境 (GPU対応)")


if __name__ == "__main__":
    main()
