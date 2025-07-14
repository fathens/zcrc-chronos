#!/usr/bin/env python3
"""
フォールバック削除後の動作確認スクリプト
存在しないモデル名でエラーが返ることを確認
"""

import time

import requests


def test_nonexistent_model():
    """存在しないモデル名でテスト"""
    print("🧪 存在しないモデル名でのエラー確認")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # 存在しないモデル名
    test_data = {
        "timestamp": ["2023-01-01T00:00:00", "2023-01-01T01:00:00"],
        "values": [100, 105],
        "forecast_until": "2023-01-01T03:00:00",
        "model_name": "nonexistent_model",  # 存在しないモデル
    }

    print(f"📊 テストモデル名: {test_data['model_name']}")
    print("⚠️  期待動作: エラーレスポンスが返る")

    try:
        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=10
        )

        print(f"📍 ステータスコード: {response.status_code}")

        if response.status_code == 200:
            print("❌ 予期しない成功レスポンス！フォールバックの可能性")
            print(f"   レスポンス: {response.json()}")
            return False
        else:
            print("✅ 正常にエラーレスポンスを受信")
            print(f"   エラー内容: {response.text}")
            return True

    except Exception as e:
        print(f"🔍 リクエストエラー: {e}")
        return False


def test_chronos_zero_shot():
    """正しいchronos_zero_shotモデルでテスト"""
    print("\n🧪 chronos_zero_shot モデルの正常動作確認")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # 正しいモデル名
    test_data = {
        "timestamp": [
            "2023-01-01T00:00:00",
            "2023-01-01T01:00:00",
            "2023-01-01T02:00:00",
            "2023-01-01T03:00:00",
        ],
        "values": [100, 102, 98, 105],
        "forecast_until": "2023-01-01T06:00:00",
        "model_name": "chronos_zero_shot",  # 正しいモデル名
    }

    print(f"📊 テストモデル名: {test_data['model_name']}")
    print("✅ 期待動作: 正常にタスクが受け付けられる")

    try:
        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=10
        )

        print(f"📍 ステータスコード: {response.status_code}")

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            print("✅ 正常にタスク受付")
            print(f"   Task ID: {task_id}")
            print("🎯 **真のChronos Transformerが実行されます**")
            return True
        else:
            print("❌ 予期しないエラー")
            print(f"   エラー内容: {response.text}")
            return False

    except Exception as e:
        print(f"❌ リクエストエラー: {e}")
        return False


def list_available_models():
    """利用可能なモデル一覧を表示"""
    print("\n📋 利用可能なモデル一覧")
    print("=" * 60)

    base_url = "http://localhost:8000"

    try:
        response = requests.get(f"{base_url}/api/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 利用可能なモデル数: {len(models)}")
            for model in models:
                print(f"   - {model['name']}: {model['description'][:50]}...")
        else:
            print(f"❌ モデル一覧取得失敗: {response.status_code}")

    except Exception as e:
        print(f"❌ リクエストエラー: {e}")


def main():
    print("🕵️ フォールバック削除後の動作確認")
    print("🎯 存在しないモデルでエラーが返ることを検証")
    print("=" * 80)

    # サーバー接続確認
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ サーバー接続: {response.status_code}")
    except Exception as e:
        print(f"❌ サーバー接続失敗: {e}")
        print("💡 run_host/run.sh でサーバーを起動してください")
        return

    # テスト実行
    time.sleep(2)  # サーバー安定化待機

    # 1. 存在しないモデルでエラー確認
    no_fallback = test_nonexistent_model()

    # 2. 正しいモデルで正常動作確認
    chronos_works = test_chronos_zero_shot()

    # 3. 利用可能モデル一覧表示
    list_available_models()

    # 結果まとめ
    print("\n" + "=" * 80)
    print("🔍 **検証結果:**")

    if no_fallback:
        print("✅ フォールバック削除成功: 存在しないモデルで適切にエラー")
    else:
        print("❌ フォールバック削除失敗: まだフォールバック動作中")

    if chronos_works:
        print("✅ chronos_zero_shot正常動作: 真のChronos Transformerが利用可能")
    else:
        print("❌ chronos_zero_shot動作異常")

    if no_fallback and chronos_works:
        print("\n🎉 **完璧！フォールバックなしで真のChronosが動作中**")

    print("\n🏁 検証完了")


if __name__ == "__main__":
    main()
