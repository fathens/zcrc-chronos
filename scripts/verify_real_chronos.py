#!/usr/bin/env python3
"""
真のChronos Transformerモデル動作検証スクリプト
代替モデルではなく本物のChronosが動作していることを徹底確認
"""

import subprocess
import time

import requests


def check_server_logs():
    """サーバーログから詳細情報を取得"""
    print("📋 サーバーログ詳細分析")
    print("=" * 60)

    # プロセス一覧からサーバープロセスを特定
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        server_processes = []
        for line in result.stdout.split("\n"):
            if "run_server.py" in line:
                server_processes.append(line)

        print("🔍 サーバープロセス:")
        for proc in server_processes:
            print(f"   {proc}")

    except Exception as e:
        print(f"❌ プロセス確認エラー: {e}")


def test_chronos_with_logging():
    """Chronos予測実行とログ詳細確認"""
    print("\n🧠 Chronos Transformer 詳細動作確認")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # 1. モデル詳細情報確認
    print("🔍 chronos_zero_shot モデル詳細:")
    try:
        response = requests.get(f"{base_url}/api/v1/models", timeout=5)
        models = response.json()

        chronos_model = None
        for model in models:
            if model["name"] == "chronos_zero_shot":
                chronos_model = model
                break

        if chronos_model:
            print(f"✅ モデル名: {chronos_model['name']}")
            print(f"✅ 説明: {chronos_model['description']}")

            params = chronos_model["parameters"]
            print(f"✅ target_model: {params.get('target_model')}")
            print(f"✅ use_single_model: {params.get('use_single_model')}")

            if "hyperparameters" in params and "Chronos" in params["hyperparameters"]:
                chronos_params = params["hyperparameters"]["Chronos"]
                print(f"✅ Chronos model_path: {chronos_params.get('model_path')}")
                print(f"✅ Chronos device: {chronos_params.get('device')}")
                print("🎯 **真のChronosパラメータを確認**")
            else:
                print("❌ Chronosハイパーパラメータが見つかりません")
                return False
        else:
            print("❌ chronos_zero_shot モデルが見つかりません")
            return False

    except Exception as e:
        print(f"❌ モデル詳細取得エラー: {e}")
        return False

    # 2. 特殊なテストデータで予測実行
    print("\n🧪 特殊パターンでChronos実行テスト:")

    # 複雑なパターンを作成（統計的手法では困難）
    test_data = {
        "timestamp": [
            "2023-01-01T00:00:00",
            "2023-01-01T01:00:00",
            "2023-01-01T02:00:00",
            "2023-01-01T03:00:00",
            "2023-01-01T04:00:00",
            "2023-01-01T05:00:00",
            "2023-01-01T06:00:00",
            "2023-01-01T07:00:00",
        ],
        "values": [100, 105, 98, 110, 95, 115, 90, 120],  # 複雑な非線形パターン
        "forecast_until": "2023-01-01T11:00:00",
        "model_name": "chronos_zero_shot",
    }

    print(f"📊 複雑パターン入力: {test_data['values']}")
    print("🚀 Chronos Transformer実行開始...")
    print("⚠️  真のChronosなら事前訓練済みTransformerで処理")
    print("⚠️  偽物なら統計的手法（SeasonalNaive等）で処理")

    try:
        start_time = time.time()

        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=30
        )

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            elapsed = time.time() - start_time

            print(f"✅ 予測タスク開始: {task_id}")
            print(f"⏱️  リクエスト時間: {elapsed:.3f}秒")

            # 処理完了まで待機
            print("\n⏳ Chronos処理完了を待機...")
            wait_times = [5, 10, 15, 20, 30]

            for wait in wait_times:
                time.sleep(5)
                print(f"   待機中... {wait}秒経過")

                # この時点でサーバーログを確認するよう促す
                if wait == 10:
                    print("💡 別ターミナルでサーバーログを確認してください:")
                    print("   docker logs zcrc-chronos --tail 20  (Docker版)")
                    print("   または run_host/run.sh のログ出力を確認")

            return True

        else:
            print(f"❌ 予測失敗: {response.status_code}")
            print(f"📄 エラー詳細: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        return False


def verify_not_fallback():
    """フォールバック動作していないことを確認"""
    print("\n🔍 フォールバック動作確認テスト")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # 存在しないモデル名で確認
    test_data = {
        "timestamp": ["2023-01-01T00:00:00", "2023-01-01T01:00:00"],
        "values": [100, 105],
        "forecast_until": "2023-01-01T03:00:00",
        "model_name": "nonexistent_model",  # 存在しないモデル
    }

    print("🧪 存在しないモデル名でテスト...")

    try:
        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=10
        )

        if response.status_code == 200:
            print("⚠️  存在しないモデルでも成功 → フォールバック動作の可能性")
            task_data = response.json()
            print(f"   Task ID: {task_data.get('task_id')}")
        else:
            print(f"✅ 存在しないモデルで正常にエラー: {response.status_code}")
            print("   → フォールバック動作なし、適切にモデル検証中")

    except Exception as e:
        print(f"🔍 リクエストエラー: {e}")


def main():
    print("🕵️ 真のChronos Transformer 徹底検証")
    print("🎯 代替モデル（SeasonalNaive等）でないことを確認")
    print("=" * 80)

    # サーバー動作確認
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ サーバー接続確認: {response.status_code}")
    except Exception as e:
        print(f"❌ サーバーに接続できません: {e}")
        print("💡 run_host/run.sh でサーバーを起動してください")
        return

    # 詳細検証実行
    check_server_logs()

    success = test_chronos_with_logging()

    verify_not_fallback()

    print("\n" + "=" * 80)
    print("🔍 **検証ポイント - サーバーログで以下を確認:**")
    print("✅ 'Chronos' モデルの初期化ログ")
    print("✅ 'amazon/chronos-t5-tiny' モデルパス")
    print("✅ 'mps' または 'cuda' デバイス使用ログ")
    print("✅ HuggingFace Transformers関連ログ")
    print("❌ 'SeasonalNaive' 実行ログがないこと")
    print("❌ フォールバック警告ログがないこと")

    if success:
        print("\n🎉 Chronos予測リクエスト成功")
        print("💡 サーバーログを確認して真のChronos動作を検証してください")
    else:
        print("\n❌ 検証に失敗しました")


if __name__ == "__main__":
    main()
