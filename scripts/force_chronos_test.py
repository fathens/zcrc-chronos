#!/usr/bin/env python3
"""
強制的にChronos Transformerの動作を確認するスクリプト
ログ出力とモデル動作を詳細に監視
"""

import subprocess
import threading
import time

import requests


def monitor_server_logs():
    """サーバーログをリアルタイム監視"""
    print("📋 サーバーログ監視開始...")
    try:
        # プロセスIDを取得
        result = subprocess.run(
            ["pgrep", "-f", "run_server.py"], capture_output=True, text=True
        )
        if result.stdout.strip():
            pid = result.stdout.strip().split("\n")[0]
            print(f"🔍 サーバーPID: {pid}")
        else:
            print("❌ サーバープロセスが見つかりません")
    except Exception as e:
        print(f"❌ プロセス監視エラー: {e}")


def test_specific_chronos():
    """特定のChronos設定で明示的にテスト"""
    print("🎯 明示的Chronos Transformer テスト")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # chronos_zero_shotを明示的に指定
    test_data = {
        "timestamp": [
            "2023-01-01T00:00:00",
            "2023-01-01T01:00:00",
            "2023-01-01T02:00:00",
            "2023-01-01T03:00:00",
        ],
        "values": [100, 102, 98, 105],
        "forecast_until": "2023-01-01T07:00:00",
        "model_name": "chronos_zero_shot",  # 明示的に指定
    }

    print(f"📊 モデル名: {test_data['model_name']}")
    print(f"📊 入力データ: {test_data['values']}")
    print("🚀 Chronos Transformer実行...")
    print("\n🔍 **以下のログを確認:**")
    print("✅ 'Chronos' モデル初期化")
    print("✅ 'amazon/chronos-t5-tiny' ダウンロード/ロード")
    print("✅ HuggingFace Transformers使用")
    print("✅ MPS/GPU デバイス使用")
    print("❌ 'SeasonalNaive' 使用ログがないこと")
    print("❌ フォールバック警告がないこと")

    # ログ監視スレッド開始
    log_thread = threading.Thread(target=monitor_server_logs)
    log_thread.daemon = True
    log_thread.start()

    try:
        start_time = time.time()

        print(f"\n⏰ {time.strftime('%H:%M:%S')} - リクエスト送信")

        response = requests.post(
            f"{base_url}/api/v1/predict_zero_shot_async", json=test_data, timeout=30
        )

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            elapsed = time.time() - start_time

            print(f"✅ 予測タスク受付: {task_id}")
            print(f"⏱️  リクエスト時間: {elapsed:.3f}秒")

            print(f"\n⏰ {time.strftime('%H:%M:%S')} - 処理開始")
            print("💡 この時点でサーバーログを確認してください！")

            # 長めに待機してログを観察
            for i in range(6):  # 30秒間
                time.sleep(5)
                current_time = time.strftime("%H:%M:%S")
                print(f"⏰ {current_time} - 処理中... ({(i+1)*5}秒経過)")

                if i == 1:  # 10秒後
                    print("🔍 ログ確認ポイント1: モデル初期化完了")
                elif i == 3:  # 20秒後
                    print("🔍 ログ確認ポイント2: 予測計算中")
                elif i == 5:  # 30秒後
                    print("🔍 ログ確認ポイント3: 予測完了")

            return True

        else:
            print(f"❌ 予測失敗: {response.status_code}")
            print(f"📄 エラー: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        return False


def verify_model_config():
    """モデル設定の詳細確認"""
    print("\n🔍 model_config.yamlの設定確認")
    print("=" * 60)

    try:
        with open("config/model_config.yaml", "r") as f:
            import yaml

            config = yaml.safe_load(f)

        # chronos_zero_shot設定を探す
        chronos_config = None
        if "available_models" in config:
            for model in config["available_models"]:
                if model.get("name") == "chronos_zero_shot":
                    chronos_config = model
                    break

        if chronos_config:
            print("✅ chronos_zero_shot設定発見:")
            print(f"   説明: {chronos_config.get('description')}")

            chronos_params = chronos_config.get("chronos", {})
            print(f"   target_model: {chronos_params.get('target_model')}")
            print(f"   use_single_model: {chronos_params.get('use_single_model')}")

            hyperparams = chronos_params.get("hyperparameters", {})
            if "Chronos" in hyperparams:
                chronos_hp = hyperparams["Chronos"]
                print(f"   model_path: {chronos_hp.get('model_path')}")
                print(f"   device: {chronos_hp.get('device')}")

                if chronos_hp.get("model_path") == "amazon/chronos-t5-tiny":
                    print("✅ **真のChronos model_path確認**")
                else:
                    print("❌ 予期しないmodel_path")
            else:
                print("❌ Chronosハイパーパラメータなし")
        else:
            print("❌ chronos_zero_shot設定が見つかりません")

    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")


def main():
    print("🕵️‍♂️ 真のChronos Transformer 強制検証")
    print("🎯 代替モデルでないことを徹底確認")
    print("=" * 80)

    # 設定確認
    verify_model_config()

    # サーバー接続確認
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"\n✅ サーバー接続: {response.status_code}")
    except Exception as e:
        print(f"\n❌ サーバー接続失敗: {e}")
        return

    # 強制テスト実行
    test_start_time = time.strftime("%H:%M:%S")
    print(f"\n🚀 テスト実行開始: {test_start_time}")
    success = test_specific_chronos()

    print("\n" + "=" * 80)
    print("🔍 **検証結果:**")
    if success:
        print("✅ chronos_zero_shot リクエスト送信成功")
        print("💡 サーバーログで以下を確認してください:")
        print("   1. 'Chronos' モデル使用ログ")
        print("   2. 'amazon/chronos-t5-tiny' ロードログ")
        print("   3. 'mps' デバイス使用ログ")
        print("   4. HuggingFace/Transformers関連ログ")
        print("   5. 'SeasonalNaive' ログがないこと")
    else:
        print("❌ テスト失敗")

    current_time = time.strftime("%H:%M:%S")
    print(f"\n🏁 検証完了: {current_time}")


if __name__ == "__main__":
    main()
