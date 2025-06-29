#!/usr/bin/env python3
"""
本番環境での直線的予測問題をデバッグするスクリプト
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def debug_current_implementation():
    """
    現在の実装状況をデバッグ
    """
    print("=== 現在の実装状況デバッグ ===")

    try:
        from src.api.routes import normalize_time_series_data
        from src.models.predictor import TimeSeriesPredictor

        # 実際のスクリーンショットに近いデータパターンを作成
        base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)

        # akaiaのようなパターン（上昇→ピーク→下降）
        timestamps = []
        values = []

        # 実際のパターンを模擬
        price_pattern = [
            39000,
            40000,
            39500,
            37000,
            37500,
            38000,
            39000,
            39500,  # 初期変動
            40000,
            41000,
            41500,
            42000,
            42500,
            43000,
            43200,  # 上昇
            42800,
            42000,
            41000,
            40000,
            38000,
            36500,
            36000,
            36200,  # 下降
        ]

        for i, price in enumerate(price_pattern):
            timestamps.append(base_time + datetime.timedelta(hours=i * 6))  # 6時間間隔
            values.append(price)

        print("テストデータ:")
        print(f"  データポイント数: {len(timestamps)}")
        print(f"  時間範囲: {(timestamps[-1] - timestamps[0]).days}日")
        print(f"  価格範囲: {min(values):.0f} - {max(values):.0f}")
        print(f"  最初の5値: {values[:5]}")
        print(f"  最後の5値: {values[-5:]}")

        # STEP 1: 正規化処理の確認
        print("\n--- STEP 1: 正規化処理 ---")
        norm_timestamps, norm_values = normalize_time_series_data(timestamps, values)

        print("正規化結果:")
        print(f"  入力: {len(timestamps)} → 出力: {len(norm_values)}")
        print(f"  データ増加率: {len(norm_values) / len(timestamps):.2f}x")

        # 変動保持の確認
        orig_range = max(values) - min(values)
        norm_range = max(norm_values) - min(norm_values)
        print(
            f"  変動保持: {orig_range:.0f} → {norm_range:.0f} "
            f"({norm_range / orig_range:.2%})"
        )

        # STEP 2: 予測処理の確認
        print("\n--- STEP 2: 予測処理 ---")
        predictor = TimeSeriesPredictor()

        # 24時間予測をテスト
        print("24時間予測を実行中...")
        try:
            pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
                timestamp=norm_timestamps, values=norm_values, horizon=24
            )

            print("予測結果:")
            print("  要求期間: 24時間")
            print(f"  実際期間: {len(pred_values)}時間")
            print(f"  予測値の範囲: {min(pred_values):.0f} - {max(pred_values):.0f}")
            print(f"  予測値サンプル: {[int(v) for v in pred_values[:5]]}")

            # 直線性のチェック
            if len(pred_values) > 1:
                differences = [
                    abs(pred_values[i + 1] - pred_values[i])
                    for i in range(len(pred_values) - 1)
                ]
                max_diff = max(differences)
                avg_diff = sum(differences) / len(differences)

                print(f"  最大変化: {max_diff:.2f}")
                print(f"  平均変化: {avg_diff:.2f}")

                if max_diff < 10:
                    print("  ❌ 予測が直線的（ほぼ変化なし）")
                elif avg_diff < 50:
                    print("  ⚠️  予測の変動が少ない")
                else:
                    print("  ✅ 予測に適切な変動がある")

            # メタデータの確認
            print("\nメタデータ:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

            return pred_values

        except Exception as e:
            print(f"予測エラー: {e}")
            import traceback

            traceback.print_exc()
            return None

    except ImportError as e:
        print(f"インポートエラー: {e}")
        print("環境が正しくセットアップされていない可能性があります")
        return None
    except Exception as e:
        print(f"全般エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def check_model_exclusion():
    """
    Naiveモデル除外が機能しているかチェック
    """
    print(f"\n{'=' * 50}")
    print("=== Naiveモデル除外チェック ===")

    try:
        # 小さなデータセットでテスト（Naiveが選ばれやすい条件）
        base_time = datetime.datetime(2025, 6, 14, 0, 0, 0)
        timestamps = [base_time + datetime.timedelta(hours=i) for i in range(10)]
        values = [36000 + i * 100 for i in range(10)]  # 単純な上昇トレンド

        from src.models.predictor import TimeSeriesPredictor

        predictor = TimeSeriesPredictor()

        print(f"小さなデータセット（{len(timestamps)}ポイント）でテスト")
        pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=6
        )

        print(f"使用されたモデル: {metadata.get('model_type', 'Unknown')}")

        # 直線性の強いチェック
        if len(pred_values) > 1:
            all_same = all(
                abs(pred_values[i] - pred_values[0]) < 1
                for i in range(len(pred_values))
            )
            if all_same:
                print("❌ Naiveモデルが使用された可能性（すべて同じ値）")
            else:
                print("✅ Naiveモデルは回避されている")

    except Exception as e:
        print(f"チェックエラー: {e}")


def verify_code_changes():
    """
    コードの変更が実際に反映されているか確認
    """
    print(f"\n{'=' * 50}")
    print("=== コード変更確認 ===")

    try:
        # predictorモジュールのインポートと確認
        import inspect

        from src.models.predictor import TimeSeriesPredictor

        # zero_shot_predict関数のソースコードから改善が含まれているかチェック
        source = inspect.getsource(TimeSeriesPredictor.zero_shot_predict)

        checks = {
            "Naiveモデル除外": "excluded_model_types" in source and "Naive" in source,
            "予測期間調整": "horizon <= 6" in source or "horizon <= 12" in source,
            "動的モデル選択": "retry_hyperparameters" in source,
            "検証設定調整": "num_val_windows" in source,
        }

        print("実装確認:")
        for check_name, is_present in checks.items():
            status = "✅" if is_present else "❌"
            print(f"  {status} {check_name}: {'実装済み' if is_present else '未実装'}")

        # 重要な改善が含まれているかの総合判定
        if all(checks.values()):
            print("\n✅ すべての改善が実装されています")
        else:
            print("\n❌ 一部の改善が実装されていません")

    except Exception as e:
        print(f"コード確認エラー: {e}")


def main():
    """
    メイン実行関数
    """
    print("本番環境での直線的予測問題デバッグ")
    print("=" * 60)

    # 1. 現在の実装状況をデバッグ
    result = debug_current_implementation()

    # 2. Naiveモデル除外の確認
    check_model_exclusion()

    # 3. コード変更の確認
    verify_code_changes()

    print(f"\n{'=' * 60}")
    print("デバッグ完了")

    if result is None:
        print("❌ 予測処理で重大なエラーが発生しています")
        print("   → 環境設定またはコード実装を確認してください")
    else:
        print("✅ 予測処理は動作していますが、結果の品質を確認してください")


if __name__ == "__main__":
    main()
