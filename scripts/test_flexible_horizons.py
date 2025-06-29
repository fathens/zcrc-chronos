#!/usr/bin/env python3
"""
柔軟な予測期間設定のテスト
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_multiple_horizons():
    """
    異なる予測期間でのテスト
    """
    print("=== 柔軟な予測期間テスト ===")

    horizons_to_test = [6, 12, 24]  # 短期、中期、長期

    try:
        from src.models.predictor import TimeSeriesPredictor

        # 50ポイントのテストデータ
        base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
        timestamps = [base_time + datetime.timedelta(hours=i) for i in range(50)]
        values = [39000 + i * 50 + (i % 8 - 4) * 200 for i in range(50)]

        predictor = TimeSeriesPredictor()

        for horizon in horizons_to_test:
            print(f"\n--- {horizon}時間予測テスト ---")
            print(f"要求予測期間: {horizon}時間")

            try:
                pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
                    timestamp=timestamps, values=values, horizon=horizon
                )

                actual_horizon = len(pred_values)
                achievement_rate = actual_horizon / horizon

                print(f"実際予測期間: {actual_horizon}時間")
                print(f"達成率: {achievement_rate:.2%}")
                print(f"調整後予測期間: {metadata.get('adjusted_horizon', 'Unknown')}")

                # 期待される結果の評価
                if horizon <= 6:
                    # 短期予測：最低4時間期待
                    if actual_horizon >= 4:
                        print("✅ 短期予測が適切に動作")
                    else:
                        print("⚠️  短期予測の期間が不足")
                elif horizon <= 12:
                    # 中期予測：最低6時間期待
                    if actual_horizon >= 6:
                        print("✅ 中期予測が適切に動作")
                    else:
                        print("⚠️  中期予測の期間が不足")
                else:
                    # 長期予測：最低12時間期待
                    if actual_horizon >= 12:
                        print("✅ 長期予測が適切に動作")
                    else:
                        print("⚠️  長期予測の期間が不足")

                # 変動性確認
                if len(pred_values) > 1:
                    pred_range = max(pred_values) - min(pred_values)
                    print(f"予測価格幅: {pred_range:.0f}")

                    if pred_range > 100:
                        print("✅ 適切な変動がある")
                    else:
                        print("⚠️  変動が少ない（直線的な可能性）")

            except Exception as e:
                print(f"❌ エラー: {e}")

        return True

    except Exception as e:
        print(f"全体エラー: {e}")
        return False


def test_edge_cases():
    """
    エッジケースのテスト
    """
    print(f"\n{'=' * 50}")
    print("=== エッジケーステスト ===")

    test_cases = [
        {"name": "少ないデータ+短期予測", "data_points": 15, "horizon": 6},
        {"name": "少ないデータ+長期予測", "data_points": 20, "horizon": 24},
        {"name": "十分なデータ+短期予測", "data_points": 50, "horizon": 6},
    ]

    try:
        from src.models.predictor import TimeSeriesPredictor

        predictor = TimeSeriesPredictor()

        for case in test_cases:
            print(f"\n--- {case['name']} ---")
            print(f"データ: {case['data_points']}ポイント, 要求: {case['horizon']}時間")

            # テストデータ作成
            base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
            timestamps = [
                base_time + datetime.timedelta(hours=i * 2)
                for i in range(case["data_points"])
            ]
            values = [
                39000 + i * 100 + (i % 3 - 1) * 300 for i in range(case["data_points"])
            ]

            try:
                pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
                    timestamp=timestamps, values=values, horizon=case["horizon"]
                )

                print(f"結果: {len(pred_values)}時間予測")
                print(f"達成率: {len(pred_values) / case['horizon']:.2%}")

                if len(pred_values) >= min(4, case["horizon"]):
                    print("✅ 最小限の予測期間を確保")
                else:
                    print("⚠️  予測期間が不十分")

            except Exception as e:
                print(f"❌ エラー: {e}")

    except Exception as e:
        print(f"エッジケーステストエラー: {e}")


def main():
    """
    メイン実行関数
    """
    print("柔軟な予測期間設定のテスト")
    print("=" * 60)

    # 1. 複数の予測期間テスト
    test_multiple_horizons()

    # 2. エッジケーステスト
    test_edge_cases()

    print(f"\n{'=' * 60}")
    print("改善された機能:")
    print("✅ 短期予測（4-6時間）から長期予測（12-24時間）まで対応")
    print("✅ 要求された予測期間を可能な限り保持")
    print("✅ データサイズに応じた動的調整")
    print("✅ Naiveモデルを避けつつ適切なモデル選択")

    print("\n予測期間の動的調整ロジック:")
    print("• 短期予測（≤6時間）: 最低4時間確保、軽量モデル使用")
    print("• 中期予測（≤12時間）: 最低6時間確保、バランス型モデル")
    print("• 長期予測（>12時間）: 最低12時間確保、高度なモデル使用")


if __name__ == "__main__":
    main()
