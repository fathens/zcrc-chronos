#!/usr/bin/env python3
"""
改善された予測ロジックをテストする
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_prediction_horizon_fix():
    """
    予測期間制限の修正をテスト
    """
    print("=== 予測期間制限修正のテスト ===")

    try:
        from src.models.predictor import TimeSeriesPredictor

        # 100ポイントのテストデータ
        base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
        timestamps = [base_time + datetime.timedelta(hours=i) for i in range(100)]

        # より現実的な価格変動パターン
        values = []
        base_price = 39000
        for i in range(100):
            trend = i * 20  # 上昇トレンド
            cycle = 500 * ((i % 24) / 24 - 0.5)  # 24時間周期
            noise = ((i * 7) % 13 - 6) * 50  # ランダム変動
            price = base_price + trend + cycle + noise
            values.append(max(35000, min(45000, price)))

        predictor = TimeSeriesPredictor()

        print(f"テストデータ: {len(timestamps)}ポイント")
        print(f"価格範囲: {min(values):.0f} - {max(values):.0f}")

        # 24時間予測をテスト
        print("\n24時間予測をテスト中...")
        pred_timestamps, result, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=24
        )

        print("\n結果:")
        print(f"予測期間: {len(result)}時間")
        print(f"予測価格範囲: {min(result):.0f} - {max(result):.0f}")

        if len(result) > 1:
            # 予測の変動性を確認
            pred_changes = [
                abs(result[i + 1] - result[i]) for i in range(len(result) - 1)
            ]
            avg_change = sum(pred_changes) / len(pred_changes)
            max_change = max(pred_changes)

            print(f"予測の平均変化: {avg_change:.2f}")
            print(f"予測の最大変化: {max_change:.2f}")

            # 元データとの比較
            orig_changes = [
                abs(values[i + 1] - values[i]) for i in range(len(values) - 1)
            ]
            orig_avg_change = sum(orig_changes) / len(orig_changes)

            print(f"元データの平均変化: {orig_avg_change:.2f}")
            print(f"変動保持率: {avg_change / orig_avg_change:.2%}")

            # 改善の評価
            if len(result) >= 12:  # 少なくとも12時間の予測
                print("✅ 予測期間が適切に確保されました")
            else:
                print("⚠️  予測期間がまだ短すぎます")

            if avg_change > 50:  # 適度な変動があるか
                print("✅ 予測に適度な変動があります")
            else:
                print("⚠️  予測の変動が少なすぎます（直線的）")

        return result

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_different_horizons():
    """
    異なる予測期間でのテスト
    """
    print(f"\n{'=' * 50}")
    print("=== 異なる予測期間でのテスト ===")

    horizons = [6, 12, 24, 48]

    for horizon in horizons:
        print(f"\n{horizon}時間予測のテスト:")

        try:
            from src.models.predictor import TimeSeriesPredictor

            # データサイズを予測期間に応じて調整
            data_size = max(50, horizon * 3)
            base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
            timestamps = [
                base_time + datetime.timedelta(hours=i) for i in range(data_size)
            ]

            # 変動のあるテストデータ
            values = [39000 + i * 15 + (i % 8 - 4) * 100 for i in range(data_size)]

            predictor = TimeSeriesPredictor()

            pred_timestamps, result, metadata = predictor.zero_shot_predict(
                timestamp=timestamps, values=values, horizon=horizon
            )

            actual_horizon = len(result)
            retention_rate = actual_horizon / horizon

            print(f"  要求期間: {horizon}時間")
            print(f"  実際期間: {actual_horizon}時間")
            print(f"  期間保持率: {retention_rate:.2%}")

            if retention_rate >= 0.5:  # 50%以上保持
                print("  ✅ 予測期間が適切に保持されました")
            else:
                print("  ⚠️  予測期間が大幅に削減されました")

        except Exception as e:
            print(f"  エラー: {e}")


def main():
    """
    メイン実行関数
    """
    print("改善された予測ロジックのテスト")
    print("=" * 60)

    # 1. 予測期間制限の修正テスト
    test_prediction_horizon_fix()

    # 2. 異なる予測期間でのテスト
    test_different_horizons()

    print(f"\n{'=' * 60}")
    print("テスト完了")
    print("\n実装した改善:")
    print("✅ 予測期間制限を大幅に緩和（データサイズの25%まで）")
    print("✅ 最小予測期間を6時間に設定")
    print("✅ Naiveモデルを除外して高度なモデルを優先")
    print("✅ データ要件を実用的なレベルに緩和")


if __name__ == "__main__":
    main()
