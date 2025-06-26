#!/usr/bin/env python3
"""
予測期間改善の迅速テスト
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_prediction_test():
    """
    24時間予測の迅速テスト
    """
    print("=== 迅速予測テスト ===")

    try:
        from src.models.predictor import TimeSeriesPredictor

        # 50ポイントのデータで24時間予測をテスト
        base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
        timestamps = [base_time + datetime.timedelta(hours=i * 2) for i in range(50)]

        # シンプルなトレンドデータ
        values = [39000 + i * 100 + (i % 5 - 2) * 200 for i in range(50)]

        print(f"入力データ: {len(timestamps)}ポイント")
        print(f"価格範囲: {min(values):.0f} - {max(values):.0f}")

        predictor = TimeSeriesPredictor()

        print("24時間予測を実行中...")
        pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=24
        )

        print("\n結果:")
        print("要求予測期間: 24時間")
        print(f"実際予測期間: {len(pred_values)}時間")
        print(f"予測期間達成率: {len(pred_values) / 24:.2%}")

        if len(pred_values) >= 18:  # 75%以上達成
            print("✅ 大幅な改善が確認されました")
        elif len(pred_values) >= 12:  # 50%以上達成
            print("✅ 改善が確認されました")
        else:
            print("⚠️  さらなる改善が必要")

        # 変動性確認
        if len(pred_values) > 1:
            pred_changes = [
                abs(pred_values[i + 1] - pred_values[i])
                for i in range(len(pred_values) - 1)
            ]
            avg_change = sum(pred_changes) / len(pred_changes)
            max_change = max(pred_changes)

            print("\n予測の変動:")
            print(f"平均変化: {avg_change:.2f}")
            print(f"最大変化: {max_change:.2f}")

            if max_change > 100:
                print("✅ 予測に適切な変動があります")
            else:
                print("⚠️  予測の変動が少ない")

        return True

    except Exception as e:
        print(f"エラー: {e}")
        return False


def main():
    """
    メイン実行関数
    """
    print("予測期間改善の迅速確認")
    print("=" * 40)

    success = quick_prediction_test()

    print(f"\n{'=' * 40}")
    if success:
        print("テスト完了")
        print("実際のアプリケーションでの確認をお勧めします")
    else:
        print("テスト失敗 - さらなる調整が必要")


if __name__ == "__main__":
    main()
