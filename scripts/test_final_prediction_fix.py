#!/usr/bin/env python3
"""
最終的な予測期間修正をテストする
"""
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_24_hour_prediction():
    """
    24時間予測がちゃんと実行されるかテスト
    """
    print("=== 24時間予測テスト ===")

    try:
        from src.models.predictor import TimeSeriesPredictor

        # 実際のケースに近い24ポイントのデータ
        base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
        timestamps = [base_time + datetime.timedelta(hours=i * 6) for i in range(24)]

        # スクリーンショットのような価格変動
        values = []
        base_price = 39000
        for i in range(24):
            if i < 8:  # 初期段階
                price = base_price + i * 100 + (i % 3 - 1) * 300
            elif i < 16:  # 上昇段階
                price = base_price + 1000 + (i - 8) * 500 + (i % 2) * 200
            else:  # 下降段階
                price = base_price + 4000 - (i - 16) * 600 + (i % 2) * 100
            values.append(max(35000, min(45000, price)))

        print(f"入力データ: {len(timestamps)}ポイント")
        print(f"価格範囲: {min(values):.0f} - {max(values):.0f}")
        print(f"データ期間: {(timestamps[-1] - timestamps[0]).days}日")

        predictor = TimeSeriesPredictor()

        # 24時間予測を要求
        print("\n24時間予測を実行中...")
        pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=24
        )

        print("\n結果:")
        print("要求予測期間: 24時間")
        print(f"実際予測期間: {len(pred_values)}時間")
        print(f"予測期間達成率: {len(pred_values) / 24:.2%}")

        if len(pred_values) >= 12:
            print("✅ 十分な予測期間が確保されました")
        else:
            print("⚠️  予測期間がまだ不十分です")

        # 予測の変動性確認
        if len(pred_values) > 1:
            pred_range = max(pred_values) - min(pred_values)
            pred_changes = [
                abs(pred_values[i + 1] - pred_values[i])
                for i in range(len(pred_values) - 1)
            ]
            avg_change = sum(pred_changes) / len(pred_changes)

            print("\n予測の変動性:")
            print(f"予測価格範囲: {min(pred_values):.0f} - {max(pred_values):.0f}")
            print(f"予測価格幅: {pred_range:.0f}")
            print(f"平均変化量: {avg_change:.2f}")

            # 元データとの比較
            orig_range = max(values) - min(values)
            print(f"元データ価格幅: {orig_range:.0f}")
            print(f"変動保持率: {pred_range / orig_range:.2%}")

            if avg_change > 100:
                print("✅ 予測に十分な変動があります")
            else:
                print("⚠️  予測の変動が少ない（直線的な可能性）")

        # メタデータから実際に使用されたモデルを確認
        print("\n使用されたモデル:")
        print(f"モデルタイプ: {metadata.get('model_type', 'Unknown')}")
        print(f"調整後予測期間: {metadata.get('adjusted_horizon', 'Unknown')}")

        return pred_values

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_longer_horizon():
    """
    より長い予測期間でのテスト
    """
    print(f"\n{'='*50}")
    print("=== 長期予測テスト ===")

    try:
        from src.models.predictor import TimeSeriesPredictor

        # より多くのデータでより長い予測を試す
        base_time = datetime.datetime(2025, 6, 1, 0, 0, 0)
        timestamps = [base_time + datetime.timedelta(hours=i * 2) for i in range(100)]

        # 複雑な価格変動パターン
        values = []
        for i in range(100):
            base = 40000
            trend = i * 30  # 長期トレンド
            cycle = 1000 * ((i % 24) / 24 - 0.5)  # 日次サイクル
            noise = ((i * 13) % 29 - 14) * 100  # ランダム変動
            price = base + trend + cycle + noise
            values.append(max(35000, min(50000, price)))

        predictor = TimeSeriesPredictor()

        print(f"入力データ: {len(timestamps)}ポイント")
        print("48時間予測を実行中...")

        pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=48
        )

        print("\n結果:")
        print("要求予測期間: 48時間")
        print(f"実際予測期間: {len(pred_values)}時間")
        print(f"予測期間達成率: {len(pred_values) / 48:.2%}")

        if len(pred_values) >= 24:
            print("✅ 実用的な予測期間が確保されました")

        return pred_values

    except Exception as e:
        print(f"エラー: {e}")
        return None


def main():
    """
    メイン実行関数
    """
    print("最終的な予測期間修正のテスト")
    print("=" * 60)

    # 1. 24時間予測テスト
    test_24_hour_prediction()

    # 2. 長期予測テスト
    test_longer_horizon()

    print(f"\n{'='*60}")
    print("最終修正内容:")
    print("✅ 最大予測期間をデータサイズの50%に拡大")
    print("✅ 最低予測期間を12-24時間に設定")
    print("✅ 最小データ要件を予測期間+10ポイントに削減")
    print("✅ 長期予測を優先する設定に変更")

    print("\n期待される効果:")
    print("• より長い予測期間の確保")
    print("• 価格変動パターンの適切な表現")
    print("• 直線的予測の大幅な改善")


if __name__ == "__main__":
    main()
