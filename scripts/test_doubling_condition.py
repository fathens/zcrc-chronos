#!/usr/bin/env python3
"""
データポイント2倍増加が発生する条件をテストする
"""
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_doubling_trigger_conditions():
    """
    2倍増加が発生する具体的な条件をテスト
    """
    print("=== データポイント2倍増加トリガー条件テスト ===")

    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    # 条件1: 非常に短い時間範囲に多くのデータポイント
    print("\n条件1: 短時間に密集したデータ")
    dense_timestamps = []
    dense_values = []
    for i in range(10):
        # 10分間に10ポイント
        dense_timestamps.append(base_time + datetime.timedelta(minutes=i))
        dense_values.append(100 + i)

    print(f"データ: {len(dense_timestamps)}ポイント, 時間範囲: 10分")
    test_normalization("dense", dense_timestamps, dense_values)

    # 条件2: 非常に長い時間範囲に少数のデータポイント
    print("\n条件2: 長時間に散らばったデータ")
    sparse_timestamps = []
    sparse_values = []
    for i in range(3):
        # 1週間に3ポイント
        sparse_timestamps.append(base_time + datetime.timedelta(days=i * 3))
        sparse_values.append(50 + i * 10)

    print(f"データ: {len(sparse_timestamps)}ポイント, 時間範囲: 6日")
    test_normalization("sparse", sparse_timestamps, sparse_values)

    # 条件3: 極端に不規則な間隔
    print("\n条件3: 極端に不規則な間隔")
    irregular_timestamps = [
        base_time,
        base_time + datetime.timedelta(seconds=1),  # 1秒後
        base_time + datetime.timedelta(hours=1),  # 1時間後
        base_time + datetime.timedelta(days=1),  # 1日後
        base_time + datetime.timedelta(days=1, seconds=1),  # さらに1秒後
    ]
    irregular_values = [10, 20, 30, 40, 50]

    print(f"データ: {len(irregular_timestamps)}ポイント, 極端に不規則")
    test_normalization("irregular", irregular_timestamps, irregular_values)


def test_normalization(test_name, timestamps, values):
    """
    正規化をテストし、結果を分析
    """
    try:
        from src.api.routes import normalize_time_series_data

        norm_timestamps, norm_values = normalize_time_series_data(
            timestamps, values, interpolation_method="linear"
        )

        increase_ratio = len(norm_values) / len(values)
        print(
            f"  結果: {len(values)} → {len(norm_values)}ポイント (増加率: {increase_ratio:.1f}x)"
        )

        if increase_ratio > 1.5:
            print("  ⚠️  2倍増加が発生！")

            # 時間間隔を分析
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            original_avg_interval = (
                total_duration / (len(timestamps) - 1) if len(timestamps) > 1 else 0
            )

            if len(norm_timestamps) > 1:
                norm_intervals = [
                    (norm_timestamps[i + 1] - norm_timestamps[i]).total_seconds()
                    for i in range(len(norm_timestamps) - 1)
                ]
                norm_avg_interval = sum(norm_intervals) / len(norm_intervals)

                print(f"    元の平均間隔: {original_avg_interval:.1f}秒")
                print(f"    正規化後平均間隔: {norm_avg_interval:.1f}秒")
                print(f"    間隔比: {original_avg_interval / norm_avg_interval:.1f}x")

        # 変動保持をチェック
        if len(values) > 1:
            orig_range = max(values) - min(values)
            norm_range = max(norm_values) - min(norm_values)
            range_retention = norm_range / orig_range if orig_range > 0 else 1
            print(f"  変動保持率: {range_retention:.2%}")

            if range_retention < 0.95:
                print("  ⚠️  変動が減少")

    except Exception as e:
        print(f"  エラー: {e}")


def test_realistic_trading_scenario():
    """
    実際の取引データのようなシナリオをテスト
    """
    print(f"\n{'='*60}")
    print("=== 実際の取引データシナリオ ===")

    base_time = datetime.datetime(2025, 6, 14, 9, 0, 0)  # 取引開始時刻

    # シナリオ1: 高頻度取引データ（1分間隔で60ポイント）
    print("\nシナリオ1: 高頻度取引データ（1分間隔）")
    hft_timestamps = []
    hft_values = []
    base_price = 36000

    for i in range(60):  # 1時間分の1分足データ
        hft_timestamps.append(base_time + datetime.timedelta(minutes=i))
        # 実際の価格変動のようなランダムウォーク
        price_change = (i % 7 - 3) * 50  # -150 to +150 の変動
        hft_values.append(base_price + price_change + i * 2)  # 微小なトレンド

    print(f"データ点数: {len(hft_timestamps)}")
    print(
        f"時間範囲: {(hft_timestamps[-1] - hft_timestamps[0]).total_seconds()/3600:.1f}時間"
    )
    test_normalization("hft", hft_timestamps, hft_values)

    # シナリオ2: 日足データ（30日間）
    print("\nシナリオ2: 日足データ（30日間）")
    daily_timestamps = []
    daily_values = []

    for i in range(30):
        daily_timestamps.append(base_time + datetime.timedelta(days=i))
        # 日足レベルの価格変動
        daily_change = (i % 5 - 2) * 500  # より大きな変動
        daily_values.append(base_price + daily_change + i * 100)

    print(f"データ点数: {len(daily_timestamps)}")
    print(f"時間範囲: {(daily_timestamps[-1] - daily_timestamps[0]).days}日")
    test_normalization("daily", daily_timestamps, daily_values)


def main():
    """
    メイン実行関数
    """
    print("データポイント2倍増加条件の詳細テスト")
    print("=" * 60)

    test_doubling_trigger_conditions()
    test_realistic_trading_scenario()

    print(f"\n{'='*60}")
    print("テスト結果まとめ:")
    print("- 2倍増加が発生する具体的条件を特定")
    print("- 実際の取引データでの動作確認")
    print("- 各シナリオでの変動保持度評価")


if __name__ == "__main__":
    main()
