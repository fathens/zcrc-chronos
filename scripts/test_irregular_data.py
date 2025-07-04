#!/usr/bin/env python3
"""
不規則な時間間隔のデータで正規化処理をテスト
"""

import datetime
import os
import sys
import numpy as np
from src.api.routes import normalize_time_series_data

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_irregular_timestamps():
    """
    不規則な時間間隔でのテスト
    """
    print("=== 不規則な時間間隔での正規化テスト ===")

    # 不規則な時間間隔のデータ
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    # 不規則な間隔（実際のトークン価格データのような）
    irregular_intervals = [0, 3, 8, 15, 35, 62, 95]  # 分単位
    timestamps = []
    values = [100, 102, 105, 180, 95, 98, 100]  # スパイクを含む価格データ

    for i, minutes in enumerate(irregular_intervals):
        timestamps.append(base_time + datetime.timedelta(minutes=minutes))

    print("元データ（不規則な間隔）:")
    print("  時間間隔: {}".format([t.strftime("%H:%M") for t in timestamps]))
    print("  値: {}".format(values))
    max_val = max(values)
    min_val = min(values)
    value_range = max_val - min_val
    print("  最大値: {}, 最小値: {}, 値域: {}".format(max_val, min_val, value_range))

    # 時間間隔をチェック
    time_diffs = [
        (timestamps[i + 1] - timestamps[i]).total_seconds() / 60
        for i in range(len(timestamps) - 1)
    ]
    print(f"  時間間隔（分）: {time_diffs}")

    # 正規化処理を実行
    normalized_timestamps, normalized_values = normalize_time_series_data(
        timestamps, values, interpolation_method="linear"
    )

    print("\n正規化後:")
    print("  タイムスタンプ数: {}".format(len(normalized_timestamps)))
    print("  値の数: {}".format(len(normalized_values)))
    print("  最大値: {:.6f}".format(max(normalized_values)))
    print("  最小値: {:.6f}".format(min(normalized_values)))
    norm_range = max(normalized_values) - min(normalized_values)
    print("  値域: {:.6f}".format(norm_range))
    value_range = max(values) - min(values)
    retention = (norm_range / value_range * 100) if value_range != 0 else 100.0
    print("  値域保持率: {:.2f}%".format(retention))

    # 新しい時間間隔をチェック
    new_time_diffs = []
    for i in range(len(normalized_timestamps) - 1):
        time_diff = normalized_timestamps[i + 1] - normalized_timestamps[i]
        new_time_diffs.append(time_diff.total_seconds() / 60)
    formatted_diffs = ["{:.1f}".format(float(diff)) for diff in new_time_diffs]
    print("  新しい時間間隔（分）: {}".format(formatted_diffs))

    # 詳細な値の比較
    print("\n詳細な値の比較:")
    for i, (ts, val) in enumerate(zip(normalized_timestamps, normalized_values)):
        original_val = values[i] if i < len(values) else "N/A"
        print(
            "  [{}] {}: {:.6f} (元: {})".format(
                i, ts.strftime("%H:%M"), val, original_val
            )
        )

    # スパイクの保持確認
    original_spike_value = max(values)
    normalized_spike_value = max(normalized_values)
    spike_retention = normalized_spike_value / original_spike_value * 100

    print("\nスパイク保持分析:")
    print("  元のスパイク値: {}".format(original_spike_value))
    print("  正規化後のスパイク値: {:.6f}".format(normalized_spike_value))
    print("  スパイク保持率: {:.2f}%".format(spike_retention))

    return normalized_timestamps, normalized_values


def test_very_irregular_data():
    """
    非常に不規則なデータでのテスト
    """

    print("\n=== 非常に不規則なデータでのテスト ===")

    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    # 非常に不規則な間隔（実際のDeFiトークンのような）
    very_irregular_intervals = [0, 1, 5, 6, 25, 180, 185]  # 分単位
    timestamps = []
    values = [50, 55, 48, 85, 45, 92, 50]  # 激しい変動

    for i, minutes in enumerate(very_irregular_intervals):
        timestamps.append(base_time + datetime.timedelta(minutes=minutes))

    print("元データ（非常に不規則な間隔）:")
    print("  時間間隔: {}".format([t.strftime("%H:%M") for t in timestamps]))
    print("  値: {}".format(values))

    # 時間間隔をチェック
    time_diffs = [
        (timestamps[i + 1] - timestamps[i]).total_seconds() / 60
        for i in range(len(timestamps) - 1)
    ]
    print("  時間間隔（分）: {}".format(time_diffs))

    # 変動係数を計算
    time_diff_array = np.array(time_diffs, dtype=np.float64)
    mean_interval = np.mean(time_diff_array)
    std_interval = np.std(time_diff_array)
    cv = std_interval / mean_interval if mean_interval > 0 else 0
    print("  時間間隔の変動係数（CV）: {:.3f}".format(cv))

    # 正規化処理を実行
    normalized_timestamps, normalized_values = normalize_time_series_data(
        timestamps, values, interpolation_method="linear"
    )

    print("\n正規化後:")
    value_range = max(values) - min(values)
    norm_range = max(normalized_values) - min(normalized_values)
    retention = (norm_range / value_range * 100) if value_range != 0 else 100.0
    print("  値域保持率: {:.2f}%".format(retention))

    # 新しい時間間隔をチェック
    new_time_diffs = []
    for i in range(len(normalized_timestamps) - 1):
        time_diff = normalized_timestamps[i + 1] - normalized_timestamps[i]
        new_time_diffs.append(time_diff.total_seconds() / 60)
    formatted_diffs = ["{:.1f}".format(float(diff)) for diff in new_time_diffs]
    print("  新しい時間間隔（分）: {}".format(formatted_diffs))


if __name__ == "__main__":
    test_irregular_timestamps()
    test_very_irregular_data()
