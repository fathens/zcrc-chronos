#!/usr/bin/env python3
"""
実際の正規化処理の詳細な動作を確認するスクリプト
"""

import datetime
import os
import sys
import pandas as pd
from src.api.routes import normalize_time_series_data

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def debug_normalize_processing():
    """
    正規化処理の詳細な動作を確認する
    """
    print("=== 正規化処理の詳細デバッグ ===")

    # 問題のあるスパイクパターンを使用
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)
    timestamps = []
    values = [100, 102, 105, 180, 95, 98, 100]  # 急激な上昇と下降

    for i, value in enumerate(values):
        timestamps.append(base_time + datetime.timedelta(hours=i))

    print("元データ:")

    # 正規化処理を実行
    normalized_timestamps, normalized_values = normalize_time_series_data(
        timestamps, values, interpolation_method="linear"
    )

    print("\n3. 正規化パラメータ:")
    print(
        "   目標間隔: {} 秒".format(3600)
    )  # target_intervalは定義されていないので3600に置き換え
    print("   予測されるデータポイント数: {:.0f}".format(len(timestamps)))
    print("   実際のデータポイント数: {}".format(len(normalized_timestamps)))
    print("   元のデータポイント数: {}".format(len(timestamps)))
    ratio = len(normalized_timestamps) / len(timestamps)
    print("   拡大率: {:.2f}倍".format(ratio))
    print("   最小値: {:.6f}".format(min(normalized_values)))
    print("   最大値: {:.6f}".format(max(normalized_values)))
    print("   値域: {:.6f}".format(max(normalized_values) - min(normalized_values)))
    value_range = max(values) - min(values)
    norm_range = max(normalized_values) - min(normalized_values)
    retention_rate = (norm_range / value_range * 100) if value_range != 0 else 100.0
    print("   値域保持率: {:.2f}%".format(retention_rate))

    # 実際の値を詳細に確認
    print("\n4. 正規化後の値の詳細確認:")

    for i, (ts, val) in enumerate(zip(normalized_timestamps, normalized_values)):
        print("  [{:d}] {:s}: {:.6f}".format(i, ts.strftime("%H:%M"), val))

    # 特にスパイクが保持されているかチェック
    original_spike_idx = values.index(max(values))  # 180の位置
    print(
        "\n元データのスパイク: インデックス{:d}で値{:d}".format(
            original_spike_idx, values[original_spike_idx]
        )
    )

    # 正規化後の最大値
    normalized_max_idx = normalized_values.index(max(normalized_values))
    print(
        "正規化後の最大値: インデックス{:d}で値{:.6f}".format(
            normalized_max_idx, normalized_values[normalized_max_idx]
        )
    )

    # スパイクの急激さが保持されているかチェック
    if len(normalized_values) >= 3:
        # 元データの変化量
        original_changes = []
        for i in range(len(values) - 1):
            change = abs(values[i + 1] - values[i])
            original_changes.append(change)

        # 正規化後の変化量
        normalized_changes = []
        for i in range(len(normalized_values) - 1):
            change = abs(normalized_values[i + 1] - normalized_values[i])
            normalized_changes.append(change)

        print("\n変化量の比較:")
        print("  元データの最大変化量: {:d}".format(max(original_changes)))
        print("  正規化後の最大変化量: {:.6f}".format(max(normalized_changes)))
        change_ratio = max(normalized_changes) / max(original_changes) * 100
        print("  変化量保持率: {:.2f}%".format(change_ratio))

    return normalized_timestamps, normalized_values


def test_reindex_behavior():
    """
    pandas reindex の動作を詳細にテスト
    """

    print("\n=== pandas reindex の動作テスト ===")

    # テストデータ
    timestamps = [
        datetime.datetime(2025, 6, 14, 10, 0, 0),
        datetime.datetime(2025, 6, 14, 11, 0, 0),
        datetime.datetime(2025, 6, 14, 12, 0, 0),
        datetime.datetime(2025, 6, 14, 13, 0, 0),  # ここでスパイク 180
        datetime.datetime(2025, 6, 14, 14, 0, 0),
        datetime.datetime(2025, 6, 14, 15, 0, 0),
        datetime.datetime(2025, 6, 14, 16, 0, 0),
    ]
    values = [100, 102, 105, 180, 95, 98, 100]

    # DataFrameを作成
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    df = df.set_index("timestamp")

    print("元のDataFrame:")
    print(df)

    # 新しいタイムスタンプ（同じ間隔）
    new_timestamps = timestamps.copy()  # 同じタイムスタンプを使用

    print("\n同じタイムスタンプでreindex:")
    reindexed_df = df.reindex(new_timestamps)
    print(reindexed_df)

    # 補間
    interpolated_df = reindexed_df.interpolate(method="linear")
    print("\n線形補間後:")
    print(interpolated_df)

    # より細かい間隔でテスト
    print("\n=== より細かい間隔でのテスト ===")
    start_time = min(timestamps)
    end_time = max(timestamps)
    total_duration = (end_time - start_time).total_seconds()

    # 元のデータポイント数の2倍の細かい間隔
    num_points = len(timestamps) * 2
    interval_seconds = total_duration / (num_points - 1)

    fine_timestamps = []
    current_time = start_time
    while current_time <= end_time:
        fine_timestamps.append(current_time)
        current_time += datetime.timedelta(seconds=interval_seconds)

    print("細かいタイムスタンプ（{:d}個）:".format(len(fine_timestamps)))
    for ts in fine_timestamps:
        print("  {:s}".format(ts.strftime("%H:%M:%S")))

    fine_reindexed_df = df.reindex(fine_timestamps)
    fine_interpolated_df = fine_reindexed_df.interpolate(method="linear")

    print("\n細かい間隔での補間結果:")
    print(fine_interpolated_df)

    # スパイクが平滑化されているか確認
    original_max = max(values)
    fine_max = fine_interpolated_df["value"].max()

    print("\nスパイクの保持状況:")
    print("  元の最大値: {:d}".format(original_max))
    print("  細かい補間後の最大値: {:.6f}".format(fine_max))
    print("  値の保持率: {:.2f}%".format(fine_max / original_max * 100))


if __name__ == "__main__":
    debug_normalize_processing()
    test_reindex_behavior()
