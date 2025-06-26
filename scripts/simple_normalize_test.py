#!/usr/bin/env python3
"""
正規化処理の問題を特定するためのシンプルなテスト
"""

import datetime
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_simple_pattern():
    """
    明確なパターンを持つシンプルなデータで正規化をテスト
    """
    # 簡単なパターンのデータを作成
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    # テスト1: 明確な上昇トレンド
    timestamps = []
    values = []
    for i in range(5):
        timestamps.append(base_time + datetime.timedelta(hours=i))
        values.append(10 + i * 10)  # 10, 20, 30, 40, 50

    print("=== 上昇トレンドテスト ===")
    print(f"元データ: {values}")
    print(f"データポイント数: {len(values)}")

    # 変動の指標を計算
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    print(f"元データの分散: {variance:.2f}")

    # 最大値と最小値の差
    value_range = max(values) - min(values)
    print(f"元データの値域: {value_range}")

    try:
        from src.api.routes import normalize_time_series_data

        # 正規化実行
        norm_timestamps, norm_values = normalize_time_series_data(timestamps, values)

        print(f"\n正規化後データ: {norm_values}")
        print(f"正規化後データポイント数: {len(norm_values)}")

        # 正規化後の統計
        norm_mean = sum(norm_values) / len(norm_values)
        norm_variance = sum((x - norm_mean) ** 2 for x in norm_values) / len(
            norm_values
        )
        norm_range = max(norm_values) - min(norm_values)

        print(f"正規化後の分散: {norm_variance:.2f}")
        print(f"正規化後の値域: {norm_range:.2f}")

        # 変動の保持率
        variance_retention = norm_variance / variance if variance > 0 else 0
        range_retention = norm_range / value_range if value_range > 0 else 0

        print(f"\n分散保持率: {variance_retention:.2%}")
        print(f"値域保持率: {range_retention:.2%}")

        # データポイント増加率
        point_increase = len(norm_values) / len(values)
        print(f"データポイント増加率: {point_increase:.1f}x")

        # 警告の出力
        if variance_retention < 0.8:
            print("⚠️  分散が大幅に減少しています（変動が平滑化されています）")

        if range_retention < 0.9:
            print("⚠️  値域が減少しています（極値が平滑化されています）")

        if point_increase > 1.5:
            print("⚠️  データポイントが大幅に増加しています（補間による影響の可能性）")

    except ImportError as e:
        print(f"インポートエラー: {e}")
        print("conda環境を有効化してください: conda activate zcrc-chronos")
        return False
    except Exception as e:
        print(f"正規化処理でエラー: {e}")
        return False

    return True


def test_spike_pattern():
    """
    急激な変化（スパイク）を含むデータで正規化をテスト
    """
    print("\n" + "=" * 50)
    print("=== スパイクパターンテスト ===")

    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    # スパイクを含むデータ
    timestamps = []
    values = []
    spike_data = [10, 12, 15, 100, 8, 10, 11]  # 4番目にスパイク

    for i, value in enumerate(spike_data):
        timestamps.append(base_time + datetime.timedelta(hours=i))
        values.append(value)

    print(f"元データ: {values}")

    try:
        from src.api.routes import normalize_time_series_data

        norm_timestamps, norm_values = normalize_time_series_data(timestamps, values)

        print(f"正規化後データ: {[round(x, 2) for x in norm_values]}")

        # スパイクの保持度をチェック
        orig_max = max(values)
        norm_max = max(norm_values)
        spike_retention = norm_max / orig_max

        print(f"スパイク保持率: {spike_retention:.2%}")

        if spike_retention < 0.8:
            print("⚠️  スパイクが大幅に平滑化されています")

    except Exception as e:
        print(f"エラー: {e}")
        return False

    return True


def test_interpolation_methods():
    """
    異なる補間方法での結果を比較
    """
    print("\n" + "=" * 50)
    print("=== 補間方法比較テスト ===")

    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)
    timestamps = [base_time + datetime.timedelta(hours=i * 2) for i in range(4)]
    values = [10, 50, 20, 40]  # 変動の大きいデータ

    print(f"元データ: {values}")

    methods = ["linear", "cubic", "nearest"]

    try:
        from src.api.routes import normalize_time_series_data

        for method in methods:
            try:
                norm_timestamps, norm_values = normalize_time_series_data(
                    timestamps, values, interpolation_method=method
                )

                print(f"\n{method}補間:")
                print(
                    f"  結果: {[round(x, 2) for x in norm_values[:8]]}..."
                )  # 最初の8個だけ表示
                print(f"  データ点数: {len(norm_values)}")

                # 変動の保持度
                orig_std = (
                    sum((x - sum(values) / len(values)) ** 2 for x in values)
                    / len(values)
                ) ** 0.5
                norm_mean = sum(norm_values) / len(norm_values)
                norm_std = (
                    sum((x - norm_mean) ** 2 for x in norm_values) / len(norm_values)
                ) ** 0.5

                retention = norm_std / orig_std if orig_std > 0 else 0
                print(f"  変動保持率: {retention:.2%}")

            except Exception as e:
                print(f"{method}補間でエラー: {e}")

    except Exception as e:
        print(f"エラー: {e}")
        return False

    return True


def main():
    """
    メイン関数
    """
    print("正規化処理の問題分析テストを開始します...")
    print("=" * 60)

    success = True

    # 各テストを実行
    success &= test_simple_pattern()
    success &= test_spike_pattern()
    success &= test_interpolation_methods()

    print("\n" + "=" * 60)
    if success:
        print("すべてのテストが完了しました")
    else:
        print("一部のテストでエラーが発生しました")

    print("\n主な発見:")
    print("1. 正規化処理により元データの変動が平滑化される可能性")
    print("2. スパイクや急激な変化が減衰される可能性")
    print("3. データポイント数の増加により補間による影響が大きくなる可能性")
    print("4. 補間方法により結果が大きく異なる可能性")


if __name__ == "__main__":
    main()
