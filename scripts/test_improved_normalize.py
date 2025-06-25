#!/usr/bin/env python3
"""
改善された正規化処理をテストするスクリプト
"""
import datetime
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_price_pattern_data():
    """
    価格データのような変動パターンを含むテストデータを作成
    """
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    # パターン1: 急激な上昇と下降（価格スパイク）
    spike_timestamps = []
    spike_values = []
    spike_pattern = [100, 102, 105, 180, 95, 98, 100]  # 急激な上昇後下降
    for i, value in enumerate(spike_pattern):
        spike_timestamps.append(base_time + datetime.timedelta(hours=i))
        spike_values.append(value)

    # パターン2: 段階的な価格上昇（トレンド）
    trend_timestamps = []
    trend_values = []
    for i in range(10):
        trend_timestamps.append(base_time + datetime.timedelta(hours=i * 2))
        # 基本トレンド + ランダム変動
        base_trend = 50 + i * 3
        noise = (i % 3 - 1) * 2  # -2, 0, +2の変動
        trend_values.append(base_trend + noise)

    # パターン3: 高い変動性データ
    volatile_timestamps = []
    volatile_values = []
    volatile_pattern = [50, 65, 45, 70, 40, 75, 35, 80]  # 高い変動
    for i, value in enumerate(volatile_pattern):
        volatile_timestamps.append(base_time + datetime.timedelta(hours=i * 3))
        volatile_values.append(value)

    return {
        "spike": (spike_timestamps, spike_values),
        "trend": (trend_timestamps, trend_values),
        "volatile": (volatile_timestamps, volatile_values),
    }


def analyze_interpolation_choice(timestamps, values, pattern_name):
    """
    改善された補間方法選択ロジックをテスト
    """
    print(f"\n=== {pattern_name.upper()} パターンの補間方法分析 ===")
    print(f"元データ: {values}")

    try:
        from src.api.routes import _determine_best_interpolation_method

        # 補間方法を判定
        selected_method = _determine_best_interpolation_method(timestamps, values)
        print(f"選択された補間方法: {selected_method}")

        # データの特性を分析
        value_range = max(values) - min(values)
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)

        print("データ特性:")
        print(f"  値域: {value_range:.2f}")
        print(f"  分散: {variance:.2f}")
        print(f"  変動係数: {(variance**0.5 / mean_val):.3f}")

        # 各補間方法での結果を比較
        methods_to_test = ["linear", "cubic", "quadratic"]

        from src.api.routes import normalize_time_series_data

        print("\n補間方法別の結果比較:")
        for method in methods_to_test:
            try:
                norm_timestamps, norm_values = normalize_time_series_data(
                    timestamps, values, interpolation_method=method
                )

                # 変動保持度を計算
                orig_range = max(values) - min(values)
                norm_range = max(norm_values) - min(norm_values)
                range_retention = norm_range / orig_range if orig_range > 0 else 1

                # データポイント増加率
                point_ratio = len(norm_values) / len(values)

                print(
                    f"  {method}: 変動保持率={range_retention:.2%}, "
                    f"データ点増加={point_ratio:.1f}x"
                )

                # 急激な変化の保持度（スパイクパターンの場合）
                if pattern_name == "spike" and len(values) > 3:
                    orig_max_change = max(
                        abs(values[i + 1] - values[i]) for i in range(len(values) - 1)
                    )
                    norm_max_change = max(
                        abs(norm_values[i + 1] - norm_values[i])
                        for i in range(len(norm_values) - 1)
                    )
                    change_retention = (
                        norm_max_change / orig_max_change if orig_max_change > 0 else 1
                    )
                    print(f"    急激変化保持率: {change_retention:.2%}")

            except Exception as e:
                print(f"  {method}: エラー - {e}")

        return selected_method

    except ImportError as e:
        print(f"インポートエラー: {e}")
        return None
    except Exception as e:
        print(f"分析エラー: {e}")
        return None


def test_outlier_detection_improvement():
    """
    外れ値検出の改善をテスト
    """
    print(f"\n{'='*60}")
    print("=== 外れ値検出の改善テスト ===")

    # 価格データによくある「正常な」変動パターン
    normal_price_variations = [
        [100, 105, 98, 110, 95, 108, 102],  # 通常の価格変動
        [50, 52, 75, 48, 51, 49, 53],  # 1つのピークを含む
        [80, 82, 81, 120, 78, 79, 81],  # 急激な上昇を含む
    ]

    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)

    for i, values in enumerate(normal_price_variations):
        print(f"\nテストケース {i+1}: {values}")

        timestamps = [
            base_time + datetime.timedelta(hours=j) for j in range(len(values))
        ]

        try:
            from src.api.routes import _determine_best_interpolation_method

            method = _determine_best_interpolation_method(timestamps, values)
            print(f"選択された補間方法: {method}")

            # 改善前後の比較をシミュレート
            # max_val, min_val = max(values), min(values)  # 現在未使用

            # 従来の厳しい外れ値検出（1.5 * IQR）をシミュレート
            values_sorted = sorted(values)
            q1_idx, q3_idx = len(values) // 4, 3 * len(values) // 4
            q1, q3 = values_sorted[q1_idx], values_sorted[q3_idx]
            iqr = q3 - q1

            # 従来基準
            old_outliers = [
                v for v in values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr
            ]
            # 新基準
            new_outliers = [
                v for v in values if v < q1 - 3.0 * iqr or v > q3 + 3.0 * iqr
            ]

            print(f"  従来基準での外れ値: {old_outliers}")
            print(f"  改善後基準での外れ値: {new_outliers}")
            print(f"  外れ値削減効果: {len(old_outliers)} → {len(new_outliers)}")

        except Exception as e:
            print(f"  エラー: {e}")


def main():
    """
    メイン実行関数
    """
    print("改善された正規化処理のテストを開始します...")
    print("=" * 60)

    # 1. 価格パターンでの補間方法選択テスト
    test_patterns = create_price_pattern_data()

    results = {}
    for pattern_name, (timestamps, values) in test_patterns.items():
        method = analyze_interpolation_choice(timestamps, values, pattern_name)
        results[pattern_name] = method

    # 2. 外れ値検出の改善テスト
    test_outlier_detection_improvement()

    # 3. 結果サマリー
    print(f"\n{'='*60}")
    print("=== テスト結果サマリー ===")
    print("選択された補間方法:")
    for pattern, method in results.items():
        print(f"  {pattern}: {method}")

    print("\n改善のポイント:")
    print("✓ 外れ値検出基準を1.5×IQRから3.0×IQRに緩和")
    print("✓ 価格データの特性を考慮した補間方法選択")
    print("✓ 変動保持を優先する線形補間をデフォルトに")
    print("✓ 平滑化補間（cubic, spline）の使用条件を厳格化")


if __name__ == "__main__":
    main()
