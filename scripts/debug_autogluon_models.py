#!/usr/bin/env python3
"""
AutoGluonの実際の動作を調査するデバッグスクリプト
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_time_series():
    """
    テスト用の時系列データを作成
    """
    base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)

    # 実際のスクリーンショットのような価格変動パターン
    # より多くのデータポイントで複雑なパターン
    timestamps = []
    values = []

    # 複雑なパターンを作成（トレンド + 周期 + ランダム変動）
    for i in range(100):  # 100ポイントのデータ
        timestamps.append(base_time + datetime.timedelta(hours=i))

        # 複数の要素を組み合わせ
        trend = 39000 + i * 20  # 基本的な上昇トレンド
        cycle = 1000 * (i % 12 - 6) / 6  # 12時間周期の変動
        noise = (i % 7 - 3) * 200  # ランダムノイズ

        price = trend + cycle + noise
        values.append(max(35000, min(45000, price)))  # 価格範囲を制限

    return timestamps, values


def debug_autogluon_model_selection():
    """
    AutoGluonのモデル選択過程をデバッグ
    """
    print("=== AutoGluonモデル選択デバッグ ===")

    try:
        from src.models.predictor import TimeSeriesPredictor

        # テストデータを作成
        timestamps, values = create_test_time_series()

        print(f"テストデータ: {len(timestamps)}ポイント")
        print(f"価格範囲: {min(values):.0f} - {max(values):.0f}")
        print(f"最大変動: {max(values) - min(values):.0f}")

        # 実際のAutoGluon予測器を使用
        predictor = TimeSeriesPredictor()

        # より詳細なログレベルを設定
        import logging

        logging.basicConfig(level=logging.DEBUG)

        print("\n予測実行中...")

        # 予測実行（短い予測期間で試す）
        pred_timestamps, result, metadata = predictor.zero_shot_predict(
            timestamp=timestamps,
            values=values,
            horizon=24,  # 24時間の予測
        )

        print("\n予測結果:")
        print(f"予測値数: {len(result)}")
        print(f"予測値範囲: {min(result):.0f} - {max(result):.0f}")
        print(f"予測値の変動: {max(result) - min(result):.0f}")

        # 予測の特徴を分析
        if len(result) > 1:
            # 予測の変化率を計算
            changes = [abs(result[i + 1] - result[i]) for i in range(len(result) - 1)]
            avg_change = sum(changes) / len(changes) if changes else 0
            max_change = max(changes) if changes else 0

            print(f"平均変化量: {avg_change:.2f}")
            print(f"最大変化量: {max_change:.2f}")

            # 直線性をチェック
            if max_change < 100:  # 変化が100未満
                print("⚠️  予測が非常に平坦（ほぼ直線）")
            elif avg_change < 50:
                print("⚠️  予測の変動が少ない")
            else:
                print("✅ 予測に適度な変動がある")

        # 元データとの比較
        orig_avg_change = sum(
            abs(values[i + 1] - values[i]) for i in range(len(values) - 1)
        ) / (len(values) - 1)
        pred_avg_change = (
            sum(abs(result[i + 1] - result[i]) for i in range(len(result) - 1))
            / (len(result) - 1)
            if len(result) > 1
            else 0
        )

        print("\n変動性比較:")
        print(f"元データ平均変化: {orig_avg_change:.2f}")
        print(f"予測データ平均変化: {pred_avg_change:.2f}")
        print(
            f"変動保持率: {pred_avg_change / orig_avg_change:.2%}"
            if orig_avg_change > 0
            else "計算不可"
        )

        return result

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_different_data_sizes():
    """
    異なるデータサイズでの動作確認
    """
    print(f"\n{'=' * 60}")
    print("=== 異なるデータサイズでのテスト ===")

    data_sizes = [10, 50, 100, 200]

    for size in data_sizes:
        print(f"\nデータサイズ {size}ポイントのテスト:")

        try:
            from src.models.predictor import TimeSeriesPredictor

            # サイズに応じたテストデータ作成
            base_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
            timestamps = [base_time + datetime.timedelta(hours=i) for i in range(size)]
            values = [39000 + i * 10 + (i % 5 - 2) * 100 for i in range(size)]

            predictor = TimeSeriesPredictor()

            # 予測期間をデータサイズに応じて調整
            horizon = min(24, size // 4)

            pred_timestamps, result, metadata = predictor.zero_shot_predict(
                timestamp=timestamps, values=values, horizon=horizon
            )

            if result:
                variation = max(result) - min(result) if len(result) > 1 else 0
                print(f"  予測期間: {horizon}時間")
                print(f"  予測変動: {variation:.0f}")

                if variation < 100:
                    print("  ⚠️  直線的な予測")
                else:
                    print("  ✅ 適度な変動")

        except Exception as e:
            print(f"  エラー: {e}")


def main():
    """
    メイン実行関数
    """
    print("AutoGluonモデル動作の詳細調査")
    print("=" * 60)

    # 1. 基本的なモデル選択デバッグ
    debug_autogluon_model_selection()

    # 2. 異なるデータサイズでのテスト
    test_different_data_sizes()

    print(f"\n{'=' * 60}")
    print("調査完了")
    print("直線的な予測の原因を特定するため、")
    print("実際に使用されているモデルとその設定を確認しました。")


if __name__ == "__main__":
    main()
