#!/usr/bin/env python3
"""
データポイント倍増問題を詳細に調査するテスト
"""
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_point_multiplication():
    """
    データポイント倍増のロジックを詳細テスト
    """
    print("=== データポイント倍増問題の調査 ===")
    
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)
    
    # テストケース1: 規則的な間隔のデータ
    regular_timestamps = [base_time + datetime.timedelta(hours=i) for i in range(5)]
    regular_values = [10, 20, 30, 40, 50]
    
    print(f"\nテストケース1 - 規則的間隔:")
    print(f"元データ: {len(regular_timestamps)}ポイント")
    print(f"値: {regular_values}")
    
    try:
        from src.api.routes import normalize_time_series_data
        
        norm_timestamps, norm_values = normalize_time_series_data(
            regular_timestamps, regular_values, interpolation_method="linear"
        )
        
        print(f"正規化後: {len(norm_values)}ポイント")
        print(f"増加率: {len(norm_values) / len(regular_values):.1f}x")
        print(f"正規化後の値（最初の10個）: {norm_values[:10]}")
        
        # 時間間隔をチェック
        if len(norm_timestamps) > 1:
            intervals = [(norm_timestamps[i+1] - norm_timestamps[i]).total_seconds() 
                        for i in range(min(5, len(norm_timestamps)-1))]
            print(f"時間間隔（秒）: {intervals}")
        
    except Exception as e:
        print(f"エラー: {e}")
    
    # テストケース2: 不規則な間隔のデータ
    irregular_timestamps = [
        base_time,
        base_time + datetime.timedelta(minutes=30),
        base_time + datetime.timedelta(hours=2),
        base_time + datetime.timedelta(hours=5),
        base_time + datetime.timedelta(hours=6)
    ]
    irregular_values = [15, 25, 35, 45, 55]
    
    print(f"\nテストケース2 - 不規則間隔:")
    print(f"元データ: {len(irregular_timestamps)}ポイント")
    print(f"値: {irregular_values}")
    
    try:
        norm_timestamps, norm_values = normalize_time_series_data(
            irregular_timestamps, irregular_values, interpolation_method="linear"
        )
        
        print(f"正規化後: {len(norm_values)}ポイント")
        print(f"増加率: {len(norm_values) / len(irregular_values):.1f}x")
        print(f"正規化後の値（最初の10個）: {[round(x, 2) for x in norm_values[:10]]}")
        
        # 時間間隔をチェック
        if len(norm_timestamps) > 1:
            intervals = [(norm_timestamps[i+1] - norm_timestamps[i]).total_seconds() 
                        for i in range(min(5, len(norm_timestamps)-1))]
            print(f"時間間隔（秒）: {intervals}")
        
    except Exception as e:
        print(f"エラー: {e}")

def test_actual_price_scenario():
    """
    実際の価格データシナリオをテスト
    """
    print(f"\n{'='*50}")
    print("=== 実際の価格データシナリオテスト ===")
    
    # スクリーンショットのような価格変動パターンを模擬
    base_time = datetime.datetime(2025, 6, 13, 0, 0, 0)
    
    # 24時間分の価格データ（1時間間隔）
    timestamps = []
    values = []
    
    # 実際の価格のような変動パターン
    price_pattern = [
        39000, 39200, 39500, 39800, 40200, 40500, 41000, 41500,  # 上昇
        42000, 42800, 43000, 43200, 42800, 42000, 41000, 40000,  # ピーク後下降
        39000, 38000, 37000, 36500, 36200, 36000, 36100, 36200   # 安定
    ]
    
    for i, price in enumerate(price_pattern):
        timestamps.append(base_time + datetime.timedelta(hours=i))
        values.append(price)
    
    print(f"価格データ: {len(timestamps)}ポイント")
    print(f"価格範囲: {min(values):.0f} - {max(values):.0f}")
    print(f"最大変動: {max(values) - min(values):.0f}")
    
    try:
        from src.api.routes import normalize_time_series_data
        
        norm_timestamps, norm_values = normalize_time_series_data(
            timestamps, values, interpolation_method="linear"
        )
        
        print(f"\n正規化結果:")
        print(f"データポイント数: {len(timestamps)} → {len(norm_values)}")
        print(f"増加率: {len(norm_values) / len(timestamps):.1f}x")
        
        # 変動保持度
        orig_range = max(values) - min(values)
        norm_range = max(norm_values) - min(norm_values)
        range_retention = norm_range / orig_range if orig_range > 0 else 1
        
        print(f"変動保持率: {range_retention:.2%}")
        print(f"正規化後価格範囲: {min(norm_values):.0f} - {max(norm_values):.0f}")
        
        # 急激な変化の保持
        orig_max_change = max(abs(values[i+1] - values[i]) for i in range(len(values)-1))
        norm_max_change = max(abs(norm_values[i+1] - norm_values[i]) for i in range(len(norm_values)-1))
        change_retention = norm_max_change / orig_max_change if orig_max_change > 0 else 1
        
        print(f"急激変化保持率: {change_retention:.2%}")
        
        # データポイント増加が問題かどうか
        if len(norm_values) > len(timestamps) * 1.5:
            print("⚠️  データポイントが大幅に増加しています")
            print("   → 補間による平滑化の可能性があります")
        
        if range_retention < 0.95:
            print("⚠️  価格変動が減少しています")
            
        if change_retention < 0.90:
            print("⚠️  急激な価格変化が平滑化されています")
        
    except Exception as e:
        print(f"エラー: {e}")

def main():
    """
    メイン実行関数
    """
    print("データポイント倍増問題の詳細調査")
    print("="*60)
    
    test_data_point_multiplication()
    test_actual_price_scenario()
    
    print(f"\n{'='*60}")
    print("調査結果まとめ:")
    print("1. データポイント増加率とその原因を特定")
    print("2. 不規則間隔データでの動作確認")
    print("3. 実際の価格データでの変動保持度確認")

if __name__ == "__main__":
    main()