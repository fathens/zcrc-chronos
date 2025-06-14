#!/usr/bin/env python3
"""
2倍増加が実際に発生するエッジケースを見つけるテスト
"""
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_specific_edge_cases():
    """
    len(new_timestamps) < num_points が発生する具体的なケースをテスト
    """
    print("=== 2倍増加が発生するエッジケース探索 ===")
    
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)
    
    # エッジケース1: 2つのデータポイント + 非常に短い時間範囲
    print("\nエッジケース1: 2ポイント + 非常に短い時間範囲")
    edge1_timestamps = [
        base_time,
        base_time + datetime.timedelta(seconds=1)  # 1秒差
    ]
    edge1_values = [100, 101]
    
    print(f"データ: {len(edge1_timestamps)}ポイント, 時間差: 1秒")
    test_and_analyze("edge1", edge1_timestamps, edge1_values)
    
    # エッジケース2: 同一時刻のデータポイント（あり得ないが）
    print("\nエッジケース2: 時間差0のデータ")
    edge2_timestamps = [base_time, base_time]  # 同じ時刻
    edge2_values = [100, 100]
    
    print(f"データ: {len(edge2_timestamps)}ポイント, 時間差: 0秒")
    test_and_analyze("edge2", edge2_timestamps, edge2_values)
    
    # エッジケース3: 単一データポイント
    print("\nエッジケース3: 単一データポイント")
    edge3_timestamps = [base_time]
    edge3_values = [100]
    
    print(f"データ: {len(edge3_timestamps)}ポイント")
    test_and_analyze("edge3", edge3_timestamps, edge3_values)

def test_and_analyze(test_name, timestamps, values):
    """
    正規化をテストし、内部ロジックを分析
    """
    try:
        # 正規化前にロジックを手動で再現
        print(f"  正規化前の内部ロジック分析:")
        
        if len(timestamps) <= 1:
            print(f"    データポイントが1以下のため、正規化をスキップする可能性")
            
        elif len(timestamps) >= 2:
            start_time = min(timestamps)
            end_time = max(timestamps)
            total_duration = (end_time - start_time).total_seconds()
            num_points = len(timestamps)
            
            print(f"    総時間: {total_duration}秒")
            print(f"    データポイント数: {num_points}")
            
            # 最初の間隔計算をシミュレート
            interval_seconds = max(1, total_duration / (num_points - 1)) if num_points > 1 else 1
            print(f"    計算された間隔: {interval_seconds}秒")
            
            # new_timestampsの予想サイズを計算
            current_time = start_time
            predicted_new_count = 0
            while current_time <= end_time:
                predicted_new_count += 1
                current_time += datetime.timedelta(seconds=interval_seconds)
                if predicted_new_count > num_points * 3:  # 無限ループ防止
                    break
            
            print(f"    予想される新タイムスタンプ数: {predicted_new_count}")
            print(f"    元ポイント数との比較: {predicted_new_count} vs {num_points}")
            
            if predicted_new_count < num_points:
                print(f"    → 2倍増加条件が満たされる可能性！")
                adjusted_interval = total_duration / (num_points * 2)
                print(f"    → 調整後間隔: {adjusted_interval}秒")
        
        # 実際の正規化を実行
        from src.api.routes import normalize_time_series_data
        
        norm_timestamps, norm_values = normalize_time_series_data(
            timestamps, values, interpolation_method="linear"
        )
        
        increase_ratio = len(norm_values) / len(values) if len(values) > 0 else 0
        print(f"  実際の結果: {len(values)} → {len(norm_values)}ポイント (増加率: {increase_ratio:.1f}x)")
        
        if increase_ratio >= 1.8:  # 1.8倍以上を「2倍増加」とみなす
            print(f"  🎯 2倍増加を検出！")
            
            # 詳細分析
            if len(norm_timestamps) > 1:
                intervals = [(norm_timestamps[i+1] - norm_timestamps[i]).total_seconds() 
                           for i in range(len(norm_timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                print(f"    正規化後の平均間隔: {avg_interval:.3f}秒")
                print(f"    間隔の一様性: {min(intervals):.3f} - {max(intervals):.3f}秒")
        
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()

def test_floating_point_precision():
    """
    浮動小数点精度が原因の問題をテスト
    """
    print(f"\n{'='*60}")
    print("=== 浮動小数点精度問題のテスト ===")
    
    base_time = datetime.datetime(2025, 6, 14, 10, 0, 0)
    
    # 非常に短い間隔で多くのポイント
    print("\n浮動小数点精度テスト: 0.1秒間隔で10ポイント")
    fp_timestamps = []
    fp_values = []
    
    for i in range(10):
        # 0.1秒間隔（浮動小数点計算で問題が起きやすい）
        fp_timestamps.append(base_time + datetime.timedelta(milliseconds=i*100))
        fp_values.append(100 + i)
    
    print(f"データ: {len(fp_timestamps)}ポイント, 間隔: 0.1秒")
    test_and_analyze("floating_point", fp_timestamps, fp_values)

def main():
    """
    メイン実行関数
    """
    print("2倍増加発生条件の徹底調査")
    print("="*60)
    
    test_specific_edge_cases()
    test_floating_point_precision()
    
    print(f"\n{'='*60}")
    print("調査結果:")
    print("- エッジケースでの2倍増加条件の発生確認")
    print("- 内部ロジックの詳細分析")
    print("- 浮動小数点精度問題の影響確認")

if __name__ == "__main__":
    main()