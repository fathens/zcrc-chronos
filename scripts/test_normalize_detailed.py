import datetime
import sys
import os
import math
import random

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.routes import normalize_time_series_data

def create_test_data_with_patterns():
    """
    様々なパターンを含むテストデータを作成
    """
    # 基準時刻
    base_time = datetime.datetime.now() - datetime.timedelta(days=7)
    
    # パターン1: 上昇トレンド
    trend_timestamps = []
    trend_values = []
    for i in range(20):
        trend_timestamps.append(base_time + datetime.timedelta(hours=i*2 + random.uniform(-0.5, 0.5)))
        trend_values.append(100 + i * 5 + random.uniform(-2, 2))
    
    # パターン2: 周期的パターン（日次サイクル）
    cyclic_timestamps = []
    cyclic_values = []
    for i in range(20):
        cyclic_timestamps.append(base_time + datetime.timedelta(hours=i*3 + random.uniform(-1, 1)))
        # 24時間周期の正弦波 + ノイズ
        cyclic_values.append(50 + 20 * math.sin(2 * math.pi * i / 8) + random.uniform(-3, 3))
    
    # パターン3: 急激な変化を含むデータ
    spike_timestamps = []
    spike_values = []
    for i in range(20):
        spike_timestamps.append(base_time + datetime.timedelta(hours=i*1.5 + random.uniform(-0.3, 0.3)))
        if i == 10:  # 中間で急激な上昇
            spike_values.append(200)
        elif i == 11:  # その後急激な下降
            spike_values.append(80)
        else:
            spike_values.append(120 + random.uniform(-5, 5))
    
    return {
        'trend': (trend_timestamps, trend_values),
        'cyclic': (cyclic_timestamps, cyclic_values),
        'spike': (spike_timestamps, spike_values)
    }

def analyze_pattern_preservation(original_values, normalized_values, pattern_name):
    """
    正規化前後でパターンがどの程度保持されているかを分析
    """
    print(f"\n=== {pattern_name} パターンの分析 ===")
    
    # 基本統計
    orig_mean = sum(original_values) / len(original_values)
    orig_std = math.sqrt(sum((x - orig_mean) ** 2 for x in original_values) / len(original_values))
    norm_mean = sum(normalized_values) / len(normalized_values)
    norm_std = math.sqrt(sum((x - norm_mean) ** 2 for x in normalized_values) / len(normalized_values))
    
    print(f"元データ - 平均: {orig_mean:.2f}, 標準偏差: {orig_std:.2f}")
    print(f"正規化後 - 平均: {norm_mean:.2f}, 標準偏差: {norm_std:.2f}")
    
    # 変動の保持度
    variation_preservation = norm_std / orig_std if orig_std > 0 else 0
    print(f"変動保持率: {variation_preservation:.2%}")
    
    # 元データと正規化後データの相関（簡易計算）
    def simple_correlation(x, y):
        if len(x) != len(y):
            return None
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        return numerator / denominator if denominator != 0 else 0
    
    if len(original_values) == len(normalized_values):
        correlation = simple_correlation(original_values, normalized_values)
        print(f"相関係数: {correlation:.3f}" if correlation is not None else "相関計算不可")
    else:
        print(f"データ長が異なるため相関計算をスキップ ({len(original_values)} vs {len(normalized_values)})")
        correlation = None
    
    # トレンドの保持（一次差分の相関）
    trend_correlation = None
    if len(original_values) > 1 and len(normalized_values) > 1:
        orig_diff = [original_values[i+1] - original_values[i] for i in range(len(original_values)-1)]
        norm_diff = [normalized_values[i+1] - normalized_values[i] for i in range(len(normalized_values)-1)]
        
        if len(orig_diff) == len(norm_diff):
            trend_correlation = simple_correlation(orig_diff, norm_diff)
            if trend_correlation is not None:
                print(f"トレンド保持度（差分相関）: {trend_correlation:.3f}")
        else:
            print(f"差分データ長が異なるためトレンド相関計算をスキップ ({len(orig_diff)} vs {len(norm_diff)})")
    
    return {
        'variation_preservation': variation_preservation,
        'correlation': correlation,
        'trend_correlation': trend_correlation if 'trend_correlation' in locals() else None
    }

def test_interpolation_methods():
    """
    異なる補間方法が結果に与える影響をテスト
    """
    print("\n=== 補間方法の影響テスト ===")
    
    # スパイクを含むテストデータを作成
    base_time = datetime.datetime.now()
    timestamps = [base_time + datetime.timedelta(hours=i*2) for i in range(10)]
    values = [10, 12, 15, 45, 50, 20, 18, 16, 14, 12]  # 急激な変化を含む
    
    methods = ['linear', 'cubic', 'nearest', 'quadratic']
    results = {}
    
    for method in methods:
        try:
            norm_timestamps, norm_values = normalize_time_series_data(
                timestamps, values, interpolation_method=method
            )
            results[method] = norm_values
            
            # 最大値と最小値の保持度をチェック
            orig_max, orig_min = max(values), min(values)
            norm_max, norm_min = max(norm_values), min(norm_values)
            
            print(f"{method}補間:")
            print(f"  元データ範囲: {orig_min:.1f} - {orig_max:.1f}")
            print(f"  正規化後範囲: {norm_min:.1f} - {norm_max:.1f}")
            print(f"  最大値保持率: {norm_max/orig_max:.2%}")
            print(f"  データポイント数: {len(timestamps)} → {len(norm_values)}")
            
        except Exception as e:
            print(f"{method}補間でエラー: {e}")
    
    return results

def test_data_point_multiplication_effect():
    """
    データポイント数の倍増が平滑化に与える影響をテスト
    """
    print("\n=== データポイント倍増の影響テスト ===")
    
    # シンプルな変動データ
    base_time = datetime.datetime.now()
    timestamps = [base_time + datetime.timedelta(hours=i) for i in range(5)]
    values = [10, 20, 15, 25, 12]  # 明確な変動
    
    print(f"元データ: {values}")
    mean_val = sum(values) / len(values)
    std_val = math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))
    print(f"元データの標準偏差: {std_val:.2f}")
    
    # 正規化処理
    norm_timestamps, norm_values = normalize_time_series_data(timestamps, values)
    
    print(f"正規化後データポイント数: {len(norm_values)}")
    norm_mean = sum(norm_values) / len(norm_values)
    norm_std = math.sqrt(sum((x - norm_mean) ** 2 for x in norm_values) / len(norm_values))
    print(f"正規化後の標準偏差: {norm_std:.2f}")
    print(f"変動の減少率: {(std_val - norm_std) / std_val:.2%}")
    
    return norm_values

def main():
    """
    メイン実行関数
    """
    print("正規化処理の詳細分析を開始します...")
    
    # 1. 各種パターンでのテスト
    test_patterns = create_test_data_with_patterns()
    
    for pattern_name, (timestamps, values) in test_patterns.items():
        print(f"\n{'='*50}")
        print(f"{pattern_name.upper()} パターンのテスト")
        print(f"{'='*50}")
        
        # 正規化実行
        norm_timestamps, norm_values = normalize_time_series_data(timestamps, values)
        
        # パターン保持度分析
        analysis = analyze_pattern_preservation(values, norm_values, pattern_name)
        
        # 警告の出力
        if analysis['variation_preservation'] < 0.8:
            print(f"⚠️  変動保持率が低い ({analysis['variation_preservation']:.2%})")
        
        if analysis['correlation'] < 0.9:
            print(f"⚠️  相関が低い ({analysis['correlation']:.3f})")
        
        if analysis['trend_correlation'] is not None and analysis['trend_correlation'] < 0.7:
            print(f"⚠️  トレンド保持度が低い ({analysis['trend_correlation']:.3f})")
    
    # 2. 補間方法の比較
    interpolation_results = test_interpolation_methods()
    
    # 3. データポイント倍増の影響
    multiplication_result = test_data_point_multiplication_effect()
    
    print(f"\n{'='*50}")
    print("分析完了")
    print(f"{'='*50}")
    print("主な発見:")
    print("1. 正規化処理によりデータポイント数が約2倍に増加")
    print("2. 補間処理により急激な変化が平滑化される可能性")
    print("3. 補間方法により結果が大きく異なる")

if __name__ == "__main__":
    main()