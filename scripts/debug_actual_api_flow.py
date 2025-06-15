#!/usr/bin/env python3
"""
実際のAPI処理フローを詳細に調査するスクリプト
"""
import datetime
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def trace_api_flow():
    """
    実際のAPI処理フローをトレース
    """
    print("=== 実際のAPI処理フロー調査 ===")
    
    try:
        from src.api.routes import normalize_time_series_data, _determine_best_interpolation_method
        from src.models.predictor import TimeSeriesPredictor
        
        # 実際のリクエストに近いデータを作成
        # スクリーンショットの時間範囲を模擬
        start_time = datetime.datetime(2025, 6, 8, 0, 0, 0)
        end_time = datetime.datetime(2025, 6, 14, 0, 0, 0)
        
        # 6日間のデータ（6時間間隔で24ポイント）
        timestamps = []
        values = []
        
        for i in range(24):
            current_time = start_time + datetime.timedelta(hours=i*6)
            timestamps.append(current_time)
            
            # 実際のチャートのような価格変動を模擬
            if i < 8:  # 最初の部分：緩やかな変動
                price = 39000 + i * 100 + (i % 3 - 1) * 200
            elif i < 16:  # 中間部分：上昇トレンド
                price = 40000 + (i - 8) * 400 + (i % 2) * 300
            else:  # 最後の部分：急激な下降
                price = 43000 - (i - 16) * 800 + (i % 2) * 100
            
            values.append(max(35000, min(45000, price)))
        
        print(f"入力データ: {len(timestamps)}ポイント")
        print(f"時間範囲: {timestamps[0]} to {timestamps[-1]}")
        print(f"価格範囲: {min(values):.0f} - {max(values):.0f}")
        print(f"元データの値: {[int(v) for v in values[:5]]}... (最初の5個)")
        
        # STEP 1: 正規化処理を詳細に追跡
        print(f"\n--- STEP 1: 正規化処理 ---")
        
        # 補間方法の判定
        method = _determine_best_interpolation_method(timestamps, values)
        print(f"選択された補間方法: {method}")
        
        # 正規化実行
        norm_timestamps, norm_values = normalize_time_series_data(
            timestamps, values, interpolation_method=method
        )
        
        print(f"正規化後: {len(norm_values)}ポイント")
        print(f"正規化後の値: {[int(v) for v in norm_values[:5]]}... (最初の5個)")
        
        # 正規化での変化をチェック
        orig_range = max(values) - min(values)
        norm_range = max(norm_values) - min(norm_values)
        print(f"変動保持: {orig_range:.0f} → {norm_range:.0f} ({norm_range/orig_range:.2%})")
        
        # STEP 2: AutoGluon予測処理を詳細に追跡
        print(f"\n--- STEP 2: AutoGluon予測処理 ---")
        
        predictor = TimeSeriesPredictor()
        
        # 実際の予測を実行（詳細ログ付き）
        print(f"予測実行中（24時間予測）...")
        pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
            timestamp=norm_timestamps,
            values=norm_values,
            horizon=24
        )
        
        print(f"\n--- STEP 3: 予測結果分析 ---")
        print(f"予測ポイント数: {len(pred_values)}")
        print(f"予測値範囲: {min(pred_values):.0f} - {max(pred_values):.0f}")
        print(f"予測値: {[int(v) for v in pred_values[:5]]}... (最初の5個)")
        
        # 直線性チェック
        if len(pred_values) > 2:
            differences = [abs(pred_values[i+1] - pred_values[i]) for i in range(len(pred_values)-1)]
            avg_diff = sum(differences) / len(differences)
            max_diff = max(differences)
            
            print(f"予測の平均変化: {avg_diff:.2f}")
            print(f"予測の最大変化: {max_diff:.2f}")
            
            # 元データとの比較
            orig_differences = [abs(norm_values[i+1] - norm_values[i]) for i in range(len(norm_values)-1)]
            orig_avg_diff = sum(orig_differences) / len(orig_differences)
            
            print(f"元データの平均変化: {orig_avg_diff:.2f}")
            print(f"変動比率: {avg_diff / orig_avg_diff:.3f}")
            
            # 直線性の判定
            if max_diff < 50:
                print("⚠️  予測が非常に平坦（ほぼ直線）")
            elif avg_diff < orig_avg_diff * 0.1:
                print("⚠️  予測の変動が元データより大幅に小さい")
            else:
                print("✅ 予測に適度な変動がある")
        
        # メタデータ確認
        print(f"\n--- STEP 4: メタデータ確認 ---")
        print(f"メタデータ: {json.dumps(metadata, indent=2, default=str)}")
        
        return {
            'original_data': values,
            'normalized_data': norm_values,
            'predictions': pred_values,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_data_scenarios():
    """
    異なるデータシナリオで直線化の原因を特定
    """
    print(f"\n{'='*60}")
    print("=== 異なるデータシナリオでのテスト ===")
    
    scenarios = [
        {
            'name': '単調増加データ',
            'data': [36000 + i * 200 for i in range(20)]
        },
        {
            'name': '高変動データ',
            'data': [36000 + (i % 4 - 2) * 1000 + i * 50 for i in range(20)]
        },
        {
            'name': 'ランダムウォーク',
            'data': [36000] + [36000 + sum((j % 7 - 3) * 100 for j in range(i+1)) for i in range(19)]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        
        try:
            from src.models.predictor import TimeSeriesPredictor
            
            # タイムスタンプ作成
            base_time = datetime.datetime(2025, 6, 14, 0, 0, 0)
            timestamps = [base_time + datetime.timedelta(hours=i) for i in range(len(scenario['data']))]
            
            predictor = TimeSeriesPredictor()
            
            pred_timestamps, pred_values, metadata = predictor.zero_shot_predict(
                timestamp=timestamps,
                values=scenario['data'],
                horizon=12
            )
            
            # 変動性分析
            if len(pred_values) > 1:
                pred_var = sum((pred_values[i+1] - pred_values[i])**2 for i in range(len(pred_values)-1))
                orig_var = sum((scenario['data'][i+1] - scenario['data'][i])**2 for i in range(len(scenario['data'])-1))
                
                print(f"  予測ポイント: {len(pred_values)}")
                print(f"  予測変動: {pred_var:.0f}")
                print(f"  元データ変動: {orig_var:.0f}")
                print(f"  変動比: {pred_var / orig_var:.3f}" if orig_var > 0 else "  変動比: N/A")
                
                if pred_var < orig_var * 0.01:
                    print("  ⚠️  予測が元データより大幅に平坦")
                
        except Exception as e:
            print(f"  エラー: {e}")

def main():
    """
    メイン実行関数
    """
    print("実際のAPI処理フローの詳細調査")
    print("="*60)
    
    # 1. 実際のAPI処理フローをトレース
    result = trace_api_flow()
    
    # 2. 異なるデータシナリオでテスト
    test_different_data_scenarios()
    
    print(f"\n{'='*60}")
    print("調査完了")
    print("\n判明した可能性:")
    print("1. 正規化処理での変動の損失")
    print("2. AutoGluonモデルの設定問題")
    print("3. 予測アルゴリズム自体の限界")
    print("4. データの特性に対するモデルの不適合")

if __name__ == "__main__":
    main()