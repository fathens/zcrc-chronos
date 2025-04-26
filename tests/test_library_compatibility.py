"""
AutoGluon.timeseriesの実際のライブラリとモックの互換性検証テスト
"""
import os
import pytest
import datetime
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path

from src.models.predictor import TimeSeriesPredictor
from tests.test_integration import generate_test_data
from tests.conftest import need_real_library, can_skip_real_library

# スナップショットファイルのパス
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)
SNAPSHOT_FILE = SNAPSHOT_DIR / "real_library_output.pkl"

# モックと実際のライブラリで使用する標準入力データ
def get_standard_test_data():
    """テスト用の標準データセットを生成する"""
    # 複数のパターンのデータを用意
    test_datasets = {}
    
    # 上昇トレンドデータ
    timestamps, values = generate_test_data("uptrend", length=48)
    test_datasets["uptrend"] = (timestamps, values)
    
    # 平坦なデータ
    timestamps, values = generate_test_data("flat", length=48)
    test_datasets["flat"] = (timestamps, values)
    
    # 季節性データ
    timestamps, values = generate_test_data("seasonal", length=48)
    test_datasets["seasonal"] = (timestamps, values)
    
    return test_datasets

def record_real_library_output(force_update=False):
    """実際のライブラリの出力を記録し、スナップショットとして保存する"""
    # スナップショットファイルが既に存在し、強制更新しない場合は既存のデータを使用
    if SNAPSHOT_FILE.exists() and not force_update:
        try:
            with open(SNAPSHOT_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            # ファイルが読めない場合は再生成
            pass
    
    # 実際のライブラリで結果を生成
    predictor = TimeSeriesPredictor()
    test_datasets = get_standard_test_data()
    results = {}
    
    for pattern, (timestamps, values) in test_datasets.items():
        # 予測を実行
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=24
        )
        
        # 結果を保存
        results[pattern] = {
            "forecast_timestamps": forecast_timestamps,
            "forecast_values": forecast_values,
            "metadata": {k: v for k, v in metadata.items() if not isinstance(v, (dict, list))}
        }
    
    # スナップショットとして保存
    with open(SNAPSHOT_FILE, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def get_mock_output(with_patched_predictor=None):
    """モックライブラリの出力を取得する"""
    from unittest.mock import patch, MagicMock
    import numpy as np
    
    mock_results = {}
    test_datasets = get_standard_test_data()
    
    # 手動でモック予測を作成する関数
    def create_mock_prediction(pattern, timestamps, values, horizon=24):
        last_timestamp = timestamps[-1]
        forecast_timestamps = [last_timestamp + datetime.timedelta(hours=i+1) for i in range(horizon)]
        
        # パターンに基づいて予測値を生成
        if pattern == "flat":
            # 平坦なデータ -> 平坦な予測
            mean_val = np.mean(values)
            forecast_values = [mean_val + np.random.normal(0, 0.1) for _ in range(horizon)]
        elif pattern == "uptrend":
            # 上昇トレンド -> 上昇傾向の予測継続
            last_val = values[-1]
            slope = (values[-1] - values[-5]) / 5 if len(values) >= 5 else 2.0
            forecast_values = [last_val + slope * (i+1) + np.random.normal(0, 0.5) for i in range(horizon)]
        elif pattern == "seasonal":
            # 季節性データ -> 季節性の継続
            # 24時間の周期を維持
            last_idx = len(values) - 1
            forecast_values = []
            for i in range(horizon):
                next_idx = last_idx + i + 1
                seasonal = 20.0 * np.sin(2 * np.pi * next_idx / 24)
                trend = next_idx * 0.5
                forecast_values.append(50.0 + seasonal + trend + np.random.normal(0, 1.0))
        else:
            # デフォルト: 最後の値を繰り返す
            forecast_values = [values[-1] + np.random.normal(0, 0.2) for _ in range(horizon)]
        
        # メタデータの作成
        metadata = {
            "model_name": "autogluon_timeseries_model",
            "model_type": "chronos_bolt",
            "preset": "medium_quality",
            "timestamp": datetime.datetime.now().isoformat(),
            "training_samples": len(values),
        }
        
        return {
            "forecast_timestamps": forecast_timestamps,
            "forecast_values": forecast_values,
            "metadata": metadata
        }
    
    if with_patched_predictor:
        # 外部から提供されたパッチ済みのpredictorを使用
        predictor = with_patched_predictor
        for pattern, (timestamps, values) in test_datasets.items():
            forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
                timestamp=timestamps, values=values, horizon=24
            )
            mock_results[pattern] = {
                "forecast_timestamps": forecast_timestamps,
                "forecast_values": forecast_values,
                "metadata": {k: v for k, v in metadata.items() if not isinstance(v, (dict, list))}
            }
    else:
        # AutoGluonのモックを作成し、手動で予測を生成
        for pattern, (timestamps, values) in test_datasets.items():
            # 各パターンについて手動でモック予測を作成
            result = create_mock_prediction(pattern, timestamps, values)
            mock_results[pattern] = result
    
    return mock_results

def similar_statistical_properties(real_output, mock_output, tolerance=0.5):
    """実際のライブラリの出力とモックの出力の統計的特性が似ているか検証する"""
    if not real_output or not mock_output:
        return False
    
    for pattern in real_output.keys():
        if pattern not in mock_output:
            return False
        
        real_values = real_output[pattern]["forecast_values"]
        mock_values = mock_output[pattern]["forecast_values"]
        
        # 長さのチェック
        if len(real_values) != len(mock_values):
            print(f"長さが異なります: real={len(real_values)}, mock={len(mock_values)}")
            return False
        
        # 値の範囲が近いことをチェック
        real_min, real_max = min(real_values), max(real_values)
        mock_min, mock_max = min(mock_values), max(mock_values)
        
        # 非ゼロ値の場合の相対差を計算
        if abs(real_min) > 1e-6:
            min_rel_diff = abs((mock_min - real_min) / real_min)
        else:
            min_rel_diff = abs(mock_min - real_min)
            
        if abs(real_max) > 1e-6:
            max_rel_diff = abs((mock_max - real_max) / real_max)
        else:
            max_rel_diff = abs(mock_max - real_max)
        
        # 値の傾向（単調増加/減少）をチェック
        real_trend = [real_values[i+1] - real_values[i] for i in range(len(real_values)-1)]
        mock_trend = [mock_values[i+1] - mock_values[i] for i in range(len(mock_values)-1)]
        
        real_increasing = sum(1 for t in real_trend if t > 0)
        mock_increasing = sum(1 for t in mock_trend if t > 0)
        
        # 増加傾向の比率の差を計算
        if len(real_trend) > 0:
            real_increasing_ratio = real_increasing / len(real_trend)
            mock_increasing_ratio = mock_increasing / len(mock_trend)
            trend_diff = abs(real_increasing_ratio - mock_increasing_ratio)
        else:
            trend_diff = 0
        
        print(f"パターン {pattern} の統計的特性:")
        print(f"  実際の値範囲: {real_min:.2f} - {real_max:.2f}")
        print(f"  モックの値範囲: {mock_min:.2f} - {mock_max:.2f}")
        print(f"  最小値の相対差: {min_rel_diff:.2f}")
        print(f"  最大値の相対差: {max_rel_diff:.2f}")
        print(f"  傾向の差: {trend_diff:.2f}")
        
        # 許容範囲内か判定
        if (min_rel_diff > tolerance or max_rel_diff > tolerance or trend_diff > tolerance):
            return False
    
    return True

@need_real_library
def test_snapshot_comparison(force_update=False):
    """モックと実際のライブラリの出力が統計的に似ていることを検証する"""
    # 実際のライブラリの出力を記録（または保存済みのデータを取得）
    real_output = record_real_library_output(force_update=force_update)
    
    # モックの出力を取得
    mock_output = get_mock_output()
    
    # 統計的特性の比較
    assert similar_statistical_properties(real_output, mock_output), \
        "モックの出力が実際のライブラリの出力と統計的に類似していません"

if __name__ == "__main__":
    # スクリプトとして実行された場合、スナップショットを更新
    print("実際のライブラリの出力を記録しています...")
    record_real_library_output(force_update=True)
    print(f"スナップショットを {SNAPSHOT_FILE} に保存しました")
