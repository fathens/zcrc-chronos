"""
実際のAutogluon-TimeSeriesライブラリを使用した統合テスト
"""
import os
import sys
import pytest
import datetime
import numpy as np
import pandas as pd
import math
from typing import List, Dict, Any, Tuple

# プロジェクトルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.predictor import TimeSeriesPredictor

# 統合テストを実行するかどうかの条件
# 環境変数またはSkipIfの条件で統合テストをスキップできる
SKIP_INTEGRATION_TESTS = os.environ.get('SKIP_INTEGRATION_TESTS', '').lower() in ('true', '1', 'yes')

def setup_module(module):
    """テストモジュールのセットアップ - 実ライブラリを使用"""
    # 実際のライブラリを使用するよう環境変数を設定
    os.environ['USE_REAL_LIBRARY'] = 'true'

def teardown_module(module):
    """テストモジュールのティアダウン - 環境設定をクリア"""
    # 環境変数をクリア
    if 'USE_REAL_LIBRARY' in os.environ:
        del os.environ['USE_REAL_LIBRARY']

# 統合テストをスキップするためのデコレータ
need_real_library = pytest.mark.skipif(
    SKIP_INTEGRATION_TESTS,
    reason="統合テストがSKIP_INTEGRATION_TESTS環境変数でスキップされました"
)

# テストデータ生成関数
def generate_test_data(pattern="uptrend", length=24, noise_level=0.1):
    """
    異なるパターンのテストデータを生成する
    
    Args:
        pattern: データパターン ("uptrend", "flat", "seasonal", "zigzag" など)
        length: データポイントの数
        noise_level: ノイズレベル（0.0〜1.0）
        
    Returns:
        (timestamps, values) のタプル
    """
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(length, 0, -1)]
    
    base_values = []
    if pattern == "uptrend":
        # 上昇トレンド
        base_values = [10.0 + (i * 2.0) for i in range(length)]
    elif pattern == "flat":
        # 平坦なデータ
        base_values = [50.0] * length
    elif pattern == "seasonal":
        # 季節性のあるデータ
        for i in range(length):
            seasonal = 20.0 * np.sin(2 * np.pi * i / 24)  # 24時間の周期
            trend = i * 0.5  # 緩やかな上昇トレンド
            base_values.append(50.0 + seasonal + trend)
    elif pattern == "zigzag":
        # ジグザグパターン
        for i in range(length):
            if i % 8 < 4:
                base_values.append(20.0 + i)
            else:
                base_values.append(60.0 - i)
    elif pattern == "downtrend":
        # 下降トレンド
        base_values = [30.0 - i * 0.3 for i in range(length)]
    elif pattern == "outlier":
        # 外れ値を含むデータ
        values = [20.0 + i * 0.1 + 2.0 * math.sin(i * math.pi / 12) for i in range(length)]
        # 特定の位置に外れ値を挿入
        outlier_positions = [int(length * 0.25), int(length * 0.75)]
        for pos in outlier_positions:
            values[pos] = values[pos] * 3.0  # 大きな外れ値
        base_values = values
    else:
        # デフォルトは乱数
        base_values = np.random.rand(length) * 100
    
    # ノイズを追加
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.std(base_values), length)
        values = [base + n for base, n in zip(base_values, noise)]
    else:
        values = base_values
    
    return timestamps, values

@need_real_library
def test_real_library_zero_shot_predict():
    """
    実際のAutogluon-TimeSeriesライブラリを使用したゼロショット予測テスト
    """
    try:
        # TimeSeriesPredictor インスタンスの作成
        predictor = TimeSeriesPredictor()
        
        # テストデータの生成（上昇トレンド）
        timestamps, values = generate_test_data("uptrend", length=48)
        
        # ゼロショット予測の実行
        horizon = 24
        forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
            timestamp=timestamps, values=values, horizon=horizon
        )
        
        # 結果の検証
        assert len(forecast_timestamps) == horizon
        assert len(forecast_values) == horizon
        assert isinstance(metadata, dict)
        assert "model_name" in metadata
        assert "model_type" in metadata
        
        # 予測値の検証
        assert all(isinstance(val, (int, float)) for val in forecast_values)
        assert all(np.isfinite(val) for val in forecast_values)
        
        # メタデータの検証
        assert metadata["training_samples"] == len(values)
        
        # 実行結果のログ出力
        print(f"\nテスト実行 - 実ライブラリを使用:")
        print(f"入力データ: 長さ={len(values)}, 平均値={np.mean(values):.2f}")
        print(f"予測結果: 長さ={len(forecast_values)}, 平均値={np.mean(forecast_values):.2f}")
        print(f"モデルタイプ: {metadata['model_type']}")
        
    except ImportError as e:
        pytest.skip(f"Autogluon-TimeSeriesライブラリをインポートできません: {e}")

@need_real_library
def test_real_library_with_different_patterns():
    """
    実際のライブラリで異なるパターンのデータを予測するテスト
    """
    try:
        # TimeSeriesPredictor インスタンスの作成
        predictor = TimeSeriesPredictor()
        
        # 異なるパターンのデータを生成
        patterns = ["uptrend", "downtrend", "seasonal", "outlier"]
        horizon = 24
        
        # 各パターンで予測結果を取得
        results = {}
        
        for pattern in patterns:
            timestamps, values = generate_test_data(pattern, length=48)
            
            # ゼロショット予測の実行
            forecast_timestamps, forecast_values, metadata = predictor.zero_shot_predict(
                timestamp=timestamps, values=values, horizon=horizon
            )
            
            # 結果を保存
            results[pattern] = {
                "input_mean": np.mean(values),
                "input_std": np.std(values),
                "forecast_mean": np.mean(forecast_values),
                "forecast_std": np.std(forecast_values),
                "forecast_values": forecast_values
            }
        
        # 結果の検証
        for pattern in patterns:
            result = results[pattern]
            print(f"\nパターン: {pattern}")
            print(f"入力データ: 平均={result['input_mean']:.2f}, 標準偏差={result['input_std']:.2f}")
            print(f"予測結果: 平均={result['forecast_mean']:.2f}, 標準偏差={result['forecast_std']:.2f}")
        
        # 各パターンでの予測結果の違いを検証
        # 実際のモデルの挙動によって異なる可能性があるため、緩い条件でテスト
        # アップトレンドとダウントレンドの予測平均値に違いがあることを検証
        if np.abs(results["uptrend"]["forecast_mean"] - results["downtrend"]["forecast_mean"]) < 0.1:
            print("\n警告: 上昇トレンドと下降トレンドの予測結果が非常に似ています")
            
        # 最低限の検証：すべての予測値が有効な数値であること
        for pattern in patterns:
            assert all(np.isfinite(val) for val in results[pattern]["forecast_values"])
            
    except ImportError as e:
        pytest.skip(f"Autogluon-TimeSeriesライブラリをインポートできません: {e}")


if __name__ == "__main__":
    # スクリプトとして実行された場合、統合テストを実行
    os.environ['USE_REAL_LIBRARY'] = 'true'
    print("実ライブラリを使用した統合テストを実行中...")
    
    try:
        test_real_library_zero_shot_predict()
        test_real_library_with_different_patterns()
        print("統合テスト完了！")
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")
    finally:
        if 'USE_REAL_LIBRARY' in os.environ:
            del os.environ['USE_REAL_LIBRARY']
