#!/usr/bin/env python3
"""
predict_zero_shot の簡単なテスト実行スクリプト
"""

import sys
import types
import datetime
import json
from unittest.mock import Mock, patch

# autogluonライブラリのモック設定
def setup_mocks():
    # autogluonモックの設定
    mock_autogluon = types.ModuleType('autogluon')
    mock_timeseries = types.ModuleType('autogluon.timeseries')
    
    class MockTimeSeriesDataFrame:
        def __init__(self, *args, **kwargs):
            pass
    
    class MockTimeSeriesPredictor:
        def __init__(self, *args, **kwargs):
            pass
        
        def fit(self, *args, **kwargs):
            return self
        
        def predict(self, *args, **kwargs):
            class MockResult:
                def __init__(self):
                    # デフォルトの予測値を設定
                    self.mean = [10.0, 10.5, 11.0, 11.5, 12.0]
            return MockResult()
    
    mock_timeseries.TimeSeriesDataFrame = MockTimeSeriesDataFrame
    mock_timeseries.TimeSeriesPredictor = MockTimeSeriesPredictor
    mock_autogluon.timeseries = mock_timeseries
    
    sys.modules['autogluon'] = mock_autogluon
    sys.modules['autogluon.timeseries'] = mock_timeseries

def test_basic_functionality():
    """基本的なテスト"""
    setup_mocks()
    
    # テストクライアントを作成
    from fastapi.testclient import TestClient
    from src.api.server import app
    
    client = TestClient(app)
    
    # テストデータの準備
    now = datetime.datetime(2023, 1, 1, 12, 0, 0)
    timestamps = [
        (now - datetime.timedelta(hours=i)).isoformat() 
        for i in range(23, -1, -1)
    ]
    values = [10.0 + i * 0.1 for i in range(24)]
    forecast_until = (now + datetime.timedelta(hours=12)).isoformat()

    request_data = {
        "timestamp": timestamps,
        "values": values,
        "forecast_until": forecast_until,
        "model_name": "chronos_default"
    }

    print("テストデータ:")
    print(f"  - データポイント数: {len(timestamps)}")
    print(f"  - 予測期間: {forecast_until}")
    print(f"  - モデル名: {request_data['model_name']}")
    
    # APIエンドポイントをテスト
    print("\nAPIエンドポイントをテスト中...")
    try:
        response = client.post("/api/v1/predict_zero_shot", json=request_data)
        print(f"レスポンス ステータス: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ テスト成功!")
            print(f"  - 予測タイムスタンプ数: {len(data.get('forecast_timestamp', []))}")
            print(f"  - 予測値数: {len(data.get('forecast_values', []))}")
            print(f"  - モデル名: {data.get('model_name')}")
            
            # 信頼区間の確認
            if data.get('confidence_intervals'):
                print("  - 信頼区間: あり")
            
            # 評価指標の確認
            if data.get('metrics'):
                print("  - 評価指標: あり")
            
            return True
        else:
            print("❌ テスト失敗!")
            print(f"エラー詳細: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ テスト中にエラーが発生: {e}")
        return False

def test_error_cases():
    """エラーケースのテスト"""
    print("\n=== エラーケースのテスト ===")
    
    from fastapi.testclient import TestClient
    from src.api.server import app
    
    client = TestClient(app)
    
    # データポイント不足のテスト
    print("データポイント不足のテスト...")
    now = datetime.datetime(2023, 1, 1, 12, 0, 0)
    insufficient_data = {
        "timestamp": [now.isoformat()],
        "values": [10.0],
        "forecast_until": (now + datetime.timedelta(hours=1)).isoformat(),
        "model_name": "chronos_default"
    }
    
    try:
        response = client.post("/api/v1/predict_zero_shot", json=insufficient_data)
        if response.status_code == 400:
            print("✅ データポイント不足エラーが正しく検出されました")
        else:
            print(f"❌ 期待したエラーが発生しませんでした (status: {response.status_code})")
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")

if __name__ == "__main__":
    print("=== predict_zero_shot 基本テスト ===")
    
    # 基本機能のテスト
    success = test_basic_functionality()
    
    if success:
        # エラーケースのテスト
        test_error_cases()
        print("\n🎉 すべてのテストが完了しました!")
    else:
        print("\n💥 基本テストが失敗したため、他のテストをスキップします。")
        sys.exit(1)