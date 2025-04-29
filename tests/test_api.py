"""
APIエンドポイントのテストモジュール
"""

import datetime
import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

from src.api.routes import normalize_time_series_data, _determine_best_interpolation_method
from src.api.server import app

# テストクライアントの初期化
client = TestClient(app)


def test_root_endpoint():
    """
    ルートエンドポイントのテスト
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "description" in data
    assert "docs" in data


def test_health_endpoint():
    """
    ヘルスチェックエンドポイントのテスト
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_version" in data


def test_models_endpoint():
    """
    モデル一覧エンドポイントのテスト
    """
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    if len(models) > 0:
        model = models[0]
        assert "name" in model
        assert "version" in model
        assert "description" in model
        assert "parameters" in model


def test_zero_shot_predict_endpoint():
    """
    ゼロショット予測エンドポイントのテスト
    """
    # テスト用のリクエストデータ
    now = datetime.datetime.now()
    # 過去24時間分のダミー時系列データを生成
    timestamps = [
        (now - datetime.timedelta(hours=i)).isoformat() for i in range(24, 0, -1)
    ]
    values = [10.0 + i * 0.1 for i in range(24)]

    request_data = {
        "timestamp": timestamps,
        "values": values,
        "horizon": 12,
        "model_name": "chronos_default",
    }

    response = client.post("/api/v1/predict_zero_shot", json=request_data)
    assert response.status_code == 200
    data = response.json()

    # レスポンスの検証
    assert "forecast_timestamp" in data
    assert "forecast_values" in data
    assert "model_name" in data
    assert "confidence_intervals" in data
    assert "metrics" in data

    # 予測値の数が指定したhorizonと一致することを確認
    assert len(data["forecast_timestamp"]) == request_data["horizon"]
    assert len(data["forecast_values"]) == request_data["horizon"]


def test_zero_shot_predict_endpoint_invalid_data():
    """
    ゼロショット予測エンドポイントの無効なデータに対するテスト
    """
    # timestampとvaluesの長さが一致しない無効なデータ
    now = datetime.datetime.now()
    timestamps = [
        (now - datetime.timedelta(hours=i)).isoformat() for i in range(5, 0, -1)
    ]
    values = [10.0, 11.0, 12.0]  # 長さが一致しない

    invalid_data = {
        "timestamp": timestamps,
        "values": values,
        "horizon": 12,
        "model_name": "chronos_default",
    }

    response = client.post("/api/v1/predict_zero_shot", json=invalid_data)
    # バリデーションエラーが発生することを期待
    assert response.status_code == 422


def test_normalize_time_series_data_empty_input():
    """
    normalize_time_series_data 関数の空の入力に対するテスト
    """
    # 空のリストを入力した場合、同じ空のリストが返されることを検証
    timestamps, values = [], []
    normalized_timestamps, normalized_values = normalize_time_series_data(timestamps, values)
    assert normalized_timestamps == []
    assert normalized_values == []


def test_normalize_time_series_data_single_point():
    """
    normalize_time_series_data 関数の単一データポイントに対するテスト
    """
    # 単一のデータポイントの場合、同じ値が返されることを検証
    now = datetime.datetime.now()
    timestamps = [now]
    values = [42.0]
    
    normalized_timestamps, normalized_values = normalize_time_series_data(timestamps, values)
    assert normalized_timestamps == timestamps
    assert normalized_values == values


def test_normalize_time_series_data_regular_intervals():
    """
    normalize_time_series_data 関数の等間隔データに対するテスト
    """
    # 等間隔のデータを作成
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i) for i in range(5, 0, -1)]
    values = [10.0, 11.0, 12.0, 13.0, 14.0]
    
    # 等間隔データに対して正規化を実行
    normalized_timestamps, normalized_values = normalize_time_series_data(timestamps, values, interpolation_method="linear")
    
    # 正規化後も値が保持されていることを検証（ほぼ同じ間隔を持つデータなので、補間後も元の値に近い）
    assert len(normalized_timestamps) >= len(timestamps)
    
    # 元のデータポイントに対応するタイムスタンプで値を検証
    df = pd.DataFrame({
        'timestamp': normalized_timestamps,
        'value': normalized_values
    }).set_index('timestamp')
    
    for i, timestamp in enumerate(timestamps):
        # 最も近い正規化されたタイムスタンプでの値を検証
        closest_idx = np.abs([(t - timestamp).total_seconds() for t in normalized_timestamps]).argmin()
        assert abs(normalized_values[closest_idx] - values[i]) < 0.5  # 許容誤差内


def test_normalize_time_series_data_irregular_intervals():
    """
    normalize_time_series_data 関数の不規則な間隔のデータに対するテスト
    """
    # 不規則な間隔のデータを作成
    now = datetime.datetime.now()
    timestamps = [
        now - datetime.timedelta(hours=12),
        now - datetime.timedelta(hours=8),  # 4時間の間隔
        now - datetime.timedelta(hours=7),  # 1時間の間隔
        now - datetime.timedelta(hours=3),  # 4時間の間隔
        now,                                # 3時間の間隔
    ]
    values = [10.0, 15.0, 17.0, 20.0, 25.0]
    
    # 不規則間隔データに対して正規化を実行
    normalized_timestamps, normalized_values = normalize_time_series_data(timestamps, values)
    
    # 正規化後のデータ点が元のデータ点以上で、均等な間隔になっていることを検証
    assert len(normalized_timestamps) >= len(timestamps)
    
    # 均等な間隔かどうかを検証
    time_diffs = [(normalized_timestamps[i+1] - normalized_timestamps[i]).total_seconds() 
                 for i in range(len(normalized_timestamps)-1)]
    assert max(time_diffs) - min(time_diffs) < 1.0  # 時間間隔の差が1秒未満
    
    # 元のタイムスタンプでの値が保持されていることを検証
    df = pd.DataFrame({
        'timestamp': normalized_timestamps,
        'value': normalized_values
    }).set_index('timestamp')
    
    # 開始と終了の値が保持されていることを検証
    # 最初と最後の時刻に最も近い正規化された値を検索し比較
    first_idx = np.abs([(t - timestamps[0]).total_seconds() for t in normalized_timestamps]).argmin()
    last_idx = np.abs([(t - timestamps[-1]).total_seconds() for t in normalized_timestamps]).argmin()
    
    assert abs(normalized_values[first_idx] - values[0]) < 0.5
    assert abs(normalized_values[last_idx] - values[-1]) < 0.5


def test_normalize_time_series_data_interpolation_methods():
    """
    normalize_time_series_data 関数の異なる補間方法に対するテスト
    """
    # テスト用のデータを作成
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(hours=i*2) for i in range(5, 0, -1)]
    values = [10.0, 20.0, 15.0, 30.0, 25.0]
    
    # 異なる補間方法でテスト
    methods = ["linear", "cubic", "nearest", "time", "quadratic"]
    
    for method in methods:
        normalized_timestamps, normalized_values = normalize_time_series_data(
            timestamps, values, interpolation_method=method
        )
        
        # 正規化されたデータの検証
        assert len(normalized_timestamps) >= len(timestamps)
        assert len(normalized_values) == len(normalized_timestamps)
        
        # 開始と終了の値が保持されていることを検証
        first_idx = np.abs([(t - timestamps[0]).total_seconds() for t in normalized_timestamps]).argmin()
        last_idx = np.abs([(t - timestamps[-1]).total_seconds() for t in normalized_timestamps]).argmin()
        
        # 異なる補間方法でも、元のデータポイント付近の値は近似していることを確認
        assert abs(normalized_values[first_idx] - values[0]) < 1.0
        assert abs(normalized_values[last_idx] - values[-1]) < 1.0


def test_determine_best_interpolation_method():
    """
    _determine_best_interpolation_method 関数のテスト
    """
    now = datetime.datetime.now()
    
    # ケース1: 少数のデータ点 (2点) - linearが選択されるべき
    timestamps_few = [now - datetime.timedelta(hours=2), now]
    values_few = [10.0, 20.0]
    method = _determine_best_interpolation_method(timestamps_few, values_few)
    assert method == "linear"
    
    # ケース2: 不規則な時間間隔 (time補間が最適)
    timestamps_irregular = [
        now - datetime.timedelta(hours=10),
        now - datetime.timedelta(hours=8),  
        now - datetime.timedelta(hours=3),
        now - datetime.timedelta(minutes=30),
        now,
    ]
    values_irregular = [10.0, 15.0, 20.0, 25.0, 30.0]
    method = _determine_best_interpolation_method(timestamps_irregular, values_irregular)
    assert method == "time"
    
    # ケース3: 滑らかなデータ (quadraticが選択される)
    timestamps_smooth = [now - datetime.timedelta(hours=i) for i in range(10, 0, -1)]
    # 二次関数に従う滑らかなデータを生成 - 完全な二次関数なのでquadraticが最適
    values_smooth = [0.1 * (i ** 2) for i in range(10)]
    method = _determine_best_interpolation_method(timestamps_smooth, values_smooth)
    assert method == "quadratic"  # 二次関数データなので厳密にquadraticを期待
    
    # ケース4: 変動の大きいデータ (linearが最適)
    timestamps_volatile = [now - datetime.timedelta(hours=i) for i in range(10, 0, -1)]
    # ランダムな変動を持つデータを生成
    np.random.seed(42)  # 再現性のために固定シード
    values_volatile = [10 + np.random.normal(0, 5) for _ in range(10)]
    method = _determine_best_interpolation_method(timestamps_volatile, values_volatile)
    assert method == "linear"
    
    # ケース5: 外れ値を含むデータ (linearが最適)
    timestamps_outliers = [now - datetime.timedelta(hours=i) for i in range(10, 0, -1)]
    values_outliers = [10.0, 11.0, 12.0, 13.0, 50.0, 15.0, 16.0, 17.0, 18.0, 19.0]  # 50.0が外れ値
    method = _determine_best_interpolation_method(timestamps_outliers, values_outliers)
    assert method == "linear"
    
    # ケース6: 境界条件 - smoothness = 0.1, value_volatility = 0.15 (cubic/splineまたはquadraticが選択される可能性がある)
    timestamps_boundary1 = [now - datetime.timedelta(hours=i) for i in range(10, 0, -1)]
    # 滑らかさがちょうど閾値付近のデータを作成
    values_boundary1 = []
    for i in range(10):
        if i % 2 == 0:
            values_boundary1.append(10 + i * 0.2)  # 少し変動を持たせる
        else:
            values_boundary1.append(10 + i * 0.2 + 0.05)  # 微小な揺らぎを追加
    method = _determine_best_interpolation_method(timestamps_boundary1, values_boundary1)
    assert method in ["cubic", "spline", "quadratic"]
    
    # ケース7: 境界条件 - smoothness = 0.3 (linearが選択される)
    timestamps_boundary2 = [now - datetime.timedelta(hours=i) for i in range(10, 0, -1)]
    # 中程度の滑らかさを持つデータ
    values_boundary2 = [10.0]
    for i in range(1, 10):
        if i % 2 == 0:
            values_boundary2.append(values_boundary2[-1] + 0.5)  # 緩やかな上昇
        else:
            values_boundary2.append(values_boundary2[-1] - 0.2)  # 小さな下降
    method = _determine_best_interpolation_method(timestamps_boundary2, values_boundary2)
    assert method == "linear"
    
    # ケース8: エッジケース - すべての値が同じ (linearが選択される)
    timestamps_same = [now - datetime.timedelta(hours=i) for i in range(5, 0, -1)]
    values_same = [10.0] * 5  # すべて同じ値
    method = _determine_best_interpolation_method(timestamps_same, values_same)
    assert method == "linear"  # すべて同じ値の場合、効率的なlinearが選択される
    
    # ケース9: エッジケース - 極端に大きな値の差 (linearが選択されるべき)
    timestamps_extreme = [now - datetime.timedelta(hours=i) for i in range(5, 0, -1)]
    values_extreme = [10.0, 10000.0, 10.0, 10000.0, 10.0]  # 極端な値の差
    method = _determine_best_interpolation_method(timestamps_extreme, values_extreme)
    assert method == "linear"
    
    # ケース10: 特殊なタイムスタンプパターン - 等間隔だが非常に短い間隔
    timestamps_short = [
        now,
        now + datetime.timedelta(milliseconds=100),
        now + datetime.timedelta(milliseconds=200),
        now + datetime.timedelta(milliseconds=300),
        now + datetime.timedelta(milliseconds=400),
    ]
    values_short = [10.0, 11.0, 12.0, 13.0, 14.0]
    method = _determine_best_interpolation_method(timestamps_short, values_short)
    # 短い時間間隔でも正しい補間方法が選択されることを確認
    assert method in ["linear", "cubic"]


def test_determine_best_interpolation_method_edge_cases():
    """
    _determine_best_interpolation_method 関数のエッジケースに対するテスト
    """
    now = datetime.datetime.now()
    
    # ケース1: 最少データポイント (2点) - linearが選択されるべき
    timestamps_min = [now - datetime.timedelta(hours=2), now]
    values_min = [10.0, 20.0]
    method = _determine_best_interpolation_method(timestamps_min, values_min)
    assert method == "linear"
    
    # ケース2: 単一データポイント - linearが選択されるべき
    timestamps_single = [now]
    values_single = [10.0]
    method = _determine_best_interpolation_method(timestamps_single, values_single)
    assert method == "linear"
    
    # ケース3: 時間間隔がすべて同じ (変動係数=0)
    timestamps_equal = [now + datetime.timedelta(hours=i) for i in range(5)]
    values_equal = [10.0, 12.0, 14.0, 16.0, 18.0]  # 線形増加
    method = _determine_best_interpolation_method(timestamps_equal, values_equal)
    # 時間間隔が完全に等しい場合、時間間隔の変動係数は0で、timeは選択されない
    assert method != "time"
    
    # ケース4: 直線データ (二次差分がゼロに近い)
    timestamps_linear = [now + datetime.timedelta(hours=i) for i in range(10)]
    values_linear = [10.0 + i for i in range(10)]  # 完全な直線
    method = _determine_best_interpolation_method(timestamps_linear, values_linear)
    # 直線データの場合、滑らかさは高くlinearまたはcubicが選択される可能性がある
    assert method in ["linear", "cubic", "quadratic"]
    
    # ケース5: 交互に変化する値 (二次差分が大きい)
    timestamps_alternating = [now + datetime.timedelta(hours=i) for i in range(10)]
    values_alternating = [10.0 if i % 2 == 0 else 20.0 for i in range(10)]  # 10と20を交互に
    method = _determine_best_interpolation_method(timestamps_alternating, values_alternating)
    # 交互に変化する値は変動が大きいためlinearが選択される
    assert method == "linear"
    
    # ケース6: 値がすべて0のケース
    timestamps_zeros = [now + datetime.timedelta(hours=i) for i in range(5)]
    values_zeros = [0.0] * 5
    method = _determine_best_interpolation_method(timestamps_zeros, values_zeros)
    # すべての値が同じ場合linearが選択される
    assert method == "linear"  # すべて同じ値の場合、効率的なlinearが選択される
    
    # ケース7: 非常に大きな値と小さな値の混合
    timestamps_mixed = [now + datetime.timedelta(hours=i) for i in range(5)]
    values_mixed = [1e-6, 1e6, 1e-6, 1e6, 1e-6]  # 非常に大きな値と小さな値
    method = _determine_best_interpolation_method(timestamps_mixed, values_mixed)
    # 大きな変動があるためlinearが選択される
    assert method == "linear"


def test_auto_interpolation_method_selection():
    """
    自動補間方法選択機能のテスト
    """
    now = datetime.datetime.now()
    
    # 様々なタイプのデータセットに対して自動選択をテスト
    datasets = [
        # 少数のデータ点
        {
            "timestamps": [now - datetime.timedelta(hours=i) for i in range(3, 0, -1)],
            "values": [10.0, 11.0, 12.0],
        },
        # 不規則な時間間隔
        {
            "timestamps": [
                now - datetime.timedelta(hours=10),
                now - datetime.timedelta(hours=5),
                now - datetime.timedelta(hours=1),
            ],
            "values": [10.0, 15.0, 20.0],
        },
        # 滑らかなデータ
        {
            "timestamps": [now - datetime.timedelta(hours=i) for i in range(10, 0, -1)],
            "values": [i ** 2 for i in range(10)],
        },
    ]
    
    for dataset in datasets:
        # 自動選択で正規化を実行
        normalized_timestamps, normalized_values = normalize_time_series_data(
            dataset["timestamps"], dataset["values"], interpolation_method="auto"
        )
        
        # 正規化されたデータの検証
        assert len(normalized_timestamps) >= len(dataset["timestamps"])
        assert len(normalized_values) == len(normalized_timestamps)
        
        # 均等な間隔かどうかを検証
        time_diffs = [(normalized_timestamps[i+1] - normalized_timestamps[i]).total_seconds() 
                     for i in range(len(normalized_timestamps)-1)]
        assert max(time_diffs) - min(time_diffs) < 1.0  # 時間間隔の差が1秒未満
