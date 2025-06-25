#!/usr/bin/env python3
"""
predict_zero_shot テストの実行スクリプト
autogluonライブラリをモックして実行
"""

import sys
import types


# autogluonライブラリのモック設定
def setup_autogluon_mock():
    # モックモジュールの作成
    mock_autogluon = types.ModuleType("autogluon")
    mock_timeseries = types.ModuleType("autogluon.timeseries")

    # TimeSeriesDataFrameクラスのモック
    class MockTimeSeriesDataFrame:
        def __init__(self, *args, **kwargs):
            pass

    # TimeSeriesPredictorクラスのモック
    class MockTimeSeriesPredictor:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            # モックの予測結果を返す
            class MockResult:
                def __init__(self):
                    self.mean = [10.0, 10.5, 11.0]  # デフォルトの予測値

            return MockResult()

    # モックモジュールにクラスを追加
    mock_timeseries.TimeSeriesDataFrame = MockTimeSeriesDataFrame
    mock_timeseries.TimeSeriesPredictor = MockTimeSeriesPredictor
    mock_autogluon.timeseries = mock_timeseries

    # sys.modulesにモックモジュールを追加
    sys.modules["autogluon"] = mock_autogluon
    sys.modules["autogluon.timeseries"] = mock_timeseries


if __name__ == "__main__":
    # モックを適用
    setup_autogluon_mock()

    # テストを実行
    import os
    import subprocess

    # 環境変数でモックを有効にする
    env = os.environ.copy()
    env["USE_MOCK_AUTOGLUON"] = "true"
    env["PYTHONPATH"] = "."

    # pytestを実行
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                (
                    "tests/test_predict_zero_shot.py::"
                    "TestPredictZeroShotValidInputs::test_basic_prediction_success"
                ),
                "-v",
                "--tb=short",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

    except Exception as e:
        print(f"Error running test: {e}")
