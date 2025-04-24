"""
AutoGluon-TimeSeries と Chronos-Bolt のモック
"""
import pandas as pd

class TimeSeriesDataFrame:
    """TimeSeriesDataFrameのモック"""
    def __init__(self, df=None, id_column=None, timestamp_column=None):
        self.df = df
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        # DataFrameの有無を適切に判定
        self.item_ids = ['dummy_id'] if df is None or df.empty else df[id_column].unique().tolist()

class TimeSeriesPredictor:
    """TimeSeriesPredictorのモック"""
    def __init__(self, prediction_length=24, eval_metric=None):
        """
        AutoGluon-TimeSeries の TimeSeriesPredictor クラスのモック
        公式の実装では prediction_length は必須パラメータ
        
        Args:
            prediction_length: 予測期間
            eval_metric: 評価指標
        """
        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.presets = None
        self.data = None
        self.verbosity = 2

    def fit(self, data, presets=None, verbosity=2):
        """フィットのモック。Chronos-Boltでは主に設定用"""
        self.data = data
        self.presets = presets
        self.verbosity = verbosity
        return self

    def predict(self, data):
        """予測のモック"""
        # 予測結果のモック
        class MockResult(dict):
            def __init__(self, pred_length):
                for item_id in data.item_ids:
                    self[item_id] = self.MockItemResult(pred_length)

            class MockItemResult:
                def __init__(self, pred_length):
                    # prediction_lengthパラメータに基づいて可変長の予測値を生成
                    self.values = pd.Series([10.0 + i * 0.1 for i in range(pred_length)])
                    
                    # lower_95とupper_95も適切な長さで生成
                    values_list = self.values.tolist()
                    self.confidence_intervals = {
                        'lower_95': [v - 1.0 for v in values_list],
                        'upper_95': [v + 1.0 for v in values_list]
                    }
                    
                    # オプションでメトリクスも追加
                    self.metrics = {
                        'mse': 0.15,
                        'mae': 0.12
                    }

        return MockResult(self.prediction_length)
