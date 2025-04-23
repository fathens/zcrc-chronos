"""
chronos-boltライブラリのモック
"""

class ZeroShotPredictor:
    def __init__(self, params=None):
        self.params = params or {}
    
    def predict(self, context=None, horizon=24):
        # 予測結果のモック
        class MockResult:
            def __init__(self):
                self.values = [10.0 + i * 0.1 for i in range(horizon)]
                self.confidence_intervals = {
                    'lower_95': [v - 1.0 for v in self.values],
                    'upper_95': [v + 1.0 for v in self.values]
                }
                self.metrics = {
                    'mse': 0.15,
                    'mae': 0.12
                }
        
        return MockResult()

class AutoGluonPredictor:
    def __init__(self, params=None):
        self.params = params or {}
    
    def predict(self, df, horizon=24):
        # 予測結果のモック
        class MockResult:
            def __init__(self):
                self.values = [10.0 + i * 0.1 for i in range(horizon)]
                self.confidence_intervals = {
                    'lower_95': [v - 1.0 for v in self.values],
                    'upper_95': [v + 1.0 for v in self.values]
                }
                self.metrics = {
                    'mse': 0.15,
                    'mae': 0.12
                }
        
        return MockResult()