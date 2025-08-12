"""
TimeSeriesAnalyzerの単体テスト
"""

import datetime
import unittest

import numpy as np

from src.models.time_series_analyzer import (
    TimeSeriesAnalyzer,
    TimeSeriesCharacteristics,
)


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """TimeSeriesAnalyzerのテストクラス"""

    def setUp(self):
        """テストの前処理"""
        self.analyzer = TimeSeriesAnalyzer()

    def test_init(self):
        """初期化テスト"""
        analyzer = TimeSeriesAnalyzer()
        self.assertIsInstance(analyzer, TimeSeriesAnalyzer)

    def test_detect_trend_linear(self):
        """線形トレンド検出テスト"""
        # 線形増加データ
        values = [i * 2.0 + 10.0 for i in range(100)]

        trend = self.analyzer._detect_trend(values)

        self.assertEqual(trend["strength"], "strong")
        self.assertGreater(trend["r_squared"], 0.9)
        self.assertGreater(trend["slope"], 0)

    def test_detect_trend_no_trend(self):
        """トレンドなしデータのテスト"""
        # ランダムデータ（トレンドなし）
        np.random.seed(42)
        values = np.random.normal(10.0, 1.0, 100).tolist()

        trend = self.analyzer._detect_trend(values)

        self.assertEqual(trend["strength"], "weak")
        self.assertLess(trend["r_squared"], 0.3)

    def test_detect_trend_negative(self):
        """負のトレンド検出テスト"""
        # 線形減少データ
        values = [100.0 - i * 0.5 for i in range(100)]

        trend = self.analyzer._detect_trend(values)

        self.assertIn(trend["strength"], ["strong", "moderate"])
        self.assertLess(trend["slope"], 0)

    def test_detect_seasonality_daily(self):
        """日次季節性検出テスト"""
        # 日次周期のデータ（24時間周期）
        values = []
        for i in range(240):  # 10日分
            hour_of_day = i % 24
            # 昼間高く、夜間低い値
            value = 10.0 + 5.0 * np.sin(2 * np.pi * hour_of_day / 24)
            values.append(value)

        # 時間ベースのタイムスタンプ
        timestamps = [
            datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
            for i in range(240)
        ]

        seasonality = self.analyzer._detect_seasonality(values, timestamps)

        self.assertGreater(seasonality["score"], 0.3)

    def test_detect_seasonality_no_pattern(self):
        """季節性なしデータのテスト"""
        # ランダムデータ（季節性なし）
        np.random.seed(42)
        values = np.random.normal(10.0, 2.0, 100).tolist()
        timestamps = [
            datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
            for i in range(100)
        ]

        seasonality = self.analyzer._detect_seasonality(values, timestamps)

        self.assertEqual(seasonality["strength"], "weak")
        self.assertLess(seasonality["score"], 0.3)

    def test_calculate_volatility_low(self):
        """低ボラティリティテスト"""
        # 安定したデータ
        values = [10.0 + 0.1 * i for i in range(100)]

        volatility = self.analyzer._calculate_volatility(values)

        self.assertEqual(volatility["level"], "low")
        self.assertLess(volatility["cv"], 0.1)

    def test_calculate_volatility_high(self):
        """高ボラティリティテスト"""
        # 変動の大きいデータ
        np.random.seed(42)
        base_values = [10.0] * 100
        noise = np.random.normal(0, 5.0, 100)
        values = (np.array(base_values) + noise).tolist()

        volatility = self.analyzer._calculate_volatility(values)

        self.assertEqual(volatility["level"], "high")
        self.assertGreater(volatility["cv"], 0.3)

    def test_calculate_volatility_medium(self):
        """中程度ボラティリティテスト"""
        # 中程度の変動データ
        np.random.seed(42)
        base_values = [10.0] * 100
        noise = np.random.normal(0, 1.5, 100)
        values = (np.array(base_values) + noise).tolist()

        volatility = self.analyzer._calculate_volatility(values)

        self.assertEqual(volatility["level"], "medium")
        self.assertGreaterEqual(volatility["cv"], 0.1)
        self.assertLessEqual(volatility["cv"], 0.3)

    def test_test_stationarity_stationary(self):
        """定常性テスト（定常）"""
        # 定常データ（安定した値）
        np.random.seed(42)
        values = [10.0 + np.random.normal(0, 0.5) for _ in range(100)]

        stationarity = self.analyzer._test_stationarity(values)

        self.assertTrue(stationarity["is_stationary"])
        self.assertIn("mean_difference", stationarity)
        self.assertIn("variance_ratio", stationarity)

    def test_test_stationarity_non_stationary(self):
        """定常性テスト（非定常）"""
        # 非定常データ（強いトレンド）
        values = [i * 2.0 + 100.0 for i in range(100)]

        stationarity = self.analyzer._test_stationarity(values)

        self.assertFalse(stationarity["is_stationary"])
        self.assertIn("mean_difference", stationarity)
        self.assertIn("variance_ratio", stationarity)

    def test_estimate_frequency_hourly(self):
        """時間単位頻度の推定テスト"""
        # 1時間間隔のタイムスタンプ
        timestamps = [
            datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
            for i in range(100)
        ]

        frequency = self.analyzer._estimate_frequency(timestamps)

        self.assertEqual(frequency["estimated"], "H")
        self.assertGreater(frequency["confidence"], 0.8)

    def test_estimate_frequency_daily(self):
        """日単位頻度の推定テスト"""
        # 1日間隔のタイムスタンプ
        timestamps = [
            datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i)
            for i in range(50)
        ]

        frequency = self.analyzer._estimate_frequency(timestamps)

        self.assertEqual(frequency["estimated"], "D")
        self.assertGreater(frequency["confidence"], 0.8)

    def test_estimate_frequency_irregular(self):
        """不規則頻度の推定テスト"""
        # 不規則な間隔のタイムスタンプ
        timestamps = [
            datetime.datetime(2024, 1, 1)
            + datetime.timedelta(minutes=i * np.random.randint(30, 120))
            for i in range(50)
        ]

        frequency = self.analyzer._estimate_frequency(timestamps)

        # 不規則な場合は信頼度が低くなる
        self.assertLess(frequency["confidence"], 0.8)

    def test_analyze_time_series_characteristics_comprehensive(self):
        """包括的な時系列特性分析テスト"""
        # 複合的なデータを作成（トレンド + 季節性 + ノイズ）
        np.random.seed(42)
        values = []
        timestamps = []

        for i in range(240):  # 10日分
            timestamp = datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
            timestamps.append(timestamp)

            # トレンド成分
            trend = i * 0.01

            # 日次季節性成分
            seasonal = 2.0 * np.sin(2 * np.pi * (i % 24) / 24)

            # ノイズ
            noise = np.random.normal(0, 0.5)

            value = 10.0 + trend + seasonal + noise
            values.append(value)

        # 分析実行
        characteristics = self.analyzer.analyze_time_series_characteristics(
            values, timestamps
        )

        # 結果の検証
        self.assertIsInstance(characteristics, TimeSeriesCharacteristics)
        self.assertIsNotNone(characteristics.trend)
        self.assertIsNotNone(characteristics.seasonality)
        self.assertIsNotNone(characteristics.volatility)
        self.assertIsNotNone(characteristics.stationarity)
        self.assertIsNotNone(characteristics.frequency)

        # トレンドが検出されることを期待
        self.assertIn(characteristics.trend["strength"], ["moderate", "strong"])

        # 季節性が検出されることを期待
        self.assertGreater(characteristics.seasonality["score"], 0)

    def test_analyze_time_series_characteristics_empty(self):
        """空データの分析テスト"""
        with self.assertRaises(ValueError):
            self.analyzer.analyze_time_series_characteristics([], [])

    def test_analyze_time_series_characteristics_mismatched_length(self):
        """データ長不一致の分析テスト"""
        values = [1.0, 2.0, 3.0]
        timestamps = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]

        with self.assertRaises(ValueError):
            self.analyzer.analyze_time_series_characteristics(values, timestamps)

    def test_analyze_time_series_characteristics_short_data(self):
        """短いデータの分析テスト"""
        values = [1.0, 2.0]
        timestamps = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]

        with self.assertRaises(ValueError):
            self.analyzer.analyze_time_series_characteristics(values, timestamps)

    def test_time_series_characteristics_dataclass(self):
        """TimeSeriesCharacteristicsデータクラスのテスト"""
        characteristics = TimeSeriesCharacteristics(
            trend={"strength": "strong", "slope": 0.5, "r_squared": 0.9},
            seasonality={"strength": "moderate", "score": 0.4, "period": 24},
            volatility={"cv": 0.2, "level": "medium", "std": 1.5},
            stationarity={"is_stationary": True, "adf_pvalue": 0.01},
            frequency={"estimated": "H", "confidence": 0.95},
        )

        self.assertEqual(characteristics.trend["strength"], "strong")
        self.assertEqual(characteristics.seasonality["score"], 0.4)
        self.assertEqual(characteristics.volatility["level"], "medium")
        self.assertTrue(characteristics.stationarity["is_stationary"])
        self.assertEqual(characteristics.frequency["estimated"], "H")


if __name__ == "__main__":
    unittest.main()
