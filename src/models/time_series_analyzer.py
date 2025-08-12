"""
時系列データ特性分析モジュール
データ特性に基づく動的モデル選択と階層的学習をサポート
"""

import datetime
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


class TimeSeriesCharacteristics:
    """時系列データの特性を表すデータクラス"""

    def __init__(
        self,
        trend=None,
        seasonality=None,
        volatility=None,
        stationarity=None,
        frequency=None,
        missing_pattern=None,
        density=None,
        outliers=None,
    ):
        self.trend = trend or {}
        self.seasonality = seasonality or {}
        self.volatility = volatility or {}
        self.stationarity = stationarity or {}
        self.frequency = frequency or {}
        self.missing_pattern = missing_pattern or {}
        self.density = density or {}
        self.outliers = outliers or {}


class TimeSeriesAnalyzer:
    """時系列データの特性分析クラス"""

    def __init__(self):
        self.characteristics = None

    def analyze_time_series_characteristics(
        self, values: List[float], timestamps: List[datetime.datetime]
    ) -> TimeSeriesCharacteristics:
        """
        時系列データの包括的特性分析

        Args:
            values: 時系列の値
            timestamps: 対応するタイムスタンプ

        Returns:
            TimeSeriesCharacteristics: 分析結果
        """
        logger.info("時系列データ特性分析を開始します")

        characteristics = TimeSeriesCharacteristics()

        # 基本的な検証
        if len(values) != len(timestamps) or len(values) < 3:
            logger.warning("データが不十分です。基本分析のみ実行")
            return self._basic_characteristics(values, timestamps, characteristics)

        try:
            # 1. 周期性検出
            characteristics.seasonality = self._detect_seasonality(values, timestamps)

            # 2. トレンド分析
            characteristics.trend = self._analyze_trend(values, timestamps)

            # 3. ボラティリティ測定
            characteristics.volatility = self._calculate_volatility(values)

            # 4. 欠損・外れ値パターン
            characteristics.missing_pattern = self._analyze_missing_data(
                timestamps, values
            )
            characteristics.outliers = self._detect_outliers(values)

            # 5. データ密度分析
            characteristics.density = self._analyze_time_intervals(timestamps)

            # 6. 定常性検定
            characteristics.stationarity = self._test_stationarity(values)

            # 7. 頻度推定
            characteristics.frequency = self._estimate_frequency(timestamps)

            logger.info("時系列データ特性分析が完了しました")

        except Exception as e:
            logger.warning(f"詳細分析でエラー: {e}。基本分析に切り替えます")
            return self._basic_characteristics(values, timestamps, characteristics)

        return characteristics

    def _basic_characteristics(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        characteristics: TimeSeriesCharacteristics,
    ) -> TimeSeriesCharacteristics:
        """基本的な特性分析（エラー時のフォールバック）"""
        try:
            characteristics.volatility = (
                np.std(values) / np.mean(np.abs(values)) if len(values) > 1 else 0.0
            )
            characteristics.trend = {"strength": "unknown", "direction": "unknown"}
            characteristics.seasonality = {"strength": "unknown", "period": None}
            characteristics.density = {"regular": len(values) > 10}
            characteristics.stationarity = {"is_stationary": "unknown"}
            characteristics.missing_pattern = {"has_gaps": False}
            characteristics.outliers = {"count": 0, "percentage": 0.0}
        except Exception:
            logger.error("基本分析も失敗しました")

        return characteristics

    def _detect_seasonality(
        self, values: List[float], timestamps: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """周期性検出（FFT + 自己相関）"""
        try:
            if len(values) < 10:
                return {"strength": "weak", "period": None, "score": 0.0}

            # FFT による周期検出
            n = len(values)
            fft_vals = fft(values - np.mean(values))
            frequencies = fftfreq(n)

            # 主要な周波数成分を検出
            power = np.abs(fft_vals) ** 2
            peak_indices, _ = find_peaks(power[1 : n // 2], height=np.max(power) * 0.1)

            if len(peak_indices) > 0:
                # 最も強い周期を特定
                main_freq_idx = peak_indices[np.argmax(power[peak_indices + 1])] + 1
                period = (
                    int(1 / abs(frequencies[main_freq_idx]))
                    if frequencies[main_freq_idx] != 0
                    else None
                )

                # 周期性の強度評価
                strength_score = power[main_freq_idx] / np.sum(power)

                if strength_score > 0.3:
                    strength = "strong"
                elif strength_score > 0.1:
                    strength = "moderate"
                else:
                    strength = "weak"

                logger.debug(
                    f"周期性検出: period={period}, strength={strength}, "
                    f"score={strength_score:.3f}"
                )

                return {
                    "strength": strength,
                    "period": period,
                    "score": float(strength_score),
                    "dominant_frequency": float(frequencies[main_freq_idx]),
                }

            return {"strength": "weak", "period": None, "score": 0.0}

        except Exception as e:
            logger.warning(f"周期性検出でエラー: {e}")
            return {"strength": "unknown", "period": None, "score": 0.0}

    def _analyze_trend(
        self, values: List[float], timestamps: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """トレンド分析（線形回帰 + Mann-Kendall検定）"""
        try:
            if len(values) < 3:
                return {"strength": "weak", "direction": "none", "slope": 0.0}

            # 線形回帰によるトレンド検出
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            # トレンドの強度評価（決定係数）
            r_squared = r_value**2

            if r_squared > 0.7:
                strength = "strong"
            elif r_squared > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            # トレンドの方向
            if abs(slope) < std_err:
                direction = "none"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            # Mann-Kendall 検定による確認
            mk_trend = self._mann_kendall_test(values)

            logger.debug(
                f"トレンド分析: strength={strength}, direction={direction}, "
                f"r²={r_squared:.3f}"
            )

            return {
                "strength": strength,
                "direction": direction,
                "slope": float(slope),
                "r_squared": float(r_squared),
                "p_value": float(p_value),
                "mann_kendall": mk_trend,
            }

        except Exception as e:
            logger.warning(f"トレンド分析でエラー: {e}")
            return {"strength": "unknown", "direction": "unknown", "slope": 0.0}

    def _mann_kendall_test(self, values: List[float]) -> Dict[str, Any]:
        """Mann-Kendall 傾向検定"""
        try:
            n = len(values)
            if n < 3:
                return {"trend": "unknown", "p_value": 1.0}

            # S統計量の計算
            S = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if values[j] > values[i]:
                        S += 1
                    elif values[j] < values[i]:
                        S -= 1

            # 分散の計算
            var_s = n * (n - 1) * (2 * n + 5) / 18

            # 標準化
            if S > 0:
                z = (S - 1) / np.sqrt(var_s)
            elif S < 0:
                z = (S + 1) / np.sqrt(var_s)
            else:
                z = 0

            # p値の計算
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            if p_value < 0.05:
                if S > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            else:
                trend = "none"

            return {"trend": trend, "p_value": float(p_value), "S": S}

        except Exception:
            return {"trend": "unknown", "p_value": 1.0}

    def _calculate_volatility(self, values: List[float]) -> float:
        """ボラティリティ計算（変動係数）"""
        try:
            if len(values) < 2:
                return 0.0

            # 差分の標準偏差を使用（より安定）
            diffs = np.diff(values)
            if len(diffs) == 0:
                return 0.0

            mean_val = np.mean(values)
            if mean_val == 0:
                return np.inf

            volatility = np.std(diffs) / abs(mean_val)

            logger.debug(f"ボラティリティ: {volatility:.4f}")
            return float(volatility)

        except Exception as e:
            logger.warning(f"ボラティリティ計算でエラー: {e}")
            return 0.0

    def _analyze_missing_data(
        self, timestamps: List[datetime.datetime], values: List[float]
    ) -> Dict[str, Any]:
        """欠損データパターン分析"""
        try:
            if len(timestamps) < 2:
                return {"has_gaps": False, "gap_count": 0}

            # 期待される間隔を計算
            intervals = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]

            if not intervals:
                return {"has_gaps": False, "gap_count": 0}

            expected_interval = np.median(intervals)
            tolerance = expected_interval * 1.5  # 50%の許容範囲

            # ギャップの検出
            gaps = [interval for interval in intervals if interval > tolerance]
            gap_count = len(gaps)

            has_gaps = gap_count > 0
            gap_percentage = gap_count / len(intervals) * 100 if intervals else 0

            logger.debug(
                f"欠損分析: gaps={gap_count}, percentage={gap_percentage:.1f}%"
            )

            return {
                "has_gaps": has_gaps,
                "gap_count": gap_count,
                "gap_percentage": float(gap_percentage),
                "expected_interval": float(expected_interval),
                "max_gap": float(max(gaps)) if gaps else 0.0,
            }

        except Exception as e:
            logger.warning(f"欠損データ分析でエラー: {e}")
            return {"has_gaps": False, "gap_count": 0}

    def _detect_outliers(self, values: List[float]) -> Dict[str, Any]:
        """外れ値検出（IQR法 + Z-score法）"""
        try:
            if len(values) < 4:
                return {"count": 0, "percentage": 0.0, "indices": []}

            values_array = np.array(values)

            # IQR法
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1

            lower_bound = q1 - 3.0 * iqr  # より寛容な設定（3.0×IQR）
            upper_bound = q3 + 3.0 * iqr

            iqr_outliers = np.where(
                (values_array < lower_bound) | (values_array > upper_bound)
            )[0]

            # Z-score法（補完）
            z_scores = np.abs(stats.zscore(values_array))
            z_outliers = np.where(z_scores > 3.0)[0]  # より寛容な設定（3.0σ）

            # 両方で検出されたもののみを外れ値とする
            outlier_indices = np.intersect1d(iqr_outliers, z_outliers)
            outlier_count = len(outlier_indices)
            outlier_percentage = outlier_count / len(values) * 100

            logger.debug(
                f"外れ値検出: count={outlier_count}, percentage={outlier_percentage:.1f}%"
            )

            return {
                "count": int(outlier_count),
                "percentage": float(outlier_percentage),
                "indices": outlier_indices.tolist(),
                "method": "IQR_and_Z-score",
            }

        except Exception as e:
            logger.warning(f"外れ値検出でエラー: {e}")
            return {"count": 0, "percentage": 0.0, "indices": []}

    def _analyze_time_intervals(
        self, timestamps: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """時間間隔の規則性分析"""
        try:
            if len(timestamps) < 2:
                return {"regular": False, "interval_variance": 0.0}

            # 間隔の計算
            intervals = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]

            if not intervals:
                return {"regular": False, "interval_variance": 0.0}

            mean_interval = np.mean(intervals)
            interval_variance = np.var(intervals)

            # 規則性の判定（変動係数で評価）
            cv = (
                np.sqrt(interval_variance) / mean_interval
                if mean_interval > 0
                else np.inf
            )
            is_regular = cv < 0.1  # 変動係数10%未満で規則的

            logger.debug(f"時間間隔分析: regular={is_regular}, CV={cv:.3f}")

            return {
                "regular": is_regular,
                "mean_interval": float(mean_interval),
                "interval_variance": float(interval_variance),
                "coefficient_of_variation": float(cv),
            }

        except Exception as e:
            logger.warning(f"時間間隔分析でエラー: {e}")
            return {"regular": False, "interval_variance": 0.0}

    def _test_stationarity(self, values: List[float]) -> Dict[str, Any]:
        """定常性検定（簡易版ADF検定）"""
        try:
            if len(values) < 10:
                return {"is_stationary": "unknown", "p_value": 1.0}

            # 簡易的な定常性チェック（平均と分散の安定性）
            n = len(values)
            half = n // 2

            first_half = values[:half]
            second_half = values[half:]

            # 平均の差
            mean_diff = abs(np.mean(first_half) - np.mean(second_half))
            mean_threshold = np.std(values) * 0.5

            # 分散の比
            var1 = np.var(first_half)
            var2 = np.var(second_half)
            var_ratio = max(var1, var2) / max(min(var1, var2), 1e-10)

            # 判定
            is_stationary = (mean_diff < mean_threshold) and (var_ratio < 2.0)

            logger.debug(f"定常性検定: stationary={is_stationary}")

            return {
                "is_stationary": is_stationary,
                "mean_difference": float(mean_diff),
                "variance_ratio": float(var_ratio),
            }

        except Exception as e:
            logger.warning(f"定常性検定でエラー: {e}")
            return {"is_stationary": "unknown", "p_value": 1.0}

    def _estimate_frequency(
        self, timestamps: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """時系列の頻度を推定"""
        try:
            if len(timestamps) < 2:
                return {"estimated": "unknown", "confidence": 0.0}

            # タイムスタンプの差分を計算
            intervals = []
            for i in range(1, len(timestamps)):
                delta = timestamps[i] - timestamps[i - 1]
                intervals.append(delta.total_seconds())

            # 最も頻繁な間隔を特定
            median_interval = np.median(intervals)

            # 頻度を分類
            if median_interval <= 60:  # 1分以下
                freq = "min"
            elif median_interval <= 900:  # 15分以下
                freq = "15min"
            elif median_interval <= 1800:  # 30分以下
                freq = "30min"
            elif median_interval <= 3600:  # 1時間以下
                freq = "H"
            elif median_interval <= 86400:  # 1日以下
                freq = "D"
            else:
                freq = "irregular"

            # 信頼度を計算（間隔の一貫性）
            interval_std = np.std(intervals)
            cv = interval_std / median_interval if median_interval > 0 else 1.0
            confidence = max(0.0, 1.0 - cv)

            logger.debug(f"頻度推定: {freq}, 信頼度: {confidence:.3f}")

            return {
                "estimated": freq,
                "confidence": confidence,
                "median_interval_seconds": median_interval,
            }

        except Exception as e:
            logger.warning(f"頻度推定でエラー: {e}")
            return {"estimated": "unknown", "confidence": 0.0}

    def _detect_trend(self, values: List[float]) -> Dict[str, Any]:
        """線形トレンドの検出"""
        try:
            if len(values) < 3:
                return {"strength": "weak", "slope": 0.0, "r_squared": 0.0}

            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            r_squared = r_value**2

            # トレンドの強度分類
            if r_squared > 0.7:
                strength = "strong"
            elif r_squared > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            logger.debug(f"トレンド検出: {strength}, R²={r_squared:.3f}")

            return {
                "strength": strength,
                "slope": float(slope),
                "r_squared": float(r_squared),
                "p_value": float(p_value),
            }

        except Exception as e:
            logger.warning(f"トレンド検出でエラー: {e}")
            return {"strength": "weak", "slope": 0.0, "r_squared": 0.0}
