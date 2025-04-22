"""
時系列予測モデルのラッパーモジュール
"""
import os
import yaml
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger

# 設定ファイルのパス
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "model_config.yaml")

class TimeSeriesPredictor:
    """
    時系列予測モデルのラッパークラス
    chronos-boltライブラリを使用して時系列予測を行う
    """

    def __init__(self, model_name: str = "chronos_default", model_params: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            model_name: モデル名
            model_params: モデルパラメータ（設定ファイルの値を上書き）
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        モデル設定を読み込む

        Returns:
            モデル設定
        """
        try:
            with open(MODEL_CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"モデル設定の読み込みに失敗しました: {e}")
            raise ValueError(f"モデル設定の読み込みに失敗しました: {e}")

    def _prepare_data(self, timestamps: List[datetime.datetime], values: List[float]) -> pd.DataFrame:
        """
        予測用のデータを準備

        Args:
            timestamps: タイムスタンプのリスト
            values: 値のリスト

        Returns:
            pandas DataFrame
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        df.set_index('timestamp', inplace=True)

        # 前処理設定に基づいて前処理を適用
        preprocessing_config = self.config.get('preprocessing', {})

        # 欠損値の補完
        if preprocessing_config.get('impute_missing', False):
            method = preprocessing_config.get('imputation_method', 'linear')
            if method == 'linear':
                df = df.interpolate(method='linear')
            elif method == 'ffill':
                df = df.fillna(method='ffill')
            elif method == 'bfill':
                df = df.fillna(method='bfill')
            elif method == 'mean':
                df = df.fillna(df.mean())

        # 外れ値の検出と処理
        if preprocessing_config.get('outlier_detection', False):
            method = preprocessing_config.get('outlier_method', 'iqr')
            if method == 'iqr':
                Q1 = df['value'].quantile(0.25)
                Q3 = df['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df['value'] = df['value'].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'zscore':
                mean = df['value'].mean()
                std = df['value'].std()
                df['value'] = df['value'].clip(lower=mean-3*std, upper=mean+3*std)

        # 正規化
        if preprocessing_config.get('normalize', False):
            method = preprocessing_config.get('normalization_method', 'minmax')
            if method == 'minmax':
                self.min_val = df['value'].min()
                self.max_val = df['value'].max()
                if self.max_val > self.min_val:
                    df['value'] = (df['value'] - self.min_val) / (self.max_val - self.min_val)
            elif method == 'standard':
                self.mean_val = df['value'].mean()
                self.std_val = df['value'].std()
                if self.std_val > 0:
                    df['value'] = (df['value'] - self.mean_val) / self.std_val

        return df

    def _inverse_transform(self, values: Union[List[float], np.ndarray]) -> Union[List[float], np.ndarray]:
        """
        正規化された値を元のスケールに戻す

        Args:
            values: 正規化された値

        Returns:
            元のスケールの値
        """
        preprocessing_config = self.config.get('preprocessing', {})
        if preprocessing_config.get('normalize', False):
            method = preprocessing_config.get('normalization_method', 'minmax')
            if method == 'minmax' and hasattr(self, 'min_val') and hasattr(self, 'max_val'):
                if isinstance(values, list):
                    return [v * (self.max_val - self.min_val) + self.min_val for v in values]
                else:
                    return values * (self.max_val - self.min_val) + self.min_val
            elif method == 'standard' and hasattr(self, 'mean_val') and hasattr(self, 'std_val'):
                if isinstance(values, list):
                    return [v * self.std_val + self.mean_val for v in values]
                else:
                    return values * self.std_val + self.mean_val
        return values

    def fit(self, timestamps: List[datetime.datetime], values: List[float]) -> None:
        """
        モデルを学習

        Args:
            timestamps: タイムスタンプのリスト
            values: 値のリスト
        """
        # 実際の実装では、chronos-boltライブラリを使用してモデルを学習
        # ここではダミー実装
        logger.info(f"モデル '{self.model_name}' の学習を開始します")

        # データの準備
        df = self._prepare_data(timestamps, values)

        # モデルの学習
        # 注: 実際の実装では、chronos-boltライブラリのAPIに合わせて実装
        try:
            # ダミー実装
            self.model = {
                'data': df,
                'params': {**self.config['default_model']['chronos'], **self.model_params}
            }
            logger.info(f"モデル '{self.model_name}' の学習が完了しました")
        except Exception as e:
            logger.error(f"モデルの学習に失敗しました: {e}")
            raise ValueError(f"モデルの学習に失敗しました: {e}")

    def predict(self, horizon: int = 24) -> Tuple[List[datetime.datetime], List[float], Dict[str, Any]]:
        """
        予測を実行

        Args:
            horizon: 予測期間

        Returns:
            予測期間のタイムスタンプ、予測値、メタデータのタプル
        """
        # 実際の実装では、chronos-boltライブラリを使用して予測を実行
        # ここではダミー実装
        if self.model is None:
            raise ValueError("モデルが学習されていません。先にfit()メソッドを呼び出してください。")

        logger.info(f"モデル '{self.model_name}' による予測を開始します（期間: {horizon}）")

        try:
            # ダミー予測の生成
            last_timestamp = self.model['data'].index[-1]
            frequency = self.model['params'].get('frequency', 'H')

            # 頻度に基づいて時間間隔を設定
            if frequency == 'H':
                delta = datetime.timedelta(hours=1)
            elif frequency == 'D':
                delta = datetime.timedelta(days=1)
            elif frequency == 'W':
                delta = datetime.timedelta(weeks=1)
            else:
                delta = datetime.timedelta(hours=1)

            # 予測期間のタイムスタンプを生成
            forecast_timestamps = [last_timestamp + delta * (i+1) for i in range(horizon)]

            # 最後の値を取得
            last_value = self.model['data']['value'].iloc[-1]

            # ダミーの予測値を生成（ランダムな変動を加える）
            import random
            forecast_values = [last_value + random.uniform(-0.1, 0.1) * (i+1) for i in range(horizon)]

            # 正規化されている場合は元のスケールに戻す
            forecast_values = self._inverse_transform(forecast_values)

            # 信頼区間を生成
            lower_95 = [v - abs(v) * 0.1 for v in forecast_values]
            upper_95 = [v + abs(v) * 0.1 for v in forecast_values]

            # メタデータを生成
            metadata = {
                'model_name': self.model_name,
                'confidence_intervals': {
                    'lower_95': lower_95,
                    'upper_95': upper_95
                },
                'metrics': {
                    'mse': 0.15,
                    'mae': 0.12
                }
            }

            logger.info(f"予測が完了しました（{len(forecast_values)}ポイント）")

            return forecast_timestamps, forecast_values, metadata

        except Exception as e:
            logger.error(f"予測に失敗しました: {e}")
            raise ValueError(f"予測に失敗しました: {e}")

    def save_model(self, path: str) -> None:
        """
        モデルを保存

        Args:
            path: 保存先のパス
        """
        # 実際の実装では、chronos-boltライブラリを使用してモデルを保存
        # ここではダミー実装
        logger.info(f"モデル '{self.model_name}' を保存します: {path}")

        try:
            # ダミー実装
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(f"Dummy model: {self.model_name}")
            logger.info(f"モデルの保存が完了しました: {path}")
        except Exception as e:
            logger.error(f"モデルの保存に失敗しました: {e}")
            raise ValueError(f"モデルの保存に失敗しました: {e}")

    @classmethod
    def load_model(cls, path: str) -> 'TimeSeriesPredictor':
        """
        モデルを読み込み

        Args:
            path: モデルファイルのパス

        Returns:
            TimeSeriesPredictorインスタンス
        """
        # 実際の実装では、chronos-boltライブラリを使用してモデルを読み込み
        # ここではダミー実装
        logger.info(f"モデルを読み込みます: {path}")

        try:
            # ダミー実装
            predictor = cls()
            # fit()メソッドで作成したモデルと同じ構造のモデルを作成
            import pandas as pd
            import numpy as np

            # ダミーデータの作成
            now = datetime.datetime.now()
            timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
            values = [10.0 + i * 0.1 for i in range(24)]

            # DataFrameの作成
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': values
            })
            df.set_index('timestamp', inplace=True)

            # モデルの作成
            predictor.model = {
                'data': df,
                'params': {
                    'frequency': 'H',
                    'seasonality_mode': 'multiplicative'
                }
            }

            logger.info(f"モデルの読み込みが完了しました: {path}")
            return predictor
        except Exception as e:
            logger.error(f"モデルの読み込みに失敗しました: {e}")
            raise ValueError(f"モデルの読み込みに失敗しました: {e}")
