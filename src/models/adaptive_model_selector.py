"""
データ特性に基づく適応的モデル選択モジュール
時系列の特性を分析して最適なモデル組み合わせを動的に決定
"""

import datetime
from typing import Dict, List

from loguru import logger

from .time_series_analyzer import TimeSeriesAnalyzer, TimeSeriesCharacteristics


class ModelSelectionStrategy:
    """モデル選択戦略を表すクラス"""

    def __init__(
        self,
        strategy_name: str,
        priority_models: List[str],
        excluded_models: List[str],
        time_allocation: Dict[str, float],
        preset: str = "medium_quality",
    ):
        self.strategy_name = strategy_name
        self.priority_models = priority_models
        self.excluded_models = excluded_models
        self.time_allocation = (
            time_allocation  # {"fast": 0.2, "medium": 0.5, "advanced": 0.3}
        )
        self.preset = preset


class AdaptiveModelSelector:
    """データ特性に基づく適応的モデル選択器"""

    def __init__(self):
        self.analyzer = TimeSeriesAnalyzer()
        self.strategies = self._initialize_strategies()

    def _initialize_strategies(self) -> Dict[str, ModelSelectionStrategy]:
        """事前定義された選択戦略を初期化"""

        strategies = {}

        # 強い季節性パターン用
        strategies["strong_seasonal"] = ModelSelectionStrategy(
            strategy_name="strong_seasonal",
            priority_models=["SeasonalNaive", "AutoETS", "Theta", "NPTS"],
            excluded_models=["Naive", "Chronos", "TemporalFusionTransformer"],
            time_allocation={"fast": 0.15, "medium": 0.6, "advanced": 0.25},
            preset="medium_quality",
        )

        # 強いトレンド用
        strategies["strong_trend"] = ModelSelectionStrategy(
            strategy_name="strong_trend",
            priority_models=["ARIMA", "ETS", "RecursiveTabular", "NPTS", "AutoETS"],
            excluded_models=["Naive", "SeasonalNaive", "Chronos"],
            time_allocation={"fast": 0.1, "medium": 0.5, "advanced": 0.4},
            preset="medium_quality",
        )

        # 高ボラティリティ用
        strategies["high_volatility"] = ModelSelectionStrategy(
            strategy_name="high_volatility",
            priority_models=["NPTS", "RecursiveTabular", "DirectTabular", "DeepAR"],
            excluded_models=["Naive", "SeasonalNaive", "Chronos"],
            time_allocation={"fast": 0.1, "medium": 0.4, "advanced": 0.5},
            preset="high_quality",
        )

        # 小データセット用
        strategies["small_dataset"] = ModelSelectionStrategy(
            strategy_name="small_dataset",
            priority_models=["AutoETS", "ETS", "Theta", "SeasonalNaive"],
            excluded_models=["Naive", "Chronos", "TemporalFusionTransformer", "DeepAR"],
            time_allocation={"fast": 0.3, "medium": 0.7, "advanced": 0.0},
            preset="medium_quality",
        )

        # 大データセット用
        strategies["large_dataset"] = ModelSelectionStrategy(
            strategy_name="large_dataset",
            priority_models=[
                "RecursiveTabular",
                "DirectTabular",
                "NPTS",
                "DeepAR",
                "AutoETS",
            ],
            excluded_models=["Naive"],
            time_allocation={"fast": 0.1, "medium": 0.3, "advanced": 0.6},
            preset="high_quality",
        )

        # 不規則時系列用
        strategies["irregular"] = ModelSelectionStrategy(
            strategy_name="irregular",
            priority_models=["NPTS", "RecursiveTabular", "AutoETS", "DirectTabular"],
            excluded_models=["Naive", "SeasonalNaive", "Chronos"],
            time_allocation={"fast": 0.2, "medium": 0.5, "advanced": 0.3},
            preset="medium_quality",
        )

        # デフォルト/標準パターン用
        strategies["balanced"] = ModelSelectionStrategy(
            strategy_name="balanced",
            priority_models=["AutoETS", "RecursiveTabular", "NPTS", "SeasonalNaive"],
            excluded_models=["Naive", "Chronos", "TemporalFusionTransformer"],
            time_allocation={"fast": 0.2, "medium": 0.5, "advanced": 0.3},
            preset="medium_quality",
        )

        return strategies

    def select_optimal_strategy(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        horizon: int,
        time_budget: int = 900,
    ) -> ModelSelectionStrategy:
        """
        データ特性を分析して最適な戦略を選択

        Args:
            values: 時系列の値
            timestamps: タイムスタンプ
            horizon: 予測期間
            time_budget: 時間予算（秒）

        Returns:
            ModelSelectionStrategy: 選択された戦略
        """
        logger.info("データ特性に基づく最適戦略の選択を開始します")

        try:
            # データ特性分析
            characteristics = self.analyzer.analyze_time_series_characteristics(
                values, timestamps
            )

            # データサイズによる基本分類
            data_size = len(values)

            # 戦略選択ロジック
            strategy = self._determine_strategy(
                characteristics, data_size, horizon, time_budget
            )

            # 選択された戦略をログ出力
            logger.info(f"選択された戦略: {strategy.strategy_name}")
            logger.info(f"優先モデル: {strategy.priority_models}")
            logger.info(f"除外モデル: {strategy.excluded_models}")
            logger.info(f"時間配分: {strategy.time_allocation}")

            return strategy

        except Exception as e:
            logger.warning(f"戦略選択でエラーが発生: {e}。デフォルト戦略を使用します")
            return self.strategies["balanced"]

    def _determine_strategy(
        self,
        characteristics: TimeSeriesCharacteristics,
        data_size: int,
        horizon: int,
        time_budget: int,
    ) -> ModelSelectionStrategy:
        """特性に基づく戦略決定の中核ロジック"""

        # データサイズによる基本分類
        if data_size < 50:
            base_strategy = "small_dataset"
        elif data_size > 500:
            base_strategy = "large_dataset"
        else:
            base_strategy = "balanced"

        # 特性による戦略調整
        strategy_scores = {}

        # 季節性の強さを評価
        if (
            characteristics.seasonality.get("strength") == "strong"
            and characteristics.seasonality.get("score", 0) > 0.3
        ):
            strategy_scores["strong_seasonal"] = (
                characteristics.seasonality.get("score", 0) * 2.0
            )
            logger.debug("強い季節性を検出")

        # トレンドの強さを評価
        if (
            characteristics.trend.get("strength") == "strong"
            and characteristics.trend.get("r_squared", 0) > 0.7
        ):
            strategy_scores["strong_trend"] = (
                characteristics.trend.get("r_squared", 0) * 1.5
            )
            logger.debug("強いトレンドを検出")

        # ボラティリティを評価
        if characteristics.volatility > 0.15:
            strategy_scores["high_volatility"] = min(
                characteristics.volatility * 2.0, 2.0
            )
            logger.debug(f"高ボラティリティを検出: {characteristics.volatility:.3f}")

        # 時間密度の不規則性を評価
        if not characteristics.density.get("regular", True):
            strategy_scores["irregular"] = 1.0
            logger.debug("不規則時系列を検出")

        # データサイズによる制約
        if data_size < 50:
            # 小データセットでは複雑なモデルを避ける
            strategy_scores.pop("high_volatility", None)
            strategy_scores.pop("large_dataset", None)

        # 最高スコアの戦略を選択
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"戦略スコア: {strategy_scores}")
            logger.debug(f"最高スコア戦略: {best_strategy}")
        else:
            best_strategy = base_strategy
            logger.debug(f"特性に基づく戦略なし、ベース戦略を使用: {base_strategy}")

        # 戦略の取得と時間予算による調整
        strategy = self.strategies.get(best_strategy, self.strategies["balanced"])

        # 時間予算が少ない場合の調整
        if time_budget < 300:  # 5分未満
            strategy = self._adjust_for_short_time(strategy)
        elif time_budget > 1800:  # 30分超
            strategy = self._adjust_for_long_time(strategy)

        return strategy

    def _adjust_for_short_time(
        self, strategy: ModelSelectionStrategy
    ) -> ModelSelectionStrategy:
        """短時間予算用の戦略調整"""
        logger.debug("短時間予算用に戦略を調整")

        # 高速モデルのみに制限
        fast_models = ["SeasonalNaive", "AutoETS", "ETS"]
        priority_models = [m for m in strategy.priority_models if m in fast_models]

        if not priority_models:
            priority_models = ["SeasonalNaive", "AutoETS"]

        return ModelSelectionStrategy(
            strategy_name=f"{strategy.strategy_name}_fast",
            priority_models=priority_models,
            excluded_models=strategy.excluded_models
            + ["RecursiveTabular", "DirectTabular", "DeepAR"],
            time_allocation={"fast": 0.6, "medium": 0.4, "advanced": 0.0},
            preset="fast_training",
        )

    def _adjust_for_long_time(
        self, strategy: ModelSelectionStrategy
    ) -> ModelSelectionStrategy:
        """長時間予算用の戦略調整"""
        logger.debug("長時間予算用に戦略を調整")

        # 高精度モデルも追加
        extended_models = strategy.priority_models.copy()

        # 除外リストにないモデルを追加
        candidates = ["DeepAR", "TemporalFusionTransformer", "PatchTST", "TiDE"]
        for model in candidates:
            if model not in strategy.excluded_models and model not in extended_models:
                extended_models.append(model)

        return ModelSelectionStrategy(
            strategy_name=f"{strategy.strategy_name}_extended",
            priority_models=extended_models,
            excluded_models=[
                m for m in strategy.excluded_models if m != "TemporalFusionTransformer"
            ],
            time_allocation={"fast": 0.1, "medium": 0.3, "advanced": 0.6},
            preset="high_quality",
        )

    def get_hierarchical_model_groups(
        self, strategy: ModelSelectionStrategy
    ) -> Dict[str, List[str]]:
        """
        階層的学習用のモデルグループを生成

        Args:
            strategy: 選択された戦略

        Returns:
            Dict[str, List[str]]: ステージ別モデルグループ
        """
        all_models = strategy.priority_models

        # モデルを速度と複雑性で分類
        fast_models = []
        medium_models = []
        advanced_models = []

        for model in all_models:
            if model in ["SeasonalNaive", "AutoETS", "ETS", "Theta"]:
                fast_models.append(model)
            elif model in ["RecursiveTabular", "DirectTabular", "NPTS", "ARIMA"]:
                medium_models.append(model)
            else:  # DeepAR, TemporalFusionTransformer, etc.
                advanced_models.append(model)

        groups = {}
        if fast_models:
            groups["fast"] = fast_models
        if medium_models:
            groups["medium"] = medium_models
        if advanced_models:
            groups["advanced"] = advanced_models

        logger.debug(f"階層的モデルグループ: {groups}")
        return groups

    def estimate_model_training_time(self, model_name: str, data_size: int) -> float:
        """
        モデルの推定学習時間（秒）

        Args:
            model_name: モデル名
            data_size: データサイズ

        Returns:
            float: 推定学習時間（秒）
        """
        # 基本的な学習時間の見積もり（経験的）
        base_times = {
            "SeasonalNaive": 1,
            "AutoETS": 10,
            "ETS": 8,
            "Theta": 5,
            "ARIMA": 15,
            "RecursiveTabular": 30,
            "DirectTabular": 25,
            "NPTS": 20,
            "DeepAR": 60,
            "TemporalFusionTransformer": 120,
            "PatchTST": 90,
            "TiDE": 80,
        }

        base_time = base_times.get(model_name, 30)

        # データサイズによるスケーリング
        if data_size < 100:
            scale_factor = 1.0
        elif data_size < 500:
            scale_factor = 1.5
        else:
            scale_factor = 2.0

        return base_time * scale_factor
