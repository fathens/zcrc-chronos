"""
階層的モデル学習システム
高速→中速→高精度の順でモデルを学習し、早期停止で効率化を図る
"""

import datetime
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .adaptive_model_selector import AdaptiveModelSelector, ModelSelectionStrategy


class ModelTrainingResult:
    """モデル学習結果を格納するクラス"""

    def __init__(
        self,
        model_name: str,
        score: float,
        training_time: float,
        prediction_result: Any = None,
        metadata: Dict[str, Any] = None,
    ):
        self.model_name = model_name
        self.score = score
        self.training_time = training_time
        self.prediction_result = prediction_result
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now()


class HierarchicalTrainer:
    """階層的モデル学習システム"""

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.selector = AdaptiveModelSelector()
        self.training_results = {}
        self.best_score = float("inf")
        self.early_stopping_threshold = 0.05  # 5%改善がなければ停止
        self.lock = threading.Lock()

    def train_hierarchically(
        self,
        predictor_class,
        predictor_kwargs,
        time_series_data,
        strategy: ModelSelectionStrategy,
        time_budget: int,
        horizon: int,
        excluded_models: List[str],
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        階層的モデル学習を実行

        Args:
            predictor_class: AutoGluon predictor クラス
            predictor_kwargs: predictor初期化パラメータ
            time_series_data: 学習データ
            strategy: 選択された戦略
            time_budget: 時間予算（秒）
            horizon: 予測期間
            excluded_models: 除外モデルリスト

        Returns:
            Tuple[Any, Dict[str, Any]]: (予測結果, メタデータ)
        """
        logger.info("階層的モデル学習を開始します")

        start_time = time.time()

        # モデルグループの取得
        model_groups = self.selector.get_hierarchical_model_groups(strategy)

        # 時間配分の計算
        time_allocation = self._calculate_time_allocation(
            strategy.time_allocation, time_budget
        )

        # 各段階で学習を実行
        final_result = None
        final_metadata = {}

        try:
            for stage, models in model_groups.items():
                if not models:
                    continue

                stage_time_budget = time_allocation.get(stage, 0)
                if stage_time_budget <= 0:
                    continue

                elapsed = time.time() - start_time
                remaining_time = time_budget - elapsed

                if remaining_time <= 30:  # 30秒未満なら停止
                    logger.warning("時間予算不足のため階層学習を終了")
                    break

                # 実際の利用可能時間を調整
                actual_time_budget = min(stage_time_budget, remaining_time - 10)

                logger.info(
                    f"Stage '{stage}' 開始: models={models}, "
                    f"time_budget={actual_time_budget}s"
                )

                # ステージごとの学習実行
                stage_result = self._train_stage(
                    predictor_class,
                    predictor_kwargs,
                    time_series_data,
                    models,
                    excluded_models,
                    actual_time_budget,
                    horizon,
                    stage,
                )

                if stage_result:
                    final_result = stage_result
                    final_metadata.update(
                        {
                            f"{stage}_stage": {
                                "models": models,
                                "time_spent": time.time()
                                - start_time
                                - sum(
                                    final_metadata.get(f"{s}_stage", {}).get(
                                        "time_spent", 0
                                    )
                                    for s in ["fast", "medium"]
                                    if f"{s}_stage" in final_metadata
                                ),
                                "best_score": self.best_score,
                            }
                        }
                    )

                # 早期停止の判定
                if self._should_stop_early(stage, models):
                    logger.info(f"Stage '{stage}' で十分な性能を達成、早期停止")
                    break

            # 最終メタデータの設定
            total_time = time.time() - start_time
            final_metadata.update(
                {
                    "hierarchical_training": True,
                    "total_training_time": total_time,
                    "strategy_used": strategy.strategy_name,
                    "stages_completed": list(model_groups.keys()),
                    "best_overall_score": self.best_score,
                    "training_results_summary": self._summarize_results(),
                }
            )

            logger.info(
                f"階層的学習完了: 総時間={total_time:.1f}s, 最良スコア={self.best_score:.4f}"
            )

            return final_result, final_metadata

        except Exception as e:
            logger.error(f"階層的学習でエラー: {e}")

            # フォールバック: 標準のAutoGluon学習
            return self._fallback_training(
                predictor_class,
                predictor_kwargs,
                time_series_data,
                excluded_models,
                time_budget,
            )

    def _calculate_time_allocation(
        self, allocation_ratios: Dict[str, float], total_budget: int
    ) -> Dict[str, int]:
        """時間配分を計算"""
        allocation = {}

        for stage, ratio in allocation_ratios.items():
            allocation[stage] = int(total_budget * ratio)

        logger.debug(f"時間配分: {allocation}")
        return allocation

    def _train_stage(
        self,
        predictor_class,
        predictor_kwargs,
        time_series_data,
        models: List[str],
        excluded_models: List[str],
        time_budget: int,
        horizon: int,
        stage: str,
    ) -> Optional[Any]:
        """単一ステージの学習実行"""

        if time_budget <= 0:
            return None

        try:
            # 新しいpredictorインスタンスを作成（各ステージで独立）
            stage_predictor = predictor_class(**predictor_kwargs)

            # このステージで使用するモデルを制限
            stage_excluded = excluded_models.copy()

            # 他のステージのモデルを除外に追加
            all_possible_models = [
                "SeasonalNaive",
                "AutoETS",
                "ETS",
                "Theta",
                "ARIMA",
                "RecursiveTabular",
                "DirectTabular",
                "NPTS",
                "DeepAR",
                "TemporalFusionTransformer",
                "PatchTST",
                "TiDE",
            ]

            for model in all_possible_models:
                if model not in models:
                    stage_excluded.append(model)

            # ハイパーパラメータの設定（このステージのモデルのみ）
            hyperparameters = {}
            for model in models:
                hyperparameters[model] = {}

            # 学習の実行
            logger.info(f"Stage '{stage}' モデル学習開始: {models}")

            fit_kwargs = {
                "train_data": time_series_data,
                "time_limit": time_budget,
                "hyperparameters": hyperparameters,
                "excluded_model_types": stage_excluded,
                "num_val_windows": 1,
                "val_step_size": 1,
                "skip_model_selection": False,
                "enable_ensemble": True,
            }

            stage_start = time.time()
            stage_predictor.fit(**fit_kwargs)
            stage_time = time.time() - stage_start

            # 予測の実行
            forecast_result = stage_predictor.predict(time_series_data)

            # 性能評価
            score = self._evaluate_prediction(forecast_result, time_series_data)

            # 結果の記録
            with self.lock:
                self.training_results[stage] = ModelTrainingResult(
                    model_name=f"{stage}_ensemble",
                    score=score,
                    training_time=stage_time,
                    prediction_result=forecast_result,
                    metadata={
                        "models_used": models,
                        "stage": stage,
                        "time_budget": time_budget,
                        "actual_time": stage_time,
                    },
                )

                if score < self.best_score:
                    self.best_score = score
                    logger.info(f"Stage '{stage}' で新しい最良スコア: {score:.4f}")

            logger.info(
                f"Stage '{stage}' 完了: time={stage_time:.1f}s, score={score:.4f}"
            )
            return forecast_result

        except Exception as e:
            logger.error(f"Stage '{stage}' でエラー: {e}")
            return None

    def _should_stop_early(self, current_stage: str, models: List[str]) -> bool:
        """早期停止の判定"""

        # 高速ステージでは停止しない
        if current_stage == "fast":
            return False

        # 十分な改善があれば継続
        if len(self.training_results) >= 2:
            scores = [result.score for result in self.training_results.values()]
            if len(scores) >= 2:
                improvement = (max(scores) - min(scores)) / max(scores)
                if improvement < self.early_stopping_threshold:
                    logger.debug(
                        f"改善率{improvement:.3f}が閾値{self.early_stopping_threshold}未満"
                    )
                    return True

        # スコアが十分良い場合
        if self.best_score < 0.05:  # MAEが5%未満
            logger.debug(f"スコア{self.best_score:.4f}が十分良好")
            return True

        return False

    def _evaluate_prediction(self, forecast_result, time_series_data) -> float:
        """予測結果の評価（MAE）"""
        try:
            # TimeSeriesDataFrameの処理を安全に行う
            if forecast_result is None:
                return 1.0

            # 予測結果の抽出を安全に実行
            predictions = None

            # AutoGluonのTimeSeriesDataFrameからの予測値取得
            try:
                if hasattr(forecast_result, "loc"):
                    # TimeSeriesDataFrameの場合
                    if hasattr(forecast_result, "item_ids"):
                        item_ids = forecast_result.item_ids
                        if len(item_ids) > 0:
                            first_item = item_ids[0]
                            if (first_item, "mean") in forecast_result.loc:
                                predictions = forecast_result.loc[(first_item, "mean")]
                            elif first_item in forecast_result.loc:
                                predictions = forecast_result.loc[first_item]

                # まだ取得できていない場合の他の方法
                if predictions is None:
                    if hasattr(forecast_result, "values"):
                        predictions = forecast_result.values
                    elif hasattr(forecast_result, "mean"):
                        predictions = forecast_result["mean"]

            except Exception as extract_error:
                logger.debug(f"予測値抽出でエラー: {extract_error}")
                return 1.0

            # 予測値を数値配列に変換
            if predictions is not None:
                try:
                    if hasattr(predictions, "values"):
                        pred_array = np.array(predictions.values).flatten()
                    else:
                        pred_array = np.array(predictions).flatten()

                    # 有効な予測値があることを確認
                    if len(pred_array) > 0 and not np.all(np.isnan(pred_array)):
                        # 予測値の分散を使用した簡易評価
                        mean_pred = np.nanmean(np.abs(pred_array))
                        if mean_pred > 1e-8:
                            score = np.nanstd(pred_array) / mean_pred
                        else:
                            score = 1.0
                        return float(score)

                except Exception as conv_error:
                    logger.debug(f"予測値変換でエラー: {conv_error}")

            # フォールバック: 一定の値を返す
            return 0.5

        except Exception as e:
            logger.warning(f"評価でエラー: {e}")
            return 1.0  # 悪い成績として扱う

    def _fallback_training(
        self,
        predictor_class,
        predictor_kwargs,
        time_series_data,
        excluded_models: List[str],
        time_budget: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        """フォールバック: 標準のAutoGluon学習"""

        logger.warning("フォールバック: 標準AutoGluon学習を実行")

        try:
            # 新しいpredictorインスタンスを作成
            fallback_predictor = predictor_class(**predictor_kwargs)

            fit_kwargs = {
                "train_data": time_series_data,
                "time_limit": time_budget,
                "excluded_model_types": excluded_models,
                "presets": "medium_quality",
                "num_val_windows": 1,
                "val_step_size": 1,
                "skip_model_selection": False,
                "enable_ensemble": True,
            }

            start_time = time.time()
            fallback_predictor.fit(**fit_kwargs)
            training_time = time.time() - start_time

            forecast_result = fallback_predictor.predict(time_series_data)

            metadata = {
                "hierarchical_training": False,
                "fallback_used": True,
                "total_training_time": training_time,
                "strategy_used": "fallback_standard",
            }

            return forecast_result, metadata

        except Exception as e:
            logger.error(f"フォールバック学習も失敗: {e}")
            raise

    def _summarize_results(self) -> Dict[str, Any]:
        """学習結果の要約"""
        summary = {}

        for stage, result in self.training_results.items():
            summary[stage] = {
                "score": result.score,
                "training_time": result.training_time,
                "models": result.metadata.get("models_used", []),
            }

        return summary

    def get_best_result(self) -> Optional[ModelTrainingResult]:
        """最良の学習結果を取得"""
        if not self.training_results:
            return None

        return min(self.training_results.values(), key=lambda x: x.score)

    def reset(self):
        """状態のリセット"""
        self.training_results.clear()
        self.best_score = float("inf")
