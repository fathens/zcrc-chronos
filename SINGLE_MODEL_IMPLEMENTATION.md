# AutoGluon-TimeSeries単一モデル設定の実装

## 概要

AutoGluon-TimeSeriesで指定されたモデルのみを訓練し、自動的な複数モデル訓練を停止する機能を実装しました。これにより、ユーザーが特定のモデル（例：AutoETS、NPTS、SeasonalNaive、RecursiveTabular等）を指定した場合、そのモデルのみが使用されます。

## 解決した問題

### 以前の問題
- `fast_statistical`を指定しても、AutoGluonが勝手に複数モデル（ETS、NPTS、RecursiveTabular等）を訓練
- `balanced_ml`でも同様に複数モデルが自動選択される
- ユーザーが特定のモデルを指定した場合も、他のモデルが自動的に含まれる

### 解決方法
- 新しい`use_single_model`フラグを導入
- `hyperparameters`でモデルを1つだけ指定
- `enable_ensemble=False`でアンサンブルを無効化
- `skip_model_selection=True`でモデル選択をスキップ

## 実装された新しいモデル設定

### 1. AutoETSのみ (`autoets_only`)
```yaml
chronos:
  model_type: "autogluon"
  time_limit: 300
  use_single_model: true
  target_model: "AutoETSModel"
  hyperparameters:
    AutoETSModel:
      model: "ZZZ"  # 自動モデル選択
      seasonal_period: null  # 自動季節性検出
```

### 2. NPTSのみ (`npts_only`)
```yaml
chronos:
  model_type: "autogluon"
  time_limit: 600
  use_single_model: true
  target_model: "NPTSModel"
  hyperparameters:
    NPTSModel: {}
```

### 3. SeasonalNaiveのみ (`seasonal_naive_only`)
```yaml
chronos:
  model_type: "autogluon"
  time_limit: 60
  use_single_model: true
  target_model: "SeasonalNaiveModel"
  hyperparameters:
    SeasonalNaiveModel: {}
```

### 4. RecursiveTabularのみ (`recursive_tabular_only`)
```yaml
chronos:
  model_type: "autogluon"
  time_limit: 900
  use_single_model: true
  target_model: "RecursiveTabularModel"
  hyperparameters:
    RecursiveTabularModel:
      tabular_hyperparameters:
        GBM: {}
      max_num_items: 20000
      max_num_samples: 1000000
```

### 5. 標準ETSのみ (`ets_only`)
```yaml
chronos:
  model_type: "autogluon"
  time_limit: 300
  use_single_model: true
  target_model: "ETSModel"
  hyperparameters:
    ETSModel: {}
```

## 修正されたファイル

### 1. `/src/api/routes.py`
- 新しい単一モデル設定を`hardcoded_models`に追加
- 各モデルに`use_single_model: True`フラグを設定
- 正確なモデル名（`AutoETSModel`、`NPTSModel`等）を使用

### 2. `/src/models/predictor.py`
- 単一モデル設定の検出ロジックを追加
- `fit_kwargs`で単一モデル用の設定を適用
- 再試行時も単一モデル設定を維持
- 訓練されたモデルの情報をメタデータに追加

### 3. `/config/model_config.yaml`
- 新しい単一モデル設定を追加
- 各モデルの推奨用途と期待精度を記載

## AutoGluonの単一モデル訓練の仕組み

### 重要なパラメータ
1. **`hyperparameters`**: 訓練するモデルを1つだけ指定
2. **`enable_ensemble=False`**: アンサンブルモデルの作成を無効化
3. **`skip_model_selection=True`**: モデル選択プロセスをスキップ
4. **`presets=None`**: 単一モデル時はプリセットを無効化

### 実装例
```python
predictor.fit(
    train_data=time_series_data,
    hyperparameters={
        "AutoETSModel": {
            "model": "ZZZ",
            "seasonal_period": None
        }
    },
    enable_ensemble=False,
    skip_model_selection=True,
    time_limit=300
)
```

## 使用方法

### APIでの使用
```json
{
    "timestamp": ["2023-01-01T00:00:00", ...],
    "values": [10.5, 11.2, ...],
    "forecast_until": "2023-01-04T02:00:00",
    "model_name": "autoets"
}
```

実際の実装では、以下のモデル名が使用されます：
- `autoets` （config設定の`autoets_only`に相当）
- `npts` （config設定の`npts_only`に相当）
- `seasonal_naive` （config設定の`seasonal_naive_only`に相当）
- `recursive_tabular` （config設定の`recursive_tabular_only`に相当）

### 設定の確認
予測結果のメタデータで単一モデル設定が正しく適用されているかを確認できます：
```python
metadata = {
    "use_single_model": True,
    "target_model": "AutoETSModel",
    "trained_models": ["AutoETSModel"]
}
```

## テスト

`test_single_models.py`スクリプトを実行して、実装が正しく動作することを確認できます：

```bash
python test_single_models.py
```

## 期待される結果

- `autoets`を指定した場合：AutoETSのみが訓練される
- `npts`を指定した場合：NPTSのみが訓練される
- `seasonal_naive`を指定した場合：SeasonalNaiveのみが訓練される
- `recursive_tabular`を指定した場合：RecursiveTabularのみが訓練される
- `chronos_bolt`を指定した場合：Chronos-Boltのみが使用される（事前訓練済み）
- `dynamic_theta`を指定した場合：DynamicOptimizedThetaのみが訓練される
- `temporal_fusion_transformer`を指定した場合：TemporalFusionTransformerのみが訓練される
- `deepar`を指定した場合：DeepARのみが訓練される

これにより、ユーザーが特定のモデルを指定した場合、そのモデルのみが使用され、他のモデルは一切訓練されません。

## 注意事項

1. **モデル名**: AutoGluonの正確なモデル名（`AutoETSModel`等）を使用する必要があります
2. **時間制限**: 単一モデルの場合、通常は短い時間で訓練が完了します
3. **エラーハンドリング**: 指定されたモデルが利用できない場合の再試行ロジックも実装されています
4. **後方互換性**: 既存の複数モデル設定は影響を受けません
