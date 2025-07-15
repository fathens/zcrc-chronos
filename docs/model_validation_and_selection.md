# モデル検証と選択プロセス

## 概要

本ドキュメントでは、zcrc-chronosにおけるAutoGluon-TimeSeriesを使用した時系列予測での**モデル検証（Model Validation）**と**自動モデル選択（Automatic Model Selection）**のメカニズムについて詳細に説明します。

複数のモデルが並行して訓練される理由、各モデルの精度がどのように評価されるか、そして最終的にどのモデルが選択されるかのプロセスを解説します。

## AutoGluonアンサンブル学習システム

### 基本コンセプト

zcrc-chronosでは**単一モデル専用設計**を採用しており、ユーザーが指定した特定のモデルのみを訓練・使用します。これにより予測精度と処理速度の最適化を実現しています。

各モデルは特定の用途・特性に最適化されており、ユーザーが目的に応じて選択できます：
- **高速処理**: SeasonalNaive（2分以内）
- **統計的手法**: AutoETS（5分以内）
- **機械学習**: NPTS、RecursiveTabular（10分以内）
- **深層学習**: DeepAR、TemporalFusionTransformer（15-20分以内）

### 利用可能な単一モデル一覧

```python
# 現在サポートされている単一モデル
available_models = [
    'AutoETS',              # 自動指数平滑法（統計的手法・高速）
    'NPTS',                 # Neural Prophet Time Series（高精度）
    'SeasonalNaive',        # 季節性ベースライン（超高速）
    'RecursiveTabular',     # 勾配ブースティング（高精度）
    'ChronosZeroShot',      # Transformerベース（事前訓練済み）
    'DynamicOptimizedTheta', # 動的最適化（バランス型）
    'TemporalFusionTransformer', # 注意機構（最高精度）
    'DeepAR'                # Amazon開発の深層学習
]
```

### 単一モデル設定の利点

```python
# 各モデルは特定用途に最適化
single_model_config = {
    "use_single_model": True,
    "enable_ensemble": False,
    "skip_model_selection": True
}
```

## 単一モデル検証プロセス

### 指定モデル専用訓練

zcrc-chronosでは、ユーザーが指定したモデルのみを訓練し、**モデル選択プロセスをスキップ**することで高速化を実現します。

#### 検証プロセスの詳細

```python
# 検証設定パラメータ
num_val_windows = 1      # 検証ウィンドウ数
val_step_size = prediction_length  # ウィンドウ間のステップサイズ
```

#### データ分割の実例

```
全データサイズ: 7,342 ポイント
予測期間: 734 ポイント

データ分割:
[------------- Training Data (6,608) -------------][-- Validation (734) --]
                                                   [-- Predict → Compare --]

時系列:
2025-06-25 00:00 ........................ 2025-07-04 XX:XX | 2025-07-05 XX:XX
                              ↑                                     ↑
                          訓練終了点                            検証終了点
```

### 検証ステップの詳細

1. **データ分割**
   - 最後の`prediction_length`ポイント（734ポイント）を検証用に確保
   - 残りのデータ（6,608ポイント）で各モデルを訓練

2. **予測実行**
   - 各モデルが訓練データのみを使用して734ポイント先を予測

3. **精度計算**
   - 予測値と実際の検証データ（確保した734ポイント）を比較
   - MAE（Mean Absolute Error）を計算

4. **モデル選択**
   - 最も低いMAEを持つモデルを「最優秀モデル」として選択

### 検証の利点

- **現実的な評価**: 実際の使用シナリオ（未来の予測）をシミュレート
- **データリーク防止**: 未来の情報を使って過去を予測することを防ぐ
- **時系列特性の尊重**: 時間の順序を保った検証
- **汎化性能の評価**: 見たことのないデータでの性能を測定

## 評価メトリック（MAE: Mean Absolute Error）

### MAEの計算方法

```python
# 予測値と実際値の平均絶対誤差
MAE = mean(|predicted_values - actual_values|)
```

### スコア表示の仕組み

AutoGluonは"higher is better"方式を採用しているため、MAEの符号を反転させます：

```
Validation score = -MAE

例:
- NPTS: -36,225,460 → 実際のMAE = 36,225,460
- AutoETS: -119,822,675 → 実際のMAE = 119,822,675
- ChronosZeroShot: -315,672,132 → 実際のMAE = 315,672,132
```

### 相対精度の解釈

価格データでの実例：
```
平均価格: 74,000,592,289 (約740億)
NPTS MAE: 36,225,460
相対誤差率: 36,225,460 / 74,000,592,289 ≈ 0.049% (約0.05%)
```

## 最終予測の実行

### 全データ使用の原則

モデル選択後、**全データ（7,342ポイント）**を使用して最終予測を実行：

```python
# 最終予測では検証用データも含めて全データを使用
final_prediction = best_model.predict(all_data)  # 7,342ポイント全て
```

### 予測と検証の分離

- **検証**: モデル選択のためのプロセス（データの一部を使用）
- **最終予測**: 実際の予測値生成（全データを使用）

## 実際の動作例

### ログ出力例

```
AutoGluon will gauge predictive performance using evaluation metric: 'MAE'
This metric's sign has been flipped to adhere to being higher_is_better.

Models that will be trained: ['SeasonalNaive', 'RecursiveTabular', 'DirectTabular',
'NPTS', 'DynamicOptimizedTheta', 'AutoETS', 'ChronosZeroShot[bolt_base]',
'ChronosFineTuned[bolt_small]', 'TemporalFusionTransformer', 'DeepAR', 'PatchTST', 'TiDE']

Training timeseries model NPTS. Training for up to 357.3s
    -36225460.1828= Validation score (-MAE)
    0.01s = Training runtime
    10.03s = Validation (prediction) runtime

Training timeseries model AutoETS. Training for up to 445.0s
    -119822675.9893= Validation score (-MAE)
    0.02s = Training runtime
    2.55s = Validation (prediction) runtime
```

### 結果の解釈

```
最優秀モデル: NPTS
理由: 最も低いMAE（36,225,460）を達成
精度: 約0.05%の予測誤差率
```

## パフォーマンス最適化

### 並列訓練

複数のモデルが**並列的に**訓練されるため、全体の処理時間が短縮されます：

```
シーケンシャル: Model1(10min) + Model2(15min) + Model3(20min) = 45分
並列実行: max(Model1(10min), Model2(15min), Model3(20min)) = 20分
```

### 時間制限設定

```python
time_limit = 3600  # 1時間のタイムリミット
preset = "high_quality"  # 高品質予測設定
```

## 設定カスタマイズ

### 検証設定の調整

```python
# データサイズに応じた動的調整
if data_length >= horizon + safe_margin:
    num_val_windows = 1  # 検証を有効
else:
    num_val_windows = 0  # 検証を無効化（データ不足時）
```

### モデル選択の制御

```python
# 特定のモデルのみを使用する場合
hyperparameters = {
    "NPTS": {},
    "AutoETS": {},
    "RecursiveTabular": {}
}

# より多くのモデルを試す場合
presets = "best_quality"  # より多くの検証ウィンドウを使用
```

## 品質保証メカニズム

### 自動フォールバック

```python
try:
    # メイン設定での訓練
    predictor.fit(presets="high_quality", ...)
except Exception:
    # エラー時のフォールバック
    predictor.fit(presets="medium_quality", ...)
```

### 検証品質チェック

- データサイズが不十分な場合は検証を無効化
- 予測期間が大きすぎる場合は自動調整
- エラー耐性のある設定に自動切り替え

## まとめ

### 主要な利点

1. **自動モデル選択**: 人間の介入なしに最適なモデルを選択
2. **客観的評価**: 時系列交差検証による公正な比較
3. **現実的な精度**: 実際の使用条件での性能評価
4. **堅牢性**: 複数のモデルによるリスク分散
5. **効率性**: 並列処理による時間短縮

### 技術的保証

- **時間順序の保持**: 未来の情報漏洩を防止
- **汎化性能の評価**: 未見データでの性能測定
- **統計的信頼性**: 複数モデルの比較による安定性
- **自動最適化**: データ特性に応じた設定調整

この仕組みにより、zcrc-chronosは**信頼性が高く、精度の優れた時系列予測**を自動的に提供します。

---

**最終更新**: 2025年7月8日
**バージョン**: v1.0.0
**作成者**: Claude Code
**関連ファイル**: `src/models/predictor.py`, `src/api/routes.py`, `config/model_config.yaml`
