# ホスト環境実行 (Apple Silicon 対応)

Apple Silicon Mac (M1/M2/M4) でChronos-Boltモデルを高速・CPU実行するためのセットアップです。

## クイックスタート

```bash
# プロジェクトルートから実行
./run_host/run.sh
```

## 手動セットアップ

### 1. Conda環境作成
```bash
conda env create -f environment.yml
```

### 2. 環境アクティベート
```bash
conda activate zcrc-chronos
```

### 3. PyTorch動作確認
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 4. サーバー起動
```bash
./run_host/run.sh
```

## 特徴

- ✅ **Apple Silicon 最適化**: M1/M2/M4チップで高速動作
- ✅ **Chronos-Bolt 高速モデル**: 600倍高速な事前訓練済みTransformer
- ✅ **CPU安定実行**: 互換性問題を回避した安定動作
- ✅ **依存関係自動管理**: environment.ymlで一括管理

## 高速モデル

以下のモデルがCPUで高速動作します：

- `chronos_bolt`: Chronos-Bolt高速事前訓練済みTransformer（CPU専用）
- `deep_learning`: 従来の深層学習モデル群

## 環境変数

スクリプト内で以下の環境変数が設定されます：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1          # Apple Silicon互換性確保
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # メモリ使用量最適化
export TOKENIZERS_PARALLELISM=false           # 並列処理競合回避
```

## ログ管理

### ログファイルの場所

サーバーのログは以下の場所に出力されます：

```bash
run_host/logs/app.log              # 現在のログファイル
run_host/logs/app.YYYY-MM-DD_*.log # ローテーション済みログファイル
```

### ログの確認方法

```bash
# 現在のログをリアルタイム監視
tail -f run_host/logs/app.log

# 最新の100行を表示
tail -100 run_host/logs/app.log

# ログファイル一覧を確認
ls -la run_host/logs/

# 特定のキーワードでログを検索
grep "エラー\|ERROR" run_host/logs/app*.log

# 予測処理のログのみを表示
grep "予測" run_host/logs/app*.log
```

### ログローテーション

- **ファイルサイズ**: 10MBごとに自動ローテーション
- **保持数**: 最新10個のファイルを保持
- **命名規則**: `app.YYYY-MM-DD_HH-mm-ss_xxxxxx.log`

### ログレベル調整

ログレベルは `config/app_config.yaml` で変更できます：

```yaml
logging:
  level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## トラブルシューティング

### Chronos-Bolt動作確認
```bash
# PyTorchバージョン確認
python -c "import torch; print(torch.__version__)"

# AutoGluon動作確認
python -c "from autogluon.timeseries import TimeSeriesPredictor; print('AutoGluon OK')"
```

### 依存関係エラー
```bash
# 環境リセット
conda env remove -n zcrc-chronos
conda env create -f environment.yml
```

### ログ関連のトラブルシューティング

```bash
# ログディレクトリが存在しない場合
mkdir -p run_host/logs

# ログファイルへの書き込み権限エラー
chmod 755 run_host/logs/
chmod 644 run_host/logs/app.log

# ログファイルサイズが大きすぎる場合（手動ローテーション）
mv run_host/logs/app.log run_host/logs/app.$(date +%Y%m%d_%H%M%S).log

# すべてのログファイルを削除（注意: データが失われます）
rm -f run_host/logs/app*.log

# 特定の期間のログを検索
grep "$(date +%Y-%m-%d)" run_host/logs/app*.log
```

## パフォーマンス比較

| 環境 | 実行場所 | 処理 | Chronosモデル |
|------|----------|-----|---------------|
| Docker | コンテナ内 | CPU | 統計モデル群 (SeasonalNaive等) |
| Host | macOS直接 | CPU | Chronos-Bolt高速 (600倍高速) |
