# ホスト環境実行 (Apple Silicon GPU対応)

Apple Silicon Mac (M1/M2/M4) のMetal Performance Shaders (MPS) を活用してChronosモデルをGPUで実行するためのセットアップです。

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

### 3. GPU対応確認
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 4. サーバー起動
```bash
./run_host/run.sh
```

## 特徴

- ✅ **Apple Silicon GPU (MPS) 対応**: M1/M2/M4チップのGPUを活用
- ✅ **真のChronos Zero Shot**: 事前訓練済みTransformerモデル
- ✅ **自動フォールバック**: GPU利用不可時はCPUで実行
- ✅ **依存関係自動管理**: environment.ymlで一括管理

## GPU対応モデル

以下のモデルでGPU (MPS) が活用されます：

- `chronos_zero_shot`: Chronos事前訓練済みTransformer
- `deep_learning`: ChronosZeroShot等の深層学習モデル含む

## 環境変数

スクリプト内で以下の環境変数が設定されます：

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # MPS メモリ管理
export PYTORCH_ENABLE_MPS_FALLBACK=1          # MPS フォールバック有効
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

### MPS利用不可の場合
```bash
# PyTorchバージョン確認
python -c "import torch; print(torch.__version__)"

# MPS対応確認
python -c "import torch; print(torch.backends.mps.is_built())"
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

| 環境 | 実行場所 | GPU | Chronosモデル |
|------|----------|-----|---------------|
| Docker | コンテナ内 | ❌ CPU | 疑似Zero Shot (SeasonalNaive) |
| Host | macOS直接 | ✅ MPS | 真のZero Shot (Chronos) |
