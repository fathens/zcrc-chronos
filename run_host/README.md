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
conda env create -f environment-host.yml
```

### 2. 環境アクティベート
```bash
conda activate zcrc-chronos-gpu
```

### 3. GPU対応確認
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 4. サーバー起動
```bash
python scripts/run_server.py
```

## 特徴

- ✅ **Apple Silicon GPU (MPS) 対応**: M1/M2/M4チップのGPUを活用
- ✅ **真のChronos Zero Shot**: 事前訓練済みTransformerモデル
- ✅ **自動フォールバック**: GPU利用不可時はCPUで実行
- ✅ **依存関係自動管理**: environment-host.ymlで一括管理

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
conda env remove -n zcrc-chronos-gpu
conda env create -f environment-host.yml
```

## パフォーマンス比較

| 環境 | 実行場所 | GPU | Chronosモデル |
|------|----------|-----|---------------|
| Docker | コンテナ内 | ❌ CPU | 疑似Zero Shot (SeasonalNaive) |
| Host | macOS直接 | ✅ MPS | 真のZero Shot (Chronos) |
