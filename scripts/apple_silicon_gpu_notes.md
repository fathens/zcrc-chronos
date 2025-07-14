# Apple Silicon Docker GPU対応調査結果

## 現在の制約

### Docker Desktop for Mac の GPU制限
Apple Silicon Mac (M1/M2/M4) でDockerを使用する場合、以下の制約があります：

1. **Metal Performance Shaders (MPS) 非対応**
   - Docker Desktop for Mac は現在 MPS をサポートしていない
   - コンテナ内から Apple Silicon GPU に直接アクセスできない

2. **PyTorch MPS サポート**
   - ホスト macOS: PyTorch MPS 対応 ✅
   - Docker コンテナ内: PyTorch CPU のみ ❌

### 代替案検討

#### 1. ネイティブ実行 (推奨)
```bash
# ホスト環境で直接実行
conda create -n chronos-gpu python=3.12
conda activate chronos-gpu
pip install torch torchvision torchaudio  # MPS対応版
pip install autogluon.timeseries
```

#### 2. Docker + GPU パススルー (実験的)
```yaml
# docker-compose.yml (実験的設定)
services:
  zcrc-chronos:
    # ... 他の設定
    devices:
      - /dev/dri:/dev/dri  # GPU デバイス (Linux のみ)
    environment:
      - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### 3. Lima + Docker (高度)
```bash
# Lima を使った Linux VM 経由
brew install lima
limactl start --name=gpu-docker template://docker
```

## 実装戦略

### 短期: PyTorch MPS 対応版インストール
- environment.yml で明示的に PyTorch 指定
- `device: "auto"` で MPS/CUDA/CPU 自動選択

### 中期: デバイス検出ロジック追加
```python
def get_optimal_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
```

### 長期: ハイブリッド実行
- 軽量モデル: Docker コンテナ (CPU)
- 重い処理: ホスト環境 (GPU)

## 現在の解決策

1. **PyTorch MPS 対応版をインストール**: 完了 ✅
2. **Chronos モデル設定を本来の形に戻す**: 完了 ✅
3. **デバイス自動選択を設定**: 完了 ✅
4. **フォールバック機能**: 内蔵済み ✅

Docker環境でもフォールバックでCPU実行し、将来的にはネイティブ環境でGPU活用を検討。
