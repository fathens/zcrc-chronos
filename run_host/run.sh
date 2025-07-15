#!/bin/bash

# zcrc-chronos ホスト環境実行スクリプト
# Apple Silicon GPU (MPS) 対応版

set -e

cd "$(dirname $0)"
SCRIPT_DIR="$(pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="zcrc-chronos"

echo "🚀 zcrc-chronos ホスト環境 (GPU対応) を起動します..."
echo "📁 プロジェクトルート: $PROJECT_ROOT"
echo "🏷️  Conda環境: $ENV_NAME"

# conda環境の確認・作成
echo ""
echo "🔍 Conda環境を確認中..."
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "⚠️  環境 '$ENV_NAME' が見つかりません"
    echo "📦 environment.yml から環境を作成中..."

    cd "$PROJECT_ROOT"
    conda env create -f environment.yml

    if [ $? -ne 0 ]; then
        echo "❌ 環境作成に失敗しました。手動でセットアップしてください:"
        echo "   conda create -n $ENV_NAME python=3.12 -y"
        echo "   conda activate $ENV_NAME"
        echo "   pip install torch torchvision torchaudio"
        echo "   pip install autogluon.timeseries fastapi uvicorn pydantic pyyaml python-dotenv loguru"
        exit 1
    fi
else
    echo "✅ 環境 '$ENV_NAME' が見つかりました"
fi

# conda環境をアクティベート
echo "🔄 Conda環境をアクティベート中..."
# .zshrcのエラーを無視してcondaを直接初期化
eval "$(/Users/kunio/miniconda3/bin/conda shell.zsh hook)" 2>/dev/null || {
    export PATH="/Users/kunio/miniconda3/bin:$PATH"
}
conda activate "$ENV_NAME"

# GPU対応確認
echo ""
echo "🧪 GPU (MPS) 対応を確認中..."
python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'🍎 MPS available: {torch.backends.mps.is_available()}')
print(f'🍎 MPS built: {torch.backends.mps.is_built()}')

if torch.backends.mps.is_available():
    print('🎯 Apple Silicon GPU (MPS) が利用可能です')
    # テスト用テンソル作成
    try:
        x = torch.randn(3, 3, device='mps')
        print(f'✅ MPS テンソル作成成功: {x.device}')
    except Exception as e:
        print(f'⚠️  MPS テンソル作成に失敗: {e}')
else:
    print('⚠️  MPS が利用できません。CPU で実行されます')
"

echo ""
echo "🔍 AutoGluon TimeSeries を確認中..."
python3 -c "
try:
    from autogluon.timeseries import TimeSeriesPredictor
    print('✅ AutoGluon TimeSeries インポート成功')
except ImportError as e:
    print(f'❌ AutoGluon インポート失敗: {e}')
    exit(1)
"

# 作業ディレクトリを変更
cd "$PROJECT_ROOT"

# 環境変数設定
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# APIサーバー起動
echo ""
echo "🌐 APIサーバーを起動中..."
echo "📍 URL: http://localhost:8000"
echo "📊 ヘルスチェック: http://localhost:8000/api/v1/health"
echo "📋 利用可能モデル: http://localhost:8000/api/v1/models"
echo ""
echo "💡 停止するには Ctrl+C を押してください"
echo "================================"

# ログ出力でGPU使用状況を確認
export PYTORCH_ENABLE_MPS_FALLBACK=1

# サーバー起動
# 標準出力をapp.logにリダイレクト（標準エラー出力は画面に表示）
python -c "
import sys
import os

# プロジェクトルートをPythonパスに追加
project_root = os.getcwd()
sys.path.insert(0, project_root)

print('zcrc-chronos APIサーバーを起動します...')

from src.api.server import start_server
start_server()
" >> run_host/logs/app.log 2>&1
