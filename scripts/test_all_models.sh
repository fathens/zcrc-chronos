#!/bin/bash

# 全モデル一括テストスクリプト

set -e

MODELS=(
    "chronos_default"
    "fast_statistical"
    "balanced_ml"
    "deep_learning"
    "autoets_only"
    "npts_only"
    "seasonal_naive_only"
    "recursive_tabular_only"
    "ets_only"
    "chronos_zero_shot"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUICK_TEST="$SCRIPT_DIR/quick_test.sh"

echo "🚀 全モデル動作確認を開始します..."
echo "📊 テスト対象: ${#MODELS[@]} モデル"
echo "================================"

for model in "${MODELS[@]}"; do
    echo
    echo "🔍 [$model] テスト中..."
    if bash "$QUICK_TEST" "$model"; then
        echo "✅ [$model] OK"
    else
        echo "❌ [$model] FAILED"
    fi
    echo "--------------------------------"
    sleep 2
done

echo
echo "🎉 全モデルテスト完了！"
