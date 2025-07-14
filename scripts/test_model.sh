#!/bin/bash

# モデルテスト用のシェルスクリプト
# 使用方法: ./test_model.sh <model_name> [base_value]

set -e

MODEL_NAME="${1:-chronos_default}"
BASE_VALUE="${2:-100}"
API_URL="http://localhost:8000/api/v1"

# 引数チェック
if [[ -z "$MODEL_NAME" ]]; then
    echo "使用方法: $0 <model_name> [base_value]"
    echo "例: $0 autoets_only 100"
    exit 1
fi

# テストデータを生成
generate_test_data() {
    local base=$1
    local timestamp_start="2023-$(printf "%02d" $((RANDOM % 12 + 1)))-01T00:00:00"

    cat <<EOF
{
    "timestamp": [
        "${timestamp_start}",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+1H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+2H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+3H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+4H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+5H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+6H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
        "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+7H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")"
    ],
    "values": [
        $base.0,
        $((base + 1)).5,
        $((base - 1)).8,
        $((base + 2)).1,
        $((base + 1)).9,
        $((base + 3)).2,
        $((base + 1)).4,
        $((base + 4)).0
    ],
    "forecast_until": "$(date -j -f "%Y-%m-%dT%H:%M:%S" -v+11H "$timestamp_start" "+%Y-%m-%dT%H:%M:%S")",
    "model_name": "$MODEL_NAME"
}
EOF
}

echo "🚀 モデル $MODEL_NAME をテスト中..."
echo "📊 ベース値: $BASE_VALUE"

# API リクエスト送信
TASK_ID=$(generate_test_data "$BASE_VALUE" | curl -s -X POST "$API_URL/predict_zero_shot_async" \
    -H "Content-Type: application/json" \
    -d @- | jq -r '.task_id')

if [[ "$TASK_ID" == "null" || -z "$TASK_ID" ]]; then
    echo "❌ リクエスト失敗"
    exit 1
fi

echo "📝 タスクID: $TASK_ID"
echo "⏳ 処理待機中..."

# 結果確認
sleep 3
echo "📋 ログ確認:"
docker logs zcrc-chronos --tail 15 | grep -E "(選択されたモデル設定.*$MODEL_NAME|Models trained|Best model|フォールバック)" | tail -3

echo "✅ モデル $MODEL_NAME のテスト完了"
