#!/bin/bash

# 簡単なモデルテストスクリプト
# 使用方法: ./quick_test.sh <model_name>

MODEL_NAME="${1:-chronos_default}"
BASE_VALUE=$((RANDOM % 100 + 100))

echo "🧪 Testing $MODEL_NAME (base: $BASE_VALUE)..."

TASK_ID=$(curl -s -X POST "http://localhost:8000/api/v1/predict_zero_shot_async" \
    -H "Content-Type: application/json" \
    -d "{
        \"timestamp\": [
            \"2023-01-01T00:00:00\",
            \"2023-01-01T01:00:00\",
            \"2023-01-01T02:00:00\",
            \"2023-01-01T03:00:00\",
            \"2023-01-01T04:00:00\",
            \"2023-01-01T05:00:00\"
        ],
        \"values\": [$BASE_VALUE, $((BASE_VALUE+1)), $((BASE_VALUE-1)), $((BASE_VALUE+2)), $((BASE_VALUE+1)), $((BASE_VALUE+3))],
        \"forecast_until\": \"2023-01-01T08:00:00\",
        \"model_name\": \"$MODEL_NAME\"
    }" | jq -r '.task_id')

echo "📋 Task ID: $TASK_ID"

sleep 3
echo "📊 Result: Check server logs for model selection details"

echo "✅ Done"
