#!/bin/bash

# Start server in background
echo "Starting server..."
cd /Users/kunio/devel/workspace/zcrc-chronos
python -m src.api.server &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Check health endpoint
echo "Checking health endpoint..."
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" http://localhost:8000/api/v1/health)
http_status=$(echo "$response" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
body=$(echo "$response" | sed '/^HTTP_STATUS:/d')

echo "HTTP Status: $http_status"
echo "Response body: $body"

# Also check root endpoint
echo -e "\nChecking root endpoint..."
curl -s http://localhost:8000/

# Kill server
echo -e "\n\nStopping server..."
kill $SERVER_PID

if [ "$http_status" = "200" ]; then
    echo -e "\n✅ Health check successful!"
    exit 0
else
    echo -e "\n❌ Health check failed!"
    exit 1
fi
