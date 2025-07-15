# ローカルDocker環境での実行

このディレクトリには、アプリケーションをローカルのDocker環境で実行するためのファイルが含まれています。

## 使用方法

### 前提条件
- Docker
- Docker Compose

### 起動
```bash
cd run_local
./run.sh
```

### 停止
```bash
cd run_local
docker compose down
```

## ファイル構成

- `run.sh`: Docker Composeを使用してアプリケーションを起動するスクリプト
- `docker-compose.yml`: ローカル開発用のDocker Compose設定ファイル
- `../Dockerfile`: アプリケーションのDockerイメージビルド用ファイル

## アクセス方法

アプリケーションが起動すると、以下のURLでアクセスできます：

- API: http://localhost:8000
- API ドキュメント: http://localhost:8000/api/v1/docs
- ヘルスチェック: http://localhost:8000/api/v1/health
