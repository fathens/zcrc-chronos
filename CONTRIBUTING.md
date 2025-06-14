# CONTRIBUTING.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

**重要**: このファイルは継続的に更新してください。新しいコマンド、アーキテクチャの変更、開発プロセスの改善があった場合は、必ずこのファイルを最新の状態に保ってください。

## プロジェクト概要

AutoGluon-TimeSeriesライブラリを使用した日本語対応の時系列予測APIサーバーです。FastAPI経由でゼロショット時系列予測のHTTPエンドポイントを提供し、自動データ正規化、補間処理、Swagger UIによる包括的なAPIドキュメントを含みます。

## アーキテクチャ

- **FastAPIアプリケーション**: 自動ドキュメント生成機能付きメインAPIサーバー
- **時系列予測器**: AutoGluon-TimeSeriesのラッパークラス
- **データ正規化**: 複数の補間手法による自動時系列データ前処理
- **設定システム**: アプリとモデル設定のYAMLベース管理
- **コンテナ化デプロイ**: conda環境管理によるDocker環境

## 必須コマンド

### 開発環境セットアップ
```bash
# conda環境の作成とアクティベート
conda create -n zcrc-chronos python=3.12
conda activate zcrc-chronos

# 依存関係のインストール
conda env update -f environment.yml
```

### アプリケーション実行
```bash
# 開発モード（自動リロード付き）
python scripts/run_server.py
# または
uvicorn src.api.server:app --reload

# 本番モード
python -m src.api.server
```

### テスト実行
```bash
# 全テスト実行
pytest

# 特定のテストファイル実行
pytest tests/test_api.py
pytest tests/test_models.py

# 実ライブラリを使用（モックなし）
NEED_REAL_LIBRARY=true pytest
```

### コード品質チェック
```bash
# コードフォーマット
black src tests

# コードスタイルチェック
flake8 src tests

# インポート文整理
isort src tests
```

### Dockerデプロイ
```bash
# ローカルデプロイ
cd run_local
./run.sh

# サービス停止
docker compose down
```

## 主要コンポーネント

### APIルート (`src/api/routes.py`)
- **POST /api/v1/predict_zero_shot**: 包括的バリデーション付きメイン予測エンドポイント
- **GET /api/v1/models**: 利用可能モデル一覧取得
- **GET /api/v1/health**: ヘルスチェックエンドポイント
- 自動補間手法選択による高度なデータ正規化ロジックを含む

### 時系列予測器 (`src/models/predictor.py`)
- 広範囲なエラーハンドリング付きAutoGluon-TimeSeriesラッパー
- データサイズに基づく動的パラメータ調整機能
- モデルフィッティング失敗時のフォールバック機構
- 信頼区間抽出サポート

### 設定ファイル
- `config/app_config.yaml`: サーバー、API、ログ、セキュリティ設定
- `config/model_config.yaml`: モデルパラメータ、前処理、特徴量エンジニアリング設定

## 開発ノート

### テスト戦略
- 広範囲なモック機能付きpytestを使用
- 実ライブラリ vs モックライブラリの使用を環境変数で制御：
  - `NEED_REAL_LIBRARY=true`: 実際のAutoGluonライブラリを使用
  - `CAN_SKIP_REAL_LIBRARY=true`: 実ライブラリテストをスキップ
- `tests/conftest.py`にchronos-boltライブラリ用モックフィクスチャを配置

### データ処理
- 複数の補間手法による自動時系列正規化
- データ特性に基づくインテリジェントな補間手法選択
- 様々なデータ品質問題に対する堅牢なエラーハンドリング

### APIドキュメント
実行時に以下で利用可能：
- Swagger UI: `/api/v1/docs`
- ReDoc: `/api/v1/redoc`

### ログ機能
app_config.yamlで定義されたファイルローテーションとリテンションポリシー付きloguru構造化ログを使用。

### Docker環境
- Python 3.12付きminiconda3ベース
- ヘルスチェックと適切なログ設定を含む
- データ、ログ、設定の永続化用ボリュームマウント