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

# 依存関係のインストール（テスト用パッケージも含む）
conda env update -f environment.yml
```

**重要**: 以下の全てのコマンドは `conda activate zcrc-chronos` でconda環境をアクティベートしてから実行してください。

**注意**: `environment.yml`には開発・テストに必要な全てのパッケージ（pytest, fastapi, uvicorn, httpx等）が含まれています。

#### トラブルシューティング
`conda env update`が失敗またはタイムアウトした場合：
```bash
# 個別にautogluonをインストール
pip install autogluon.timeseries>=1.3.0

# または段階的にインストール
pip install pytest fastapi uvicorn httpx pyyaml loguru pandas numpy
pip install autogluon.timeseries
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

#### 基本テスト実行
```bash
# 全テスト実行
pytest

# 特定のテストファイル実行
pytest tests/test_api.py
pytest tests/test_models.py

# 実ライブラリを使用（モックなし）
NEED_REAL_LIBRARY=true pytest
```

#### predict_zero_shot エンドポイントのテスト

**オプション1: 簡単なテスト実行（推奨）**
```bash
# 簡単なテストスクリプトの実行（autogluonモックを自動適用）
python test_predict_simple.py
```

**オプション2: フルテストスイート実行**
```bash
# predict_zero_shot の包括的テスト実行
pytest tests/test_predict_zero_shot.py -v

# 特定のテストクラス実行
pytest tests/test_predict_zero_shot.py::TestPredictZeroShotValidInputs -v
```

**注意**: オプション2にはautogluon.timeseriesライブラリが必要です（`conda env update -f environment.yml`で自動インストール）

**依存関係の確認:**
```bash
# autogluonが正しくインストールされているか確認
python -c "import autogluon.timeseries; print('autogluon.timeseries available')"

# インストールされていない場合
pip install autogluon.timeseries>=1.3.0
```

**テスト内容:**
- `test_predict_simple.py`: 基本的な予測機能とエラーハンドリングの動作確認
- `tests/test_predict_zero_shot.py`: 正常入力、無効入力、エッジケース、統合テストの包括的テストスイート

**注意事項:**
- 実際のautogluonライブラリでは、最小データポイント（2点）のテストが失敗する可能性があります（頻度推論の制限）
- これは技術的制限であり、実際のAPI機能には影響しません

### コード品質チェック

**重要**: コミット前に必ず以下の全てのチェックを実行し、すべてエラーなしで通ることを確認してください。

```bash
# 1. コードフォーマット（自動修正）
python -m black .
python -m isort .

# 2. フォーマット確認（変更が必要な場合はエラー）
python -m black --check .
python -m isort --check-only .

# 3. コードスタイルチェック（エラーが出た場合は修正が必要）
python -m flake8 .

# 4. テスト実行（すべてのテストが通ることを確認）
python -m pytest tests/ -v
```

**CI/CDパイプライン**: 
- プルリクエスト作成時に上記のチェックが自動実行されます
- すべてのチェックが通らない限りマージできません
- コミット前にローカルで実行することを強く推奨します

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

#### 主要テストファイル
- `tests/test_predict_zero_shot.py`: predict_zero_shotエンドポイントの包括的テストスイート
  - 正常入力テスト（基本予測、最小データポイント、異なる予測期間、不規則間隔データ等）
  - 無効入力テスト（データ不足、長さ不一致、過去の予測時点、時間間隔エラー等）
  - エッジケーステスト（信頼区間、評価指標、大データセット、極端値等）
  - 統合テスト（TimeSeriesPredictor連携、例外処理、レスポンス形式検証等）
- `test_predict_simple.py`: autogluonライブラリなしでの基本動作確認用スクリプト
  - 自動モック設定により依存関係なしでテスト実行可能
  - 基本的な予測機能とエラーハンドリングの動作確認

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