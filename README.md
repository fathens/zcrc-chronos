# zcrc-chronos

chronos-boltを使用した時系列予測APIサーバー

## 概要

このプロジェクトは、chronos-boltライブラリを使用して時系列データの予測を行うAPIサーバーを提供します。
FastAPIフレームワークを使用してHTTPインターフェースを実装し、様々なクライアントからの予測リクエストを処理します。

## 環境セットアップ

### 前提条件

- Miniconda または Anaconda がインストールされていること
- Python 3.12 以上

### Miniconda環境のセットアップ

1. リポジトリをクローン

```bash
git clone https://github.com/yourusername/zcrc-chronos.git
cd zcrc-chronos
```

2. Conda環境の作成

```bash
conda create -n zcrc-chronos python=3.12
conda activate zcrc-chronos
```

3. 依存パッケージのインストール

```bash
conda env update -f environment.yml
```

注意: `chronos-bolt`パッケージが公式のPyPIリポジトリに存在しない場合は、適切なソースからインストールしてください。

## プロジェクト構成

```
zcrc-chronos/
├── README.md                    # プロジェクト説明
├── environment.yml              # Conda環境設定ファイル
├── config/                      # 設定ファイル
│   ├── app_config.yaml          # アプリケーション設定
│   └── model_config.yaml        # モデル設定
├── data/                        # データ関連
│   ├── raw/                     # 生データ
│   ├── processed/               # 前処理済みデータ
│   └── models/                  # 学習済みモデル
├── src/                         # ソースコード
│   ├── api/                     # API 関連
│   ├── models/                  # モデル関連
│   ├── preprocessing/           # データ前処理
│   └── utils/                   # ユーティリティ
├── tests/                       # テスト
└── scripts/                     # 実行スクリプト
```

## 使用方法

### サーバーの起動

```bash
# 開発モード
python -m src.api.server

# または
uvicorn src.api.server:app --reload
```

### APIエンドポイント

- `GET /` - APIの基本情報
- `GET /api/v1/health` - ヘルスチェック
- `GET /api/v1/models` - 利用可能なモデル一覧
- `POST /api/v1/predict` - 時系列予測の実行

詳細なAPIドキュメントは、サーバー起動後に `/api/v1/docs` で確認できます。

## 開発

### テストの実行

```bash
pytest
```

### コードスタイル

```bash
# コードフォーマット
black src tests

# リンター
flake8 src tests
```

## ライセンス

[ライセンス情報]

## 貢献

[貢献方法の説明]
