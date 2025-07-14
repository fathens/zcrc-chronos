"""
APIサーバーの設定と初期化を行うモジュール
"""

import os

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router

# 設定ファイルのパス
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "app_config.yaml",
)


def load_config():
    """
    アプリケーション設定を読み込む
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise HTTPException(
            status_code=500, detail="サーバー設定の読み込みに失敗しました"
        )


# アプリケーション設定の読み込み
config = load_config()


# loguruの設定
def configure_logging():
    """
    loguruのログ設定を行う
    """
    try:
        # テスト環境かどうかを判定
        is_test_env = os.getenv("PYTEST_CURRENT_TEST") is not None

        if is_test_env:
            # テスト環境では最小限のログ設定
            logger.remove()
            logger.add(
                sink=lambda msg: None,  # テスト時はログを無効化
                format="{time} | {level} | {message}",
                level="ERROR",  # エラーレベル以上のみ
                catch=False,  # エラーをキャッチしない
            )
            return

        # プロジェクトルートのパスを取得
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        log_file_path = os.path.join(project_root, config["logging"]["file"])

        # ディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)

        # 既存のハンドラーを削除
        logger.remove()

        # コンソール出力を追加
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format=config["logging"]["format"],
            level=config["logging"]["level"],
            catch=False,  # エラーをキャッチしない
        )

        # ファイル出力を追加
        logger.add(
            sink=log_file_path,
            format=config["logging"]["format"],
            level=config["logging"]["level"],
            rotation=config["logging"]["rotation"],
            retention=config["logging"]["retention"],
            encoding="utf-8",
            catch=False,  # エラーをキャッチしない
        )

        logger.info(f"ログ設定完了: {log_file_path}")

    except Exception:
        # 設定が失敗した場合はデフォルトのロガーを使用
        logger.remove()
        logger.add(lambda msg: None, level="ERROR", catch=False)


# ログ設定を実行（テスト時以外）
if os.getenv("PYTEST_CURRENT_TEST") is None:
    configure_logging()

# FastAPIアプリケーションの初期化
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
    docs_url=f"{config['api']['prefix']}/docs",
    redoc_url=f"{config['api']['prefix']}/redoc",
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(router, prefix=config["api"]["prefix"])


# ルートエンドポイント
@app.get("/")
async def root():
    """
    ルートエンドポイント - APIの基本情報を返す
    """
    return {
        "name": config["api"]["title"],
        "version": config["api"]["version"],
        "description": config["api"]["description"],
        "docs": f"{config['api']['prefix']}/docs",
    }


# ヘルスチェックエンドポイント
@app.get(f"{config['api']['prefix']}/health")
async def health_check():
    """
    ヘルスチェックエンドポイント - APIの状態を返す
    """
    return {
        "status": "healthy",
        "api_version": config["api"]["version"],
    }


# 注：ゼロショット予測のエンドポイントはroutes.pyに移動しました
# routes.pyの/predict_zero_shotエンドポイントを使用してください


# サーバー起動関数
def start_server():
    """
    uvicornサーバーを起動する
    """
    uvicorn.run(
        "src.api.server:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["debug"],
        log_level=config["logging"]["level"].lower(),
        timeout_keep_alive=2000,  # 33分20秒のkeep-aliveタイムアウト
        timeout_graceful_shutdown=60,  # 60秒のグレースフルシャットダウン
    )


if __name__ == "__main__":
    # 直接実行された場合はサーバーを起動
    start_server()
