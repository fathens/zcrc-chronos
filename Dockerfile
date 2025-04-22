# ベースイメージとしてminicondaを使用
FROM continuumio/miniconda3:latest

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 環境変数を設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Condaの設定
RUN conda config --set always_yes yes \
    && conda config --set auto_update_conda false \
    && conda config --set channel_priority strict

# Conda環境を作成
RUN conda create -n zcrc-chronos python=3.12 -y

# Conda環境をアクティベートするためのシェルを設定
SHELL ["conda", "run", "-n", "zcrc-chronos", "/bin/bash", "-c"]

# environment.ymlをコピーして依存関係をインストール
COPY environment.yml .
RUN conda env update -n zcrc-chronos -f environment.yml

# アプリケーションのコードをコピー
COPY . .

# ポートを公開
EXPOSE 8000

# アプリケーションを実行
CMD ["conda", "run", "--no-capture-output", "-n", "zcrc-chronos", "python", "scripts/run_server.py"]
