#!/usr/bin/env python
"""
APIサーバーを起動するスクリプト
"""

import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.api.server import start_server  # noqa: E402

if __name__ == "__main__":
    print("zcrc-chronos APIサーバーを起動します...")
    start_server()
