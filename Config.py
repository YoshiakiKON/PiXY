"""
設定と定数を定義するモジュール。

処理パラメータやファイルパスの管理を行う。
"""

import os

# 設定や定数を記述
PROC_TARGET_WIDTH = 640  # 処理用画像の目標幅 (ピクセル)

# 最後に開いた画像ファイルのパス
LAST_IMAGE_PATH_FILE = "last_image_path.txt"


def save_last_image_path(path):
    """
    最後に開いた画像パスをファイルに保存。

    Args:
        path: 画像ファイルパス
    """
    try:
        with open(LAST_IMAGE_PATH_FILE, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception as e:
        print(f"[Config] 画像パス保存失敗: {e}")


def load_last_image_path():
    """
    最後に開いた画像パスをファイルから読み込み。

    Returns:
        画像ファイルパス (存在しない場合は空文字列)
    """
    try:
        with open(LAST_IMAGE_PATH_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


# デバッグモード: True にするとターミナルへ動作ログを出力する
DEBUG = False

# ログモード: True のときのみ高頻度の INFO/DEBUG ログを有効化する。
# DEBUG とは独立して使えるが、DEBUG=True の場合は自動的に有効。
# 一時的に有効化するには環境変数 PIXY_LOG_MODE=1 でも指定可能。
LOG_MODE = bool(DEBUG) or str(os.environ.get("PIXY_LOG_MODE", "")).strip().lower() in {
    "1", "true", "yes", "on"
}

# Default upper limit for grain area (pixels) used for initial histogram selection
DEFAULT_MAX_GRAIN_AREA = 2000
# Default minimum grain area (pixels) used to avoid too-small auto-selection
DEFAULT_MIN_GRAIN_AREA = 30