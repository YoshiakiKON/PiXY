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


# =====================================================================
# Aggressiveness Presets
# =====================================================================

def load_aggressiveness_presets():
    """
    Aggressiveness レベル (0-10) に対応する 3 パラメータを
    pixy_settings.ini から読み込む。

    Returns:
        dict: {0: (trim, neck_sep, shape_complex), 1: (...), ..., 10: (...)}
        読み込み失敗の場合はデフォルト値を返す。
    """
    import configparser

    presets = {}
    settings_file = "pixy_settings.ini"

    # デフォルト値（案D: ステップ型）
    DEFAULT_PRESETS = {
        0: (0, 0, 3),
        1: (1, 0, 3),
        2: (2, 1, 3),
        3: (3, 1, 4),
        4: (4, 2, 4),
        5: (5, 3, 5),
        6: (6, 4, 5),
        7: (7, 5, 6),
        8: (8, 6, 7),
        9: (9, 7, 8),
        10: (10, 8, 9),
    }

    try:
        if not os.path.exists(settings_file):
            if DEBUG:
                print(f"[Config] {settings_file} が見つかりません。デフォルト値を使用します。")
            return DEFAULT_PRESETS

        config = configparser.ConfigParser()
        config.read(settings_file, encoding="utf-8")

        if not config.has_section("Aggressiveness"):
            if DEBUG:
                print(f"[Config] [Aggressiveness] セクションが見つかりません。デフォルト値を使用します。")
            return DEFAULT_PRESETS

        # preset_0 から preset_10 を読み込む
        for level in range(11):
            key = f"preset_{level}"
            if config.has_option("Aggressiveness", key):
                values_str = config.get("Aggressiveness", key)
                # "0, 0, 3" → (0, 0, 3)
                values = tuple(int(v.strip()) for v in values_str.split(","))
                if len(values) == 3:
                    presets[level] = values
                else:
                    if DEBUG:
                        print(f"[Config] {key} の値が不正です（3つの値が必要）。デフォルト値を使用します。")
                    presets[level] = DEFAULT_PRESETS[level]
            else:
                if DEBUG:
                    print(f"[Config] {key} が見つかりません。デフォルト値を使用します。")
                presets[level] = DEFAULT_PRESETS[level]

        if DEBUG:
            print(f"[Config] Aggressiveness プリセットを読み込みました:")
            for level, (trim, neck_sep, shape_complex) in sorted(presets.items()):
                print(f"  Level {level}: trim={trim}, neck_sep={neck_sep}, shape_complex={shape_complex}")

        return presets

    except Exception as e:
        print(f"[Config] Aggressiveness プリセット読み込み失敗: {e}")
        return DEFAULT_PRESETS


def load_ui_defaults():
    """
    UI のデフォルト設定を pixy_settings.ini から読み込む。

    Returns:
        dict: {
            'default_aggressiveness': int (0-10),
            'unified_control_default': bool,
        }
        読み込み失敗の場合はデフォルト値を返す。
    """
    import configparser

    defaults = {
        "default_aggressiveness": 5,
        "unified_control_default": True,
    }

    settings_file = "pixy_settings.ini"

    try:
        if not os.path.exists(settings_file):
            return defaults

        config = configparser.ConfigParser()
        config.read(settings_file, encoding="utf-8")

        if config.has_section("UI"):
            if config.has_option("UI", "default_aggressiveness"):
                defaults["default_aggressiveness"] = config.getint("UI", "default_aggressiveness")
            if config.has_option("UI", "unified_control_default"):
                defaults["unified_control_default"] = config.getboolean(
                    "UI", "unified_control_default"
                )

        if DEBUG:
            print(f"[Config] UI デフォルト設定を読み込みました: {defaults}")

        return defaults

    except Exception as e:
        print(f"[Config] UI デフォルト設定読み込み失敗: {e}")
        return defaults


# グローバルに読み込み
AGGRESSIVENESS_PRESETS = load_aggressiveness_presets()
UI_DEFAULTS = load_ui_defaults()