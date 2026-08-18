"""
Module defining settings and constants.

Manages processing parameters and file paths.
"""

import os

# Define settings and constants
PROC_TARGET_WIDTH = 640  # Target processing image width (pixels)

# Path to the last opened image file
LAST_IMAGE_PATH_FILE = "last_image_path.txt"


def save_last_image_path(path):
    """
    Save the last opened image path to a file.

    Args:
        path: Image file path
    """
    try:
        with open(LAST_IMAGE_PATH_FILE, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception as e:
        print(f"[Config] Failed to save image path: {e}")


def load_last_image_path():
    """
    Load the last opened image path from a file.

    Returns:
        Image file path (empty string if unavailable)
    """
    try:
        with open(LAST_IMAGE_PATH_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


# Debug mode: when True, writes runtime logs to the terminal
DEBUG = False

# Log mode: enables high-frequency INFO/DEBUG logging only when True.
# Independent of DEBUG, but automatically enabled when DEBUG=True.
# Can also be enabled temporarily via the environment variable PIXY_LOG_MODE=1.
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
    Load the 3 parameters corresponding to Aggressiveness levels (0-10)
    from pixy_settings.ini.

    Returns:
        dict: {0: (trim, neck_sep, shape_complex), 1: (...), ..., 10: (...)}
        Returns default values if loading fails.
    """
    import configparser

    presets = {}
    settings_file = "pixy_settings.ini"

    # Default values (proposal D: stepped type)
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
                print(f"[Config] {settings_file} not found. Using default values.")
            return DEFAULT_PRESETS

        config = configparser.ConfigParser()
        config.read(settings_file, encoding="utf-8")

        if not config.has_section("Aggressiveness"):
            if DEBUG:
                print(f"[Config] [Aggressiveness] section not found. Using default values.")
            return DEFAULT_PRESETS

        # preset_0  preset_10 
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
                        print(f"[Config] {key} value is invalid (3 values required). Using default values.")
                    presets[level] = DEFAULT_PRESETS[level]
            else:
                if DEBUG:
                    print(f"[Config] {key} not found. Using default values.")
                presets[level] = DEFAULT_PRESETS[level]

        if DEBUG:
            print(f"[Config] Aggressiveness presets loaded:")
            for level, (trim, neck_sep, shape_complex) in sorted(presets.items()):
                print(f"  Level {level}: trim={trim}, neck_sep={neck_sep}, shape_complex={shape_complex}")

        return presets

    except Exception as e:
        print(f"[Config] Failed to load Aggressiveness presets: {e}")
        return DEFAULT_PRESETS


def load_ui_defaults():
    """
    Load the UI default settings from pixy_settings.ini.

    Returns:
        dict: {
            'default_aggressiveness': int (0-10),
            'unified_control_default': bool,
        }
        Returns default values if loading fails.
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
            print(f"[Config] UI defaults loaded: {defaults}")

        return defaults

    except Exception as e:
        print(f"[Config] Failed to load UI defaults: {e}")
        return defaults


# Load globally
AGGRESSIVENESS_PRESETS = load_aggressiveness_presets()
UI_DEFAULTS = load_ui_defaults()