"""
UI文字列とラベルを一元管理するモジュール。

頻繁に変更されるUI要素をここに集約し、
保守性を向上させる。
"""

# Centralized UI strings and labels

# App title
APP_TITLE = "PiXY"

# ── 2-step workflow labels ──────────────────────────────────────────────────
STEP1_LABEL = "Step 1: Off-line Targeting"
STEP1_SUBLABEL = "Load image & specify target points before using the instrument"
STEP2_LABEL = "Step 2: On-line Alignment"
STEP2_SUBLABEL = "Register fiducial points on the instrument, then export coordinates"

# Auto-detect (centroid extraction) section — treated as an auxiliary tool
SECTION_AUTO_DETECT = "Auto-detect (Auxiliary)"
SECTION_AUTO_DETECT_HINT = "Optional: auto-extract target candidates from image"

# Button labels
BUTTON_OPEN_IMAGE = "Open Image"
BUTTON_EXPORT_CENTROIDS = "Export Centroids"
BUTTON_TOGGLE_BOUNDARIES = "Boundaries"
BUTTON_ADD_REF = "Add Fiducial Point"
BUTTON_ADD_REF_CANCEL = "Cancel"
BUTTON_UPDATE_XY = "Update u, v"
BUTTON_CLEAR = "Clear"

# Flip labels
FLIP_PREFIX = "Flip:"
FLIP_LABELS = {
    'auto': 'Auto',
    'normal': 'Normal',
    'flip': 'Flip',
}

# Control/slider names
NAME_OVERLAY_RATIO = "Display Mode"
NAME_POSTERLEVEL = "PosterLevel"
NAME_MIN_AREA = "Min Area"
NAME_TRIM = "Trim (px)"

# Table row labels
TABLE_LEFT_ROW_LABELS = [
    "u",
    "v",
    "Stage.\nX",
    "Stage.\nY",
    "Stage.\nZ",
    "Res.\nX",
    "Res.\nY",
    "Res.\nZ",
    "Res.\n|R|",
    "",
]

TABLE_RIGHT_ROW_LABELS = [
    "u",
    "v",
    "Calc.\nX",
    "Calc.\nY",
    "Calc.\nZ",
]

# Dialogs and messages
OPEN_DIALOG_TITLE = "Select Image File"
FILE_FILTER = "Image Files (*.jpg *.jpeg *.png *.bmp)"

# Tooltips
FLIP_TOOLTIP = "Manual flip mode: cycle Auto → Normal → Flip"

# Export
EXPORT_FILENAME_PREFIX = "centroids_"
EXPORT_HEADER = "No,Group,Stage X,Stage Y,Stage Z\n"
