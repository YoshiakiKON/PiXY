"""
UI文字列とラベルを一元管理するモジュール。

頻繁に変更されるUI要素をここに集約し、
保守性を向上させる。
"""

# Centralized UI strings and labels

# App title
APP_TITLE = "Px2XY"

# Button labels
BUTTON_OPEN_IMAGE = "Open Image"
BUTTON_EXPORT_CENTROIDS = "Export Centroids"
BUTTON_TOGGLE_BOUNDARIES = "Boundaries"
BUTTON_ADD_REF = "Add Ref. Point"
BUTTON_UPDATE_XY = "Update XY"
BUTTON_CLEAR = "Clear"

# Flip labels
FLIP_PREFIX = "Flip:"
FLIP_LABELS = {
    'auto': 'Auto',
    'normal': 'Normal',
    'flip': 'Flip',
}

# Control/slider names
NAME_OVERLAY_RATIO = "Overlay Ratio"
NAME_POSTERLEVEL = "PosterLevel"
NAME_MIN_AREA = "Min Area"
NAME_TRIM = "Trim (px)"

# Table row labels
TABLE_LEFT_ROW_LABELS = [
    "X",
    "Y",
    "Obs.\nX",
    "Obs.\nY",
    "Obs.\nZ",
    "Res.\nX",
    "Res.\nY",
    "Res.\nZ",
    "Res.\n|R|",
]

TABLE_RIGHT_ROW_LABELS = [
    "X",
    "Y",
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
EXPORT_HEADER = "GroupNo,X,Y\n"
