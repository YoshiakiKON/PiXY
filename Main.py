"""
Centroid Finder アプリケーションのメインエントリーポイント。

このスクリプトは PyQt5 を使用した GUI アプリケーションを起動します。
主な機能:
- 画像から重心を検出・表示
- 参照点の設定とフィッティング
- 自動デバッグモード (--auto / --auto-exit) で前回画像を自動処理

使用方法:
- 通常起動: python Main.py
- 自動デバッグ: python Main.py --auto
"""

import sys
import os
import base64
from qt_compat.QtWidgets import QApplication, QSplashScreen
from qt_compat.QtGui import QPixmap, QIcon
from qt_compat.QtCore import Qt, QTimer
# qInstallMessageHandler is useful to capture Qt warnings (diagnostic only)
from qt_compat.QtCore import qInstallMessageHandler
from qt_compat.QtGui import QFont
from Ui import CentroidFinderWindow

# Embedded small PNGs (1x1 pixel) as fallbacks. These are tiny dummy assets so
# the repository contains a usable splash and icon even if the user doesn't
# supply images. They will be written to disk on first run if missing.
SPLASH_PNG_B64 = None
ICON_PNG_B64 = None

def _ensure_ppm(path: str, color=(80, 80, 200), w=256, h=128):
    """Create a simple PPM (P6) image at path if it doesn't exist.

    PPM is simple to write and Qt can read it via QPixmap.
    """
    try:
        if os.path.exists(path):
            return path
        header = f"P6\n{w} {h}\n255\n".encode('ascii')
        with open(path, 'wb') as f:
            f.write(header)
            r, g, b = [int(max(0, min(255, int(c)))) for c in color]
            pixel = bytes([r, g, b])
            f.write(pixel * (w * h))
        return path
    except Exception:
        return path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Diagnostic handler: if Qt emits the commitData warning, print a stack so we can find the origin
    try:
        def _qt_msg_handler(msg_type, context, message):
            try:
                s = str(message)
                if 'commitData' in s:
                    import traceback, sys
                    print("[QT_WARNING_CAPTURE]", s, file=sys.stderr)
                    traceback.print_stack(limit=10)
            except Exception:
                pass

        qInstallMessageHandler(_qt_msg_handler)
    except Exception:
        pass
    # Set application-wide font to Segoe UI 12px
    try:
        app.setFont(QFont('Segoe UI', 12))
    except Exception:
        pass

    # Ensure splash/icon files exist (write simple PPM dummies if necessary)
    project_dir = os.path.dirname(__file__)
    # Prefer a bundled PiXY.png for the splash if present
    bundled_png = os.path.join(project_dir, "PiXY_splash.png")
    splash_path = bundled_png if os.path.exists(bundled_png) else os.path.join(project_dir, "splash.ppm")
    # Prefer a bundled PiXY_icon.ico if present, otherwise fall back to a ppm icon asset.
    ico_path = os.path.join(project_dir, "PiXY_icon.ico")
    icon_path = ico_path if os.path.exists(ico_path) else os.path.join(project_dir, "app_icon.ppm")
    # create fallbacks if needed
    if not os.path.exists(splash_path):
        _ensure_ppm(splash_path, color=(50, 100, 200), w=480, h=200)
    # If using ppm fallback, ensure it exists on disk
    if not os.path.exists(icon_path) or icon_path.lower().endswith('.ppm'):
        _ensure_ppm(icon_path, color=(200, 80, 80), w=64, h=64)

    # Create and show splash if asset is available
    splash = None
    try:
        if os.path.exists(splash_path):
            pix = QPixmap(splash_path)
            if not pix.isNull():
                splash = QSplashScreen(pix)
                try:
                    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                except Exception:
                    pass
                splash.showMessage("Initializing...", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
                splash.show()
                try:
                    splash.raise_()
                except Exception:
                    pass
                app.processEvents()
    except Exception:
        splash = None

    # Set app icon (if available)
    try:
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                app.setWindowIcon(icon)
    except Exception:
        pass

    # Create main window (heavy init happens here)
    win = CentroidFinderWindow()
    # Start with a reasonable default that fits most screens (smaller by default)
    win.resize(800, 600)
    # Prevent the initial layout from being smaller than intended
    try:
        win.setMinimumSize(800, 600)
    except Exception:
        pass

    # Show the main window (maximized)
    win.showMaximized()
    
    # CLI auto mode handling
    args = set(arg.lower() for arg in sys.argv[1:])
    if "--auto" in args or "--auto-exit" in args:
        # Close splash immediately in auto mode and run
        if splash is not None:
            app.processEvents()
            try:
                splash.finish(win)
            except Exception:
                pass
        win.run_auto_and_exit()
    else:
        # Load default image in background while splash is showing
        default_image = os.path.join(project_dir, "DemoBMP.bmp")
        if os.path.exists(default_image):
            try:
                # Start loading image immediately (in background during splash)
                QTimer.singleShot(0, lambda: win._open_image_from_path(default_image))
            except Exception:
                pass
        
        # Close splash after 2 seconds
        if splash is not None:
            def finish_splash():
                try:
                    splash.finish(win)
                except Exception:
                    pass
            QTimer.singleShot(2000, finish_splash)

    # Use exec() for Qt6 / PySide6 compatibility (exec_ is deprecated)
    try:
        rv = app.exec()
    except AttributeError:
        rv = app.exec_()
    sys.exit(rv)