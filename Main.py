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
import json
import time
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


def _run_wheel_selftest(win, app):
    """Inject synthetic wheel events into the image viewport and quit.

    Enable with environment variable PIXY_SELFTEST_WHEEL=1.
    This is intended for debugging responsiveness/regressions without manual GUI input.
    """
    try:
        from qt_compat.QtCore import QEvent, QPoint
    except Exception:
        return

    class _DummyWheelEvent:
        def __init__(self, dy_steps: float, pos: QPoint):
            self._dy_steps = float(dy_steps)
            self._pos = QPoint(pos)

        def type(self):
            return QEvent.Wheel

        def angleDelta(self):
            return QPoint(0, int(round(120.0 * self._dy_steps)))

        def pixelDelta(self):
            return QPoint(0, 0)

        def modifiers(self):
            return Qt.NoModifier

        def pos(self):
            return QPoint(self._pos)

    class _DummyMouseEvent:
        def __init__(self, etype: QEvent.Type, pos: QPoint, button=None):
            self._etype = etype
            self._pos = QPoint(pos)
            self._button = button

        def type(self):
            return self._etype

        def pos(self):
            return QPoint(self._pos)

        def button(self):
            # Only meaningful for press/release
            return self._button

    def _inject():
        try:
            vp = win.proc_scroll.viewport()
            # Aim at viewport center
            try:
                pos = QPoint(int(vp.width() // 2), int(vp.height() // 2))
            except Exception:
                pos = QPoint(10, 10)

            # Burst of wheel events (zoom in then out)
            for _ in range(18):
                win.interactions.eventFilter(vp, _DummyWheelEvent(+1.0, pos))
            for _ in range(10):
                win.interactions.eventFilter(vp, _DummyWheelEvent(-1.0, pos))

            # Simulate a drag pan (grab image and move)
            try:
                p0 = QPoint(pos)
                p1 = QPoint(pos.x() + 80, pos.y() + 60)
                p2 = QPoint(pos.x() + 140, pos.y() + 110)
                win.interactions.eventFilter(vp, _DummyMouseEvent(QEvent.MouseButtonPress, p0, button=Qt.LeftButton))
                win.interactions.eventFilter(vp, _DummyMouseEvent(QEvent.MouseMove, p1, button=None))
                win.interactions.eventFilter(vp, _DummyMouseEvent(QEvent.MouseMove, p2, button=None))
                win.interactions.eventFilter(vp, _DummyMouseEvent(QEvent.MouseButtonRelease, p2, button=Qt.LeftButton))
            except Exception:
                pass
        except Exception:
            pass

    def _quit():
        try:
            app.quit()
        except Exception:
            pass

    # Give the window time to show/layout, then inject twice (spaced out so the
    # profiler has time to report), then quit.
    QTimer.singleShot(400, _inject)
    QTimer.singleShot(1400, _inject)
    QTimer.singleShot(2800, _quit)

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
    # When frozen by PyInstaller, bundled data files are extracted to
    # `sys._MEIPASS`. Prefer that path when available so bundled images
    # (PiXY_splash.png / PiXY_icon.ico) are found inside the one-file exe.
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        project_dir = sys._MEIPASS
    else:
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
    # Note: On Windows, the taskbar icon can depend on the window icon as well,
    # so we keep the loaded icon object and also apply it to the main window.
    app_icon = None
    try:
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                app.setWindowIcon(icon)
                app_icon = icon
    except Exception:
        app_icon = None

    # Create main window (heavy init happens here)
    win = CentroidFinderWindow()
    try:
        if app_icon is not None:
            win.setWindowIcon(app_icon)
    except Exception:
        pass
    # Start with a reasonable default that fits most screens (smaller by default)
    win.resize(800, 600)
    # Prevent the initial layout from being smaller than intended
    try:
        win.setMinimumSize(800, 600)
    except Exception:
        pass

    # Show the main window (maximized)
    win.showMaximized()

    # Self-test mode: inject wheel events and exit.
    try:
        if bool(str(os.environ.get('PIXY_SELFTEST_WHEEL', '')).strip()):
            if splash is not None:
                app.processEvents()
                try:
                    splash.finish(win)
                except Exception:
                    pass
            _run_wheel_selftest(win, app)
            try:
                rv = app.exec()
            except AttributeError:
                rv = app.exec_()
            sys.exit(rv)
    except Exception:
        pass
    
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
        # Priority: demo image(s) -> last opened image
        default_candidates = []
        startup_last_path = ""
        try:
            # Look for demo images next to this script (worktree) and also in the repo root.
            # This helps when running from a git worktree that does not carry the demo assets.
            repo_root = None
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
            except Exception:
                repo_root = None

            demo_names = [
                "DemoBSE.png",
                "DemoBSE.PNG",
                "Demo.png",
                "Demo.PNG",
                "DemoBMP.bmp",
            ]

            for name in demo_names:
                default_candidates.append(os.path.join(project_dir, name))
                if repo_root:
                    default_candidates.append(os.path.join(repo_root, name))
        except Exception:
            pass
        try:
            from Config import load_last_image_path
            startup_last_path = load_last_image_path()
            if startup_last_path:
                default_candidates.append(startup_last_path)
        except Exception:
            pass

        try:
            # Deduplicate while preserving order
            _seen = set()
            default_candidates = [p for p in default_candidates if p and not (p in _seen or _seen.add(p))]
        except Exception:
            pass

        # Startup safety guard for auto-loading last image
        STARTUP_MAX_LAST_IMAGE_MB = 120
        STARTUP_MAX_LAST_IMAGE_LOAD_SEC = 3.0
        STARTUP_BLOCK_HOURS = 24

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            guard_dir = os.path.dirname(sys.executable)
        else:
            guard_dir = os.path.dirname(__file__)
        startup_guard_file = os.path.join(guard_dir, "startup_image_guard.json")

        def _norm_path(p):
            try:
                return os.path.normcase(os.path.abspath(str(p)))
            except Exception:
                return str(p)

        def _load_startup_guard():
            try:
                if not os.path.isfile(startup_guard_file):
                    return {}
                with open(startup_guard_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        def _save_startup_guard(data):
            try:
                with open(startup_guard_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _load_startup_default_or_prompt():
            loaded = False
            guard = _load_startup_guard()
            now_ts = float(time.time())
            changed_guard = False
            last_norm = _norm_path(startup_last_path) if startup_last_path else ""

            # cleanup expired guard entries
            try:
                stale = []
                for gp, meta in guard.items():
                    try:
                        until_ts = float((meta or {}).get('blocked_until', 0))
                    except Exception:
                        until_ts = 0.0
                    if until_ts <= now_ts:
                        stale.append(gp)
                for gp in stale:
                    guard.pop(gp, None)
                    changed_guard = True
            except Exception:
                pass

            for p in default_candidates:
                try:
                    if p and os.path.isfile(p):
                        p_norm = _norm_path(p)
                        is_last_candidate = bool(last_norm and p_norm == last_norm)

                        # Safety conditions for auto-loading the last opened image
                        if is_last_candidate:
                            meta = guard.get(p_norm, {}) if isinstance(guard, dict) else {}
                            try:
                                blocked_until = float((meta or {}).get('blocked_until', 0))
                            except Exception:
                                blocked_until = 0.0
                            if blocked_until > now_ts:
                                continue

                            try:
                                sz = int(os.path.getsize(p))
                            except Exception:
                                sz = 0
                            if sz > int(STARTUP_MAX_LAST_IMAGE_MB * 1024 * 1024):
                                guard[p_norm] = {
                                    'blocked_until': now_ts + float(STARTUP_BLOCK_HOURS * 3600),
                                    'reason': f'too_large>{STARTUP_MAX_LAST_IMAGE_MB}MB',
                                }
                                changed_guard = True
                                continue

                        t0 = time.perf_counter()
                        ok = win._open_image_from_path(p, show_startup_prompt_on_fail=False)
                        dt = float(time.perf_counter() - t0)

                        if is_last_candidate and (not bool(ok) or dt > float(STARTUP_MAX_LAST_IMAGE_LOAD_SEC)):
                            guard[p_norm] = {
                                'blocked_until': now_ts + float(STARTUP_BLOCK_HOURS * 3600),
                                'reason': 'load_failed' if not bool(ok) else f'slow_startup>{STARTUP_MAX_LAST_IMAGE_LOAD_SEC:.1f}s ({dt:.2f}s)',
                            }
                            changed_guard = True

                        if bool(ok):
                            loaded = True
                            break
                except Exception:
                    continue
            if changed_guard:
                _save_startup_guard(guard)
            if not loaded:
                try:
                    win._show_open_image_prompt_message()
                except Exception:
                    pass

        try:
            # Start loading image immediately (in background during splash)
            QTimer.singleShot(0, _load_startup_default_or_prompt)
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