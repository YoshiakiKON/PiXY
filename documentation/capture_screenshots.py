"""
Automatic screenshot capture script

Usage:
    python documentation\capture_screenshots.py

Run this script in a local environment. It launches the app and exports screenshots of the main screens to
`documentation/images/` (generates quick_jp_1..4.png / quick_en_1..4.png).
"""
import sys
import os
from qt_compat.QtWidgets import QApplication
from qt_compat.QtCore import QTimer

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
img_dir = os.path.join(proj_root, 'documentation', 'images')
os.makedirs(img_dir, exist_ok=True)

try:
    from Ui import CentroidFinderWindow
except Exception as e:
    print('Failed to import CentroidFinderWindow:', e)
    sys.exit(1)


def save(widget, name):
    p = os.path.join(img_dir, name)
    try:
        pm = widget.grab()
        pm.save(p)
        print('Saved', p)
    except Exception as e:
        print('Failed save', p, e)


def main():
    app = QApplication(sys.argv)
    win = CentroidFinderWindow()
    win.resize(1024, 720)
    win.show()

    # Try to find a demo image (project root DemoBMP.bmp or last_image_path.txt)
    demo_candidates = [
        os.path.join(proj_root, 'DemoBMP.bmp'),
        os.path.join(proj_root, 'DemoBMP.png'),
        os.path.join(proj_root, 'PiXY_splash.png'),
        os.path.join(proj_root, 'splash.png'),
    ]
    demo_path = None
    for c in demo_candidates:
        if os.path.exists(c):
            demo_path = c
            break
    if not demo_path:
        # try last_image_path.txt
        last_txt = os.path.join(proj_root, 'last_image_path.txt')
        try:
            if os.path.exists(last_txt):
                with open(last_txt, 'r', encoding='utf-8') as f:
                    p = f.read().strip()
                    if os.path.exists(p):
                        demo_path = p
        except Exception:
            pass

    def step_open():
        if demo_path:
            try:
                win._open_image_from_path(demo_path)
            except Exception as e:
                print('open image failed', e)
        QTimer.singleShot(1500, step_capture_main)

    def step_capture_main():
        save(win, 'quick_jp_1.png')
        save(win, 'quick_en_1.png')
        QTimer.singleShot(1500, step_capture_detected)

    def step_capture_detected():
        save(win, 'quick_jp_2.png')
        save(win, 'quick_en_2.png')
        QTimer.singleShot(500, step_pick_mode)

    def step_pick_mode():
        try:
            win._start_pick_mode('add')
        except Exception:
            pass
        save(win, 'quick_jp_3.png')
        save(win, 'quick_en_3.png')
        QTimer.singleShot(500, step_table)

    def step_table():
        # capture reference table / export area
        save(win, 'quick_jp_4.png')
        save(win, 'quick_en_4.png')
        QTimer.singleShot(500, finish)

    def finish():
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, step_open)
    rv = app.exec()
    sys.exit(rv)


if __name__ == '__main__':
    main()
