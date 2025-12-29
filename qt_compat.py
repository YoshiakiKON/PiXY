"""
Compatibility layer focused on PySide6. PyQt5 fallback was removed per request.
It exposes QtCore, QtGui, QtWidgets module objects and provides compatibility
aliases such as `pyqtSignal` for code that imports that name.

Usage:
    from qt_compat.QtWidgets import QWidget, QLabel
    from qt_compat.QtCore import Qt, QTimer, QObject, pyqtSignal

"""

_using = "PySide6"

try:
    from PySide6 import QtCore as _QtCore, QtGui as _QtGui, QtWidgets as _QtWidgets
    Signal = _QtCore.Signal
    Slot = _QtCore.Slot
    QtCore = _QtCore
    QtGui = _QtGui
    QtWidgets = _QtWidgets
except Exception as exc:
    raise ImportError("PySide6 is required for this application. Please install PySide6.") from exc

# Provide compatibility aliases on the QtCore module for code that imports pyqtSignal/pyqtSlot
# and so on.
try:
    setattr(QtCore, 'pyqtSignal', QtCore.Signal)
    setattr(QtCore, 'pyqtSlot', QtCore.Slot)
except Exception:
    pass

# Exported names for convenience
__all__ = [
    'QtCore', 'QtGui', 'QtWidgets', 'Signal', 'Slot', 'using'
]

using = _using

# For convenience, allow `from qt_compat.QtCore import Qt, QTimer, QObject, QRect, QPoint, pyqtSignal` style imports
# by exposing the QtCore/QtGui/QtWidgets modules as submodules of this package.
# Python's import system will allow `import qt_compat.QtCore` to return this module's attribute.
import sys
sys.modules.setdefault('qt_compat.QtCore', QtCore)
sys.modules.setdefault('qt_compat.QtGui', QtGui)
sys.modules.setdefault('qt_compat.QtWidgets', QtWidgets)

# Also make top-level aliases
Qt = QtCore.Qt
QColor = QtGui.QColor
QPixmap = QtGui.QPixmap
QPainter = QtGui.QPainter
QPalette = QtGui.QPalette

# End of qt_compat
