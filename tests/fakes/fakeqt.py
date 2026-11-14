"""Faux PySide6 / pyqtgraph pour exécuter la logique de l'interface sans écran.

Vérifie l'enchaînement de NOTRE code (réglages, accord, mémoires...), pas l'usage
réel de Qt : seul un lancement sur un vrai poste valide l'affichage.
"""
import sys
import types


class _Meta(type):
    def __getattr__(cls, name):                 # constantes de classe de Qt (QPlainTextEdit.LineWrapMode...)
        if name.startswith("_"):
            raise AttributeError(name)
        return Any()


class Any(metaclass=_Meta):
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Any()
    def __call__(self, *a, **k): return Any()


class Signal:
    def __init__(self): self.slots = []
    def connect(self, f): self.slots.append(f)
    def emit(self, *a):
        for f in list(self.slots): f(*a)


class Widget(Any):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: Any()


class Valued(Widget):
    def __init__(self, *a, **k):
        self._v, self.valueChanged, self._lo, self._hi = 0, Signal(), -10**12, 10**12
        self._enabled = True
    def setEnabled(self, on): self._enabled = bool(on)
    def isEnabled(self): return self._enabled
    def setRange(self, lo, hi):
        self._lo, self._hi = lo, hi
        if not lo <= self._v <= hi:
            self.setValue(self._v)
    def value(self): return self._v
    def setValue(self, v):
        v = max(self._lo, min(self._hi, int(v)))
        if v != self._v:
            self._v = v; self.valueChanged.emit(v)


class DoubleValued(Valued):
    def setValue(self, v):
        v = max(self._lo, min(self._hi, float(v)))
        if v != self._v:
            self._v = v; self.valueChanged.emit(v)


class Button(Widget):
    def __init__(self, text="", *a, **k):
        self._c, self.text_, self.toggled, self.clicked = False, text, Signal(), Signal()
    def isChecked(self): return self._c
    def setEnabled(self, on): self._enabled = bool(on)
    def isEnabled(self): return getattr(self, "_enabled", True)
    def setChecked(self, c):
        if bool(c) != self._c:
            self._c = bool(c); self.toggled.emit(self._c)
    def setText(self, t): self.text_ = t
    def click(self): self.clicked.emit(False)


class Combo(Widget):
    def __init__(self, *a, **k): self.items, self._i, self.currentIndexChanged = [], -1, Signal()
    def addItem(self, label, data=None):
        self.items.append((label, data))
        if self._i < 0: self.setCurrentIndex(0)
    def clear(self): self.items, self._i = [], -1
    def currentIndex(self): return self._i
    def setCurrentIndex(self, i):
        if i != self._i:
            self._i = i; self.currentIndexChanged.emit(i)
    def itemData(self, i): return self.items[i][1]
    def count(self): return len(self.items)
    def setItemData(self, i, value, role=None): pass
    def currentData(self): return self.items[self._i][1] if 0 <= self._i < len(self.items) else None
    def findData(self, d): return next((i for i, it in enumerate(self.items) if it[1] == d), -1)


class ViewBox(Widget):
    def __init__(self): self.range = [0.0, 1.0]
    def viewRange(self): return [list(self.range), [-120.0, 0.0]]


class PlotWidget(Widget):
    def __init__(self, *a, **k): self.vb = ViewBox()
    def getViewBox(self): return self.vb
    def setXRange(self, a, b, padding=0): self.vb.range = [float(a), float(b)]
    def wheelEvent(self, ev): self.zoomed = True


class Label(Widget):
    def __init__(self, text="", *a, **k): self.text_ = text
    def setText(self, t): self.text_ = t


class TextEdit(Widget):
    def __init__(self, *a, **k): self.text_ = ""
    def setPlainText(self, t): self.text_ = t
    def clear(self): self.text_ = ""


class ListWidget(Widget):
    def __init__(self, *a, **k): self.rows, self._row, self.itemDoubleClicked = [], -1, Signal()
    def count(self): return len(self.rows)
    def clear(self): self.rows = []
    def addItem(self, s): self.rows.append(s)
    def currentRow(self): return self._row
    def row(self, item): return item


class MainWindowBase(Widget):
    def __init__(self, *a, **k): pass


class FileDialog:
    path = ""
    save_path = ""
    @staticmethod
    def getOpenFileName(*a, **k): return FileDialog.path, ""
    @staticmethod
    def getSaveFileName(*a, **k): return FileDialog.save_path, ""


class MessageBox:
    messages = []
    @staticmethod
    def warning(parent, title, text): MessageBox.messages.append((title, text))
    @staticmethod
    def information(parent, title, text): MessageBox.messages.append((title, text))


class InputDialog:
    text, number = "Essai", 1000
    @staticmethod
    def getText(*a, **k): return InputDialog.text, True
    @staticmethod
    def getInt(*a, **k): return InputDialog.number, True


def install():
    qtw = types.SimpleNamespace(
        QMainWindow=MainWindowBase, QSpinBox=Valued, QDoubleSpinBox=DoubleValued, QSlider=Valued, QPushButton=Button, QCheckBox=Button,
        QComboBox=Combo, QListWidget=ListWidget, QLabel=Label, QRadioButton=Button, QPlainTextEdit=TextEdit, QFileDialog=FileDialog, QMessageBox=MessageBox,
        QInputDialog=InputDialog)
    qtw_mod = types.ModuleType("PySide6.QtWidgets")
    qtw_mod.__getattr__ = lambda name: getattr(qtw, name) if hasattr(qtw, name) else Widget
    qtc_mod = types.ModuleType("PySide6.QtCore")
    qtc_mod.__getattr__ = lambda name: Any()
    pyside = types.ModuleType("PySide6")
    qtg_mod = types.ModuleType("PySide6.QtGui")
    qtg_mod.__getattr__ = lambda name: Any()
    pyside.QtWidgets, pyside.QtCore, pyside.QtGui = qtw_mod, qtc_mod, qtg_mod
    sys.modules["PySide6.QtGui"] = qtg_mod
    pg = types.ModuleType("pyqtgraph")
    pg.__getattr__ = lambda name: PlotWidget if name == "PlotWidget" else Any()
    sys.modules.update({"PySide6": pyside, "PySide6.QtWidgets": qtw_mod, "PySide6.QtCore": qtc_mod, "pyqtgraph": pg})
    return FileDialog, MessageBox
