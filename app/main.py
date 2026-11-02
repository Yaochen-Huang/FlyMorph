"""M1 prototype: pick folder -> batch prefill -> (optional) review editor -> CSV."""
import hashlib
import os
import subprocess
import sys
import tempfile
import time

from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QProgressBar, QTableWidget,
                               QTableWidgetItem, QFileDialog, QMessageBox, QCheckBox)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # flymorph/
sys.path.insert(0, HERE)

from core.infer import EyeModel, run_batch          # noqa: E402
from core.vein import VeinPipeline                   # noqa: E402
import editor as editor_mod                         # noqa: E402

BASE = getattr(sys, "_MEIPASS", HERE)                 # works frozen (PyInstaller) and from source


def _find_model(name):
    """Locate a model file: next to the exe (frozen), next to this script, or in a
    `models/` directory a few levels up (works for both repo layouts)."""
    candidates = [os.environ.get("FLYMORPH_MODEL", os.environ.get("EYEAREA_MODEL", "")),
                  os.path.join(BASE, name),
                  os.path.join(HERE, name)]
    d = HERE
    for _ in range(4):
        d = os.path.dirname(d)
        candidates.append(os.path.join(d, "models", name))
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(BASE, name)


MODEL = _find_model("eye_unet_v4.onnx")

APP_NAME = "FlyMorph"
APP_VERSION = "1.0.6"


def _find_asset(name):
    """Icon lives next to the exe (frozen) or under agent_work/assets (dev)."""
    for p in (os.path.join(BASE, name),
              os.path.join(HERE, name),
              os.path.join(HERE, "..", "..", "assets", name)):
        if os.path.exists(p):
            return os.path.abspath(p)
    return ""


# task label -> (key used in output folder / model env var, model filename, enabled)
# task label -> (key, model file, enabled, output folder name)
TASKS = {
    "眼面积": ("eye", "eye_unet_v4.onnx", True, "eye"),
    "翅膀面积": ("wing", "wing_unet_v1.onnx", True, "wing"),
    "翅脉面积": ("vein", "vein_unet_v6.onnx", True, "vein"),
    "蛹面积": ("pupa", "pupa_unet_v1.onnx", True, "pupa"),
}


def _install_crash_log():
    """Write uncaught exceptions to crash.log next to the exe (or this file)."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else HERE

    def hook(tp, val, tb):
        import traceback
        try:
            with open(os.path.join(base, "crash.log"), "a", encoding="utf-8") as fh:
                fh.write("\n=== %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                traceback.print_exception(tp, val, tb, file=fh)
        except Exception:
            pass
        traceback.print_exception(tp, val, tb)
    sys.excepthook = hook


class Worker(QThread):
    tick = Signal(int, int, str)
    done = Signal(list, list)

    def __init__(self, model, folder, out_dir):
        super().__init__()
        self.model, self.folder, self.out_dir = model, folder, out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        rows, flagged = run_batch(self.model, self.folder, self.out_dir,
                                  progress=lambda i, n, name: self.tick.emit(i, n, name),
                                  cancel=lambda: self._cancel)
        self.done.emit(rows, flagged)


def _default_out_root():
    """Somewhere the user certainly can write: Documents/FlyMorph_output."""
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    base = docs if os.path.isdir(docs) else os.path.expanduser("~")
    return os.path.join(base, "FlyMorph_output")


MODE_LABEL = {v[0]: v[3] for v in TASKS.values()}


def _norm_path(path):
    """Show one separator style only: backslash on Windows, slash elsewhere."""
    if not path:
        return path
    if sys.platform.startswith("win"):
        return path.replace("/", "\\")
    return path.replace("\\", "/")


def _fallback_root():
    """A location the user is guaranteed to be able to write without elevation."""
    for env in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(env)
        if base and os.path.isdir(base):
            return os.path.join(base, "FlyMorph")
    return os.path.join(os.path.expanduser("~"), "FlyMorph")


def _out_dir_for(root, folder, key):
    """<root>/<mode>_output, e.g. .../eye_output, .../wing_output."""
    label = MODE_LABEL.get(key, key)
    return os.path.join(root, "%s_output" % label)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("%s v%s · 果蝇表型测量" % (APP_NAME, APP_VERSION))
        self.resize(980, 700)
        self.folder = ""
        self.out_dir = ""
        self.settings = QSettings("FlyMorph", "FlyMorph")
        try:
            self.out_root = str(self.settings.value("out_root", "") or "")
        except Exception:
            self.out_root = ""
        self.model = None
        self.model_key = ""
        self.worker = None
        self.rows = []
        self.flagged = []

        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.btn_dir = QPushButton("选择图片文件夹…")
        self.btn_dir.clicked.connect(self.pick_dir)
        self.lbl_dir = QLabel("（未选择）")
        top.addWidget(self.btn_dir)
        top.addWidget(self.lbl_dir, 1)
        lay.addLayout(top)

        mid = QHBoxLayout()
        mid.addWidget(QLabel("任务"))
        self.cmb_task = QComboBox()
        self.cmb_task.addItems([k for k, v in TASKS.items() if v[2]])
        mid.addWidget(self.cmb_task)
        mid.addWidget(QLabel("模式"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["全部复核", "离群值复核", "不复核"])
        mid.addWidget(self.cmb_mode)
        self.btn_run = QPushButton("开始")
        self.btn_run.clicked.connect(self.start)
        mid.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        mid.addWidget(self.btn_cancel)
        mid.addStretch(1)
        lay.addLayout(mid)

        outrow = QHBoxLayout()
        self.btn_open_in = QPushButton("打开输入文件夹")
        self.btn_open_in.clicked.connect(lambda: self.open_folder(self.folder))
        self.btn_open_out = QPushButton("打开输出文件夹")
        self.btn_open_out.clicked.connect(lambda: self.open_folder(self.out_dir))
        self.btn_out_root = QPushButton("设置输出位置…")
        self.btn_out_root.clicked.connect(self.pick_out_root)
        outrow.addWidget(self.btn_open_in)
        outrow.addWidget(self.btn_open_out)
        outrow.addWidget(self.btn_out_root)
        outrow.addStretch(1)
        lay.addLayout(outrow)

        self.lbl_out = QLabel(self._out_label())
        self.lbl_out.setWordWrap(True)
        lay.addWidget(self.lbl_out)

        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.status = QLabel("就绪")
        lay.addWidget(self.status)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["文件", "面积(px)", "置信度"])
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)

        self.setCentralWidget(w)

    def task_key(self):
        label = self.cmb_task.currentText()
        return TASKS.get(label, ("eye", "", True))[0]

    def open_folder(self, path):
        """Reveal a folder in the OS file manager."""
        if not path or not os.path.isdir(path):
            QMessageBox.information(self, "提示", "还没有可打开的文件夹")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)                      # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "打开失败", "%s" % e)

    def pick_out_root(self):
        start = self.out_root or self.folder or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "选择输出位置（留空=与图片文件夹相同）", start)
        if d:
            self.out_root = d
            self._save_out_root()
            self.lbl_out.setText(self._out_label())
            self.status.setText("输出位置已改为：%s" % _norm_path(self.out_root))

    def _out_label(self):
        if self.out_root:
            return "输出位置：%s" % _norm_path(os.path.join(self.out_root, "<模式>_output"))
        return "输出位置：%s" % _norm_path(os.path.join("图片文件夹", "<模式>_output"))
    def _save_out_root(self):
        try:
            self.settings.setValue("out_root", self.out_root)
        except Exception:
            pass

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if d:
            self.folder = d
            self.lbl_dir.setText(_norm_path(d))
            if not self.out_root:
                self.lbl_out.setText(self._out_label())

    def _prepare_out_dir(self, key):
        """Create the output folder; if it is not writable, warn and abort the run.

        The output goes next to the images (or to the folder chosen with
        "设置输出位置…"). There is deliberately no silent fallback: writing results
        somewhere the user does not expect would be worse than stopping.
        """
        root = self.out_root or self.folder
        target = _out_dir_for(root, self.folder, key)
        if self._writable(target):
            self.out_dir = target
            return True
        QMessageBox.critical(
            self, "无法写入输出目录",
            "无法在该位置创建输出文件夹：\n%s\n\n"
            "任务已中止，没有生成任何结果。\n\n"
            "解决办法：\n"
            "  1. 点“设置输出位置…”换一个可写的文件夹；或\n"
            "  2. 把图片复制到你有写权限的位置后重新运行。" % _norm_path(target))
        return False

    @staticmethod
    def _writable(target):
        try:
            os.makedirs(target, exist_ok=True)
            probe = os.path.join(target, ".write_test")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
            return True
        except OSError:
            return False

    def start(self):
        if not self.folder:
            QMessageBox.warning(self, "提示", "先选择图片文件夹")
            return
        key, model_file, _enabled, _label = TASKS.get(
            self.cmb_task.currentText(), ("eye", "eye_unet_v4.onnx", True, "眼"))
        model_path = _find_model(model_file)
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "缺少模型", "找不到模型文件：\n%s" % model_path)
            return
        if not self._prepare_out_dir(key):
            return
        if self.model is None or self.model_key != key:
            self.status.setText("加载模型…")
            QApplication.processEvents()
            if key == "wing":
                # wing model was validated at a single 0.5 threshold
                self.model = EyeModel(model_path, thr_big=0.50, thr_small=0.50, small_frac=0.0)
            elif key == "pupa":
                # pupa: 768 input, threshold 0.90 (swept on the 20-image test set:
                # bias 0.0%, |dev| median 1.0%, max 3.7%)
                self.model = EyeModel(model_path, thr_big=0.90, thr_small=0.90, small_frac=0.0)
            elif key == "vein":
                # two-stage: wing model -> crop -> vein model (1280 crops, 332 labels)
                # thr 0.40 + spur removal (kernel 13): |dev| 2.4%, p90 7.1%
                wing_path = _find_model("wing_unet_v1.onnx")
                self.model = VeinPipeline(wing_path, model_path, vein_thr=0.40,
                                          open_kernel=13)
            else:
                self.model = EyeModel(model_path)
            self.model_key = key
        self.table.setRowCount(0)
        self.bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.worker = Worker(self.model, self.folder, self.out_dir)
        self.worker.tick.connect(self.on_tick)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()

    def on_tick(self, i, n, name):
        self.bar.setMaximum(n)
        self.bar.setValue(i)
        self.status.setText("%d/%d  %s" % (i, n, name))

    def on_done(self, rows, flagged):
        self.rows, self.flagged = rows, flagged
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(r["file"]))
            self.table.setItem(i, 1, QTableWidgetItem("%.0f" % r["area"]))
            self.table.setItem(i, 2, QTableWidgetItem("%.3f" % r["conf"]))
        self.status.setText("完成：%d 张，复核 %d 张 → %s" % (len(rows), len(flagged), self.out_dir))

        mode = self.cmb_mode.currentText()
        if mode == "不复核":
            return
        if mode == "离群值复核":
            subset = list(flagged)
            if not subset:
                QMessageBox.information(self, "无需复核", "没有发现离群样本")
                return
            if not self.ask_open_review(subset):
                return
        else:
            # 全部复核（默认）
            if not self.ask_open_review(None):
                return

    def ask_open_review(self, subset):
        msg = "进入复核页？" if subset is None else "复核队列 %d 张，进入复核页？" % len(subset)
        if QMessageBox.question(self, "复核", msg) != QMessageBox.Yes:
            return False
        app = QApplication.instance()
        self.ed = editor_mod.Editor(self.folder, self.out_dir, only=subset)
        self.ed.show()
        app.exec()
        self.reload_final()
        return True

    def reload_final(self):
        """After the review window closes, show final.csv values (prefill + corrections)."""
        import csv as _csv
        path = os.path.join(self.out_dir, "final.csv")
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8-sig") as fh:
            rows = list(_csv.DictReader(fh))
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(r.get("file", "")))
            try:
                area = "%.0f" % float(r.get("area_px") or 0)
            except Exception:
                area = r.get("area_px", "")
            self.table.setItem(i, 1, QTableWidgetItem(area))
            self.table.setItem(i, 2, QTableWidgetItem("已修改" if r.get("edited") == "1" else "预标"))
        self.status.setText("复核完成：%d 行（final.csv 已更新）" % len(rows))


def main():
    _install_crash_log()
    print("%s v%s  model=%s  exists=%s" % (APP_NAME, APP_VERSION, MODEL, os.path.exists(MODEL)))
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    ico = _find_asset("app.ico")
    png = _find_asset("logo_256.png")
    if ico:
        app.setWindowIcon(QIcon(ico))
    elif png:
        app.setWindowIcon(QIcon(png))
    m = Main()
    if ico:
        m.setWindowIcon(QIcon(ico))
    elif png:
        m.setWindowIcon(QIcon(png))
    m.show()
    app.exec()


if __name__ == "__main__":
    main()
