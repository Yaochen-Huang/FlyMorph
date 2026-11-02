"""EyeArea review editor prototype (PySide6).

Run:  pip install PySide6
      python editor.py <folder> [--roi-dir DIR] [--out-dir DIR]

Keys: S = save+next · N = skip · B = back · R = redraw · C = clear
      wheel = zoom · middle-drag = pan · Esc = cancel drawing
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tables import read_table, write_table, rebuild_final    # noqa: E402

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPolygonF, QPen, QColor, QBrush, QPainter, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene,
                               QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsEllipseItem,
                               QMessageBox, QComboBox)

STATE = {"index": 0, "saved": 0, "modified": 0, "skipped": 0}


class Vertex(QGraphicsEllipseItem):
    R = 7

    def __init__(self, pt, editor, idx):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.editor, self.idx = editor, idx
        self.setPos(pt)
        self.setBrush(QBrush(QColor(0, 200, 255)))
        self.setPen(QPen(Qt.black, 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            self.editor.update_vertex(self.idx, self.pos())
        return super().itemChange(change, value)


class Canvas(QGraphicsView):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._pan = False
        self.pix_item = None
        self.poly_item = None
        self.vertices = []
        self.drawing = []
        self.draw_mode = False

    def hard_reset(self):
        """Remove every overlay item (polygon + handles + temp drawing), keep the image.

        Bookkeeping-free: walk the scene instead of trusting vertex/poly references,
        so a stale reference can never leave an old prefill on screen.
        """
        for it in list(self.scene_.items()):
            if it is not self.pix_item:
                try:
                    self.scene_.removeItem(it)
                except Exception:
                    pass
        self.vertices = []
        self.drawing = []
        self.poly_item = None
        self.draw_mode = False

    def load_image(self, path):
        self.scene_.clear()
        self.vertices, self.drawing, self.draw_mode = [], [], False
        pm = QPixmap(path)
        self.pix_item = QGraphicsPixmapItem(pm)
        self.scene_.addItem(self.pix_item)
        self.scene_.setSceneRect(QRectF(pm.rect()))
        self.fitInView(self.pix_item, Qt.KeepAspectRatio)
        self.poly_item = None

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._pan = True
            self.setCursor(Qt.ClosedHandCursor)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton or (self._pan and e.button() == Qt.LeftButton):
            self._pan = True
            self.setCursor(Qt.ClosedHandCursor)
            self._last = e.position()
            return
        if self.draw_mode and e.button() == Qt.LeftButton:
            self.drawing.append(self.mapToScene(e.position().toPoint()))
            self._redraw_temp()
            return
        if e.button() == Qt.LeftButton and not self.draw_mode:
            item = self.itemAt(e.position().toPoint())
            if isinstance(item, Vertex):
                super().mousePressEvent(e)
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan:
            d = e.position() - self._last
            self._last = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton or self._pan:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.draw_mode and len(self.drawing) >= 3:
            self.finish_draw()
            return
        super().mouseDoubleClickEvent(e)

    def _redraw_temp(self):
        if self.poly_item:
            self.scene_.removeItem(self.poly_item)
            self.poly_item = None
        if len(self.drawing) >= 2:
            self.poly_item = QGraphicsPolygonItem(QPolygonF(self.drawing))
            self.poly_item.setPen(QPen(QColor(255, 0, 0), 3))
            self.scene_.addItem(self.poly_item)

    def finish_draw(self):
        self.draw_mode = False
        self.set_vertices(self.drawing)

    def set_vertices(self, pts):
        # remove old handles / polygon for real (never leave an empty item behind)
        for v in list(self.vertices):
            try:
                self.scene_.removeItem(v)
            except Exception:
                pass
        self.vertices = []
        if self.poly_item is not None:
            try:
                self.scene_.removeItem(self.poly_item)
            except Exception:
                pass
        self.poly_item = None
        self.drawing = []
        if not pts:
            return
        self.poly_item = QGraphicsPolygonItem(QPolygonF(pts))
        self.poly_item.setPen(QPen(QColor(255, 0, 0), 3))
        self.scene_.addItem(self.poly_item)
        for i, p in enumerate(pts):
            v = Vertex(p, self, i)
            self.scene_.addItem(v)
            self.vertices.append(v)
        self.drawing = list(pts)

    def update_vertex(self, idx, pos):
        if self.poly_item is not None:
            poly = QPolygonF([v.pos() for v in self.vertices])
            self.poly_item.setPolygon(poly)

    def current_points(self):
        return [(v.pos().x(), v.pos().y()) for v in self.vertices]


class Editor(QMainWindow):
    def __init__(self, folder, out_dir, only=None):
        super().__init__()
        self.folder = folder
        self.out_dir = out_dir
        names = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.files = [f for f in names if only is None or f in set(only)]
        self.canvas = Canvas(self)
        self.setCentralWidget(self._build_ui())
        self.setWindowTitle("EyeArea · 复核编辑器")
        self.resize(1280, 860)
        self._bind_keys()
        self.load_current()

    def _build_ui(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.lbl = QLabel("—")
        bar.addWidget(self.lbl)
        bar.addStretch(1)
        for text, fn in (("重画 (R)", self.redraw), ("清空 (C)", self.clear),
                         ("保存并下一张 (S)", self.save_next), ("下一张 (N)", self.skip),
                         ("上一张 (B)", self.back)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        lay.addLayout(bar)
        lay.addWidget(self.canvas, 1)
        self.status = QLabel("")
        lay.addWidget(self.status)
        return w

    def _bind_keys(self):
        for key, fn in (("S", self.save_next), ("N", self.skip), ("B", self.back),
                        ("R", self.redraw), ("C", self.clear), ("Escape", self.cancel_draw)):
            QShortcut(QKeySequence(key), self, activated=fn)

    def load_current(self):
        if not self.files:
            QMessageBox.information(self, "空", "文件夹里没有图片")
            return
        i = STATE["index"] % len(self.files)
        name = self.files[i]
        self.canvas.load_image(os.path.join(self.folder, name))
        self.lbl.setText("%d/%d  %s" % (i + 1, len(self.files), name))
        roi = os.path.join(self.out_dir, "rois", os.path.splitext(name)[0] + ".json")
        if os.path.exists(roi):
            pts = json.load(open(roi))
            self.canvas.set_vertices([QPointF(x, y) for x, y in pts])
        self._update_status()

    def _update_status(self):
        self.status.setText("已保存 %d · 已修改 %d · 跳过 %d" %
                            (STATE["saved"], STATE["modified"], STATE["skipped"]))

    def save_next(self):
        pts = self.canvas.current_points()
        if len(pts) < 3:
            QMessageBox.warning(self, "无轮廓", "还没有画出轮廓（按 R 开始画）")
            return
        name = self.files[STATE["index"] % len(self.files)]
        base = os.path.splitext(name)[0]
        os.makedirs(os.path.join(self.out_dir, "rois"), exist_ok=True)
        json.dump(pts, open(os.path.join(self.out_dir, "rois", base + ".json"), "w"))
        self.upsert_area(base, shoelace(pts))
        STATE["saved"] += 1
        self.advance()

    def upsert_area(self, base, area):
        """Write the correction to corrections.csv, then regenerate final.csv.

        prefill.csv is never modified: final.csv = prefill + corrections.
        """
        corr = os.path.join(self.out_dir, "corrections.csv")
        order, rows = read_table(corr) if os.path.exists(corr) else ([], {})
        if base not in rows:
            order.append(base)
            rows[base] = {"file": base, "conf": "", "thr": "", "area_px": "", "edited": "1"}
        rows[base]["area_px"] = "%.1f" % area
        rows[base]["edited"] = "1"
        write_table(corr, order, rows)
        rebuild_final(self.out_dir)

    def skip(self):
        STATE["skipped"] += 1
        self.advance()

    def back(self):
        STATE["index"] = max(0, STATE["index"] - 1)
        self.load_current()

    def advance(self):
        STATE["index"] += 1
        if STATE["index"] >= len(self.files):
            QMessageBox.information(self, "完成", "全部图片已处理完")
            STATE["index"] = 0
        self.load_current()

    def redraw(self):
        self.canvas.hard_reset()
        self.canvas.draw_mode = True
        self.status.setText("绘制中：单击加点，双击闭合")

    def clear(self):
        self.canvas.hard_reset()
        self.status.setText("已清空（按 R 重新绘制）")

    def cancel_draw(self):
        self.canvas.draw_mode = False
        self._update_status()


def shoelace(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def main():
    args = sys.argv[1:]
    folder = args[0] if args else QFileDialog.getExistingDirectory(None, "选择图片文件夹")
    if not folder:
        return
    out_dir = os.path.join(os.path.dirname(folder.rstrip("\\/")), "eyearea_out")
    app = QApplication(sys.argv)
    ed = Editor(folder, out_dir)
    ed.show()
    app.exec()


if __name__ == "__main__":
    main()
