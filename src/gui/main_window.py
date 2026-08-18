import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QProgressBar, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from src.gui.controllers.pipeline_worker import PipelineWorker


def ndarray_to_qpixmap(img: np.ndarray, size: int = 400) -> QPixmap:
    """Convert a grayscale numpy array to a QPixmap scaled to fit."""
    if img is None:
        return QPixmap()
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


class ImageLabel(QLabel):
    """A QLabel that displays an image with an optional crosshair overlay."""
    def __init__(self, placeholder: str = "No image loaded"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #333; background-color: #0d0d0d;")
        self._marker = None  # (x_frac, y_frac) as fraction of image

    def set_marker(self, x_frac: float, y_frac: float):
        self._marker = (x_frac, y_frac)
        self.update()

    def clear_marker(self):
        self._marker = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._marker and self.pixmap() and not self.pixmap().isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Calculate pixmap position within the label
            pix = self.pixmap()
            pw, ph = pix.width(), pix.height()
            lw, lh = self.width(), self.height()
            ox = (lw - pw) // 2
            oy = (lh - ph) // 2

            cx = ox + int(self._marker[0] * pw)
            cy = oy + int(self._marker[1] * ph)

            # Outer glow
            pen = QPen(QColor(0, 229, 255, 80), 3)
            painter.setPen(pen)
            painter.drawEllipse(cx - 18, cy - 18, 36, 36)

            # Crosshair
            pen = QPen(QColor(0, 229, 255), 2)
            painter.setPen(pen)
            painter.drawLine(cx - 14, cy, cx + 14, cy)
            painter.drawLine(cx, cy - 14, cx, cy + 14)

            # Inner dot
            painter.setBrush(QColor(255, 50, 50))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - 3, cy - 3, 6, 6)

            painter.end()


class ATLASMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATLAS / Drift-Sense Console")
        self.resize(1280, 820)

        self.ref_path = None
        self.search_path = None
        self.ref_img = None
        self.search_img = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # --- Header ---
        header = QLabel("ATLAS  /  Drift-Sense Pipeline Console")
        header.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #00e5ff; "
            "font-family: Consolas, monospace; padding: 4px 0;"
        )
        root_layout.addWidget(header)

        # --- Main content splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ====== LEFT: Images ======
        img_group = QGroupBox("Image Pair")
        img_layout = QVBoxLayout(img_group)

        btn_row = QHBoxLayout()
        self.btn_load_ref = QPushButton("Load Reference")
        self.btn_load_search = QPushButton("Load Search")
        self.btn_load_ref.clicked.connect(self._load_ref)
        self.btn_load_search.clicked.connect(self._load_search)
        btn_row.addWidget(self.btn_load_ref)
        btn_row.addWidget(self.btn_load_search)
        img_layout.addLayout(btn_row)

        pair_row = QHBoxLayout()

        ref_col = QVBoxLayout()
        ref_title = QLabel("Reference (100×)")
        ref_title.setStyleSheet("color: #00e5ff; font-weight: bold;")
        ref_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_display = ImageLabel("No reference loaded")
        ref_col.addWidget(ref_title)
        ref_col.addWidget(self.ref_display)

        search_col = QVBoxLayout()
        search_title = QLabel("Search (10×)")
        search_title.setStyleSheet("color: #00e5ff; font-weight: bold;")
        search_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.search_display = ImageLabel("No search image loaded")
        search_col.addWidget(search_title)
        search_col.addWidget(self.search_display)

        pair_row.addLayout(ref_col)
        pair_row.addLayout(search_col)
        img_layout.addLayout(pair_row)

        splitter.addWidget(img_group)

        # ====== RIGHT: Controls + Results ======
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Run button
        self.btn_run = QPushButton("▶  Run ATLAS Pipeline")
        self.btn_run.setStyleSheet(
            "font-size: 16px; padding: 12px; background-color: #00796b; "
            "color: white; font-weight: bold; border-radius: 4px;"
        )
        self.btn_run.clicked.connect(self._run_pipeline)
        self.btn_run.setEnabled(False)
        right_layout.addWidget(self.btn_run)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { border: 1px solid #333; background: #1a1a1a; height: 8px; } "
            "QProgressBar::chunk { background-color: #00e5ff; }"
        )
        right_layout.addWidget(self.progress)

        # Results group
        results_group = QGroupBox("Results")
        results_grid = QGridLayout(results_group)

        lbl_style = "font-size: 13px; color: #aaa;"
        val_style = "font-size: 18px; font-weight: bold; color: #00e5ff; font-family: Consolas;"

        results_grid.addWidget(self._make_label("Predicted X:", lbl_style), 0, 0)
        self.lbl_x = self._make_label("—", val_style)
        results_grid.addWidget(self.lbl_x, 0, 1)

        results_grid.addWidget(self._make_label("Predicted Y:", lbl_style), 1, 0)
        self.lbl_y = self._make_label("—", val_style)
        results_grid.addWidget(self.lbl_y, 1, 1)

        results_grid.addWidget(self._make_label("Confidence:", lbl_style), 2, 0)
        self.lbl_conf = self._make_label("—", val_style)
        results_grid.addWidget(self.lbl_conf, 2, 1)

        results_grid.addWidget(self._make_label("Runtime:", lbl_style), 3, 0)
        self.lbl_runtime = self._make_label("—", val_style)
        results_grid.addWidget(self.lbl_runtime, 3, 1)

        results_grid.addWidget(self._make_label("Low Info:", lbl_style), 4, 0)
        self.lbl_lowinfo = self._make_label("—", val_style)
        results_grid.addWidget(self.lbl_lowinfo, 4, 1)

        results_grid.addWidget(self._make_label("Tiebreak:", lbl_style), 5, 0)
        self.lbl_tiebreak = self._make_label("—", val_style)
        results_grid.addWidget(self.lbl_tiebreak, 5, 1)

        right_layout.addWidget(results_group)

        # Stage timings log
        timings_group = QGroupBox("Stage Timings")
        timings_layout = QVBoxLayout(timings_group)
        self.timings_log = QTextEdit()
        self.timings_log.setReadOnly(True)
        self.timings_log.setStyleSheet(
            "background-color: #0d0d0d; color: #76ff03; "
            "font-family: Consolas, monospace; font-size: 12px; border: none;"
        )
        self.timings_log.setMaximumHeight(200)
        timings_layout.addWidget(self.timings_log)
        right_layout.addWidget(timings_group)

        right_layout.addStretch()
        splitter.addWidget(right_panel)

        splitter.setSizes([700, 500])
        root_layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _make_label(self, text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _load_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Images (*.png *.tif *.bmp)")
        if path:
            self.ref_path = path
            self.ref_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.ref_display.setPixmap(ndarray_to_qpixmap(self.ref_img))
            self.ref_display.clear_marker()
            self._update_run_btn()

    def _load_search(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Search Image", "", "Images (*.png *.tif *.bmp)")
        if path:
            self.search_path = path
            self.search_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.search_display.setPixmap(ndarray_to_qpixmap(self.search_img))
            self.search_display.clear_marker()
            self._update_run_btn()

    def _update_run_btn(self):
        self.btn_run.setEnabled(self.ref_path is not None and self.search_path is not None)

    def _run_pipeline(self):
        if not self.ref_path or not self.search_path:
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.timings_log.clear()
        self.search_display.clear_marker()

        self.worker = PipelineWorker(self.ref_path, self.search_path)
        self.worker.finished_signal.connect(self._on_result)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_result(self, result):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

        self.lbl_x.setText(f"{result.x:.2f}")
        self.lbl_y.setText(f"{result.y:.2f}")
        self.lbl_conf.setText(f"{result.confidence:.4f}")
        self.lbl_runtime.setText(f"{result.runtime_ms:.0f} ms")
        self.lbl_lowinfo.setText("⚠ YES" if result.is_low_informativeness else "✓ No")
        self.lbl_lowinfo.setStyleSheet(
            f"font-size: 18px; font-weight: bold; font-family: Consolas; "
            f"color: {'#ff5252' if result.is_low_informativeness else '#76ff03'};"
        )
        self.lbl_tiebreak.setText("Yes" if result.tie_break_applied else "No")

        # Stage timings
        lines = []
        for stage, t in result.stage_timings.items():
            bar = "█" * max(1, int(t / 10))
            lines.append(f"  {stage:<30s} {t:>8.1f} ms  {bar}")
        self.timings_log.setText("\n".join(lines))

        # Draw crosshair on search image
        if self.search_img is not None:
            sh, sw = self.search_img.shape[:2]
            x_frac = result.x / sw
            y_frac = result.y / sh
            self.search_display.clear_marker()
            self.search_display.set_marker(x_frac, y_frac)

    def _on_error(self, msg: str):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.timings_log.setText(f"ERROR: {msg}")
