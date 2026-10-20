#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI 层: PySide6 (首选) / Tkinter / OpenCV 窗口 / 无界面 多级回退.

UI 线程只读共享 State, 工作线程只写; 拍照/切模式通过 App 句柄下发.
"""
import os
import threading

import cv2


class AppState:
    """线程间共享状态."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None           # 风格化后的显示帧 (BGR)
        self.fps = 0.0
        self.infer_ms = 0.0
        self.mode_label = "原始"
        self.capture_request = False
        self.photo_count = 0
        self.last_photo = ""
        self.events = []            # [(time_str, desc)]
        self.quit = False


class BaseUI:
    def __init__(self, state, app):
        self.state = state
        self.app = app

    def run(self):
        pass

    def close(self):
        pass


# ------------------------------------------------------------------
# PySide6 UI (首选, 现代深色主题)
# ------------------------------------------------------------------
QSS_DARK = """
QMainWindow, QWidget { background: #1e1f24; color: #e8e8ea;
                       font-family: 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                       'Microsoft YaHei', sans-serif; font-size: 12pt; }
QLabel#title  { font-size: 17pt; font-weight: bold; color: #4fc3f7; }
QLabel#subtle { color: #9a9ca3; font-size: 10pt; }
QLabel#group  { color: #9a9ca3; font-size: 10pt; padding: 6px 2px 2px 2px; }
QLabel#status { color: #b8bac0; font-size: 10pt; }
QFrame#panel  { background: #2a2b31; border-radius: 12px; }
QPushButton { background: #33353d; color: #e8e8ea; border: none;
              border-radius: 8px; padding: 8px 6px; font-size: 11pt; }
QPushButton:hover  { background: #3d3f49; }
QPushButton:pressed{ background: #2c2e36; }
QPushButton:checked{ background: #4fc3f7; color: #10222c; font-weight: bold; }
QPushButton#shot   { background: #4fc3f7; color: #10222c;
                     font-size: 14pt; font-weight: bold;
                     border-radius: 12px; padding: 14px; }
QPushButton#shot:hover  { background: #6bd0fa; }
QPushButton#quit   { background: #3a2530; color: #ef9a9a; }
QPushButton#quit:hover  { background: #4a2d3b; }
QLabel#flash      { background: #ffffff; }
QLabel#thumb      { border: 2px solid #3a3c45; border-radius: 8px;
                    background: #101114; }
QLabel#video      { background: #101114; border-radius: 12px; }
"""


class QtUI(BaseUI):
    """PySide6 深色主题界面: 视频区 + 特效网格 + 拍照白闪 + 缩略图."""

    def __init__(self, state, app, window_w=1024, modes=None):
        super().__init__(state, app)
        import sys

        from PySide6 import QtCore, QtGui, QtWidgets
        self.QtCore, self.QtGui, self.QtWidgets = QtCore, QtGui, QtWidgets
        self.modes = modes or ["原始"]
        self.window_w = window_w
        self._last_cnt = -1
        self._flash_t = 0

        qtapp = QtWidgets.QApplication.instance() \
            or QtWidgets.QApplication(sys.argv[:1])
        qtapp.setStyleSheet(QSS_DARK)
        self.qtapp = qtapp
        print("[ui] PySide6 深色界面")

        W = QtWidgets.QMainWindow()
        W.setWindowTitle("风格化特效相机 (RK3576)")
        W.setMinimumSize(window_w + 300, 640)
        self.win = W

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(12, 12, 12, 12)
        h.setSpacing(12)

        # --- 左: 视频区 (黑底 + 白闪叠层) ---
        self.video = QtWidgets.QLabel(objectName="video")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)
        self.flash = QtWidgets.QLabel(self.video, objectName="flash")
        self.flash.hide()
        h.addWidget(self.video, 1)

        # --- 右: 控制面板 ---
        panel = QtWidgets.QFrame(objectName="panel")
        panel.setFixedWidth(270)
        p = QtWidgets.QVBoxLayout(panel)
        p.setContentsMargins(16, 16, 16, 16)
        p.setSpacing(8)

        title = QtWidgets.QLabel("风格相机", objectName="title")
        p.addWidget(title)
        self.status = QtWidgets.QLabel("相机启动中…", objectName="status")
        self.status.setWordWrap(True)
        p.addWidget(self.status)

        # 特效网格: 原始+风格迁移一组, 贴纸一组
        self._btns, self._group = [], QtWidgets.QButtonGroup(self.win)
        self._group.setExclusive(True)
        try:
            from stylizer import MODE_LABELS, STICKER_MODES
            label2key = {v: k for k, v in MODE_LABELS.items()}
        except Exception:
            label2key, STICKER_MODES = {}, ()

        def _add_grid(labels):
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(6)
            for i, lb in enumerate(labels):
                b = QtWidgets.QPushButton(lb.replace(" (NPU)", ""))
                b.setToolTip(lb)
                b.setCheckable(True)
                b.setFocusPolicy(QtCore.Qt.NoFocus)
                b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                QtWidgets.QSizePolicy.Fixed)
                idx = len(self._btns)
                b.clicked.connect(lambda _=False, k=idx: self._on_mode(k))
                self._group.addButton(b, idx)
                self._btns.append(b)
                grid.addWidget(b, i // 3, i % 3)
            p.addLayout(grid)

        plain = [lb for lb in self.modes
                 if label2key.get(lb) not in STICKER_MODES]
        sticker = [lb for lb in self.modes
                   if label2key.get(lb) in STICKER_MODES]
        g1 = QtWidgets.QLabel("风格特效", objectName="group")
        p.addWidget(g1)
        _add_grid(plain)
        if sticker:
            g2 = QtWidgets.QLabel("趣味贴纸", objectName="group")
            p.addWidget(g2)
            _add_grid(sticker)
        if self._btns:                      # 默认选中 "原始"
            self._btns[0].setChecked(True)

        p.addStretch(1)

        shot = QtWidgets.QPushButton("拍照 (空格)", objectName="shot")
        shot.setFocusPolicy(QtCore.Qt.NoFocus)
        shot.clicked.connect(self._shot)
        p.addWidget(shot)
        quitb = QtWidgets.QPushButton("退出", objectName="quit")
        quitb.setFocusPolicy(QtCore.Qt.NoFocus)
        quitb.clicked.connect(self._on_close)
        p.addWidget(quitb)

        self.thumb = QtWidgets.QLabel(objectName="thumb")
        self.thumb.setFixedSize(238, 90)
        self.thumb.setAlignment(QtCore.Qt.AlignCenter)
        self.thumb.setText("拍照后此处显示缩略图")
        self.thumb.mousePressEvent = lambda _e: self._open_photo()
        p.addWidget(self.thumb)
        self.photolb = QtWidgets.QLabel("照片: 0", objectName="status")
        p.addWidget(self.photolb)

        h.addWidget(panel)
        W.setCentralWidget(central)

        # 空格拍照
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), W,
                         activated=self._shot)

        # 33ms 刷新定时器
        self.timer = QtCore.QTimer(W)
        self.timer.timeout.connect(self._tick)

    # ---------------- 回调 ----------------
    def _on_mode(self, idx):
        self.app.set_mode_by_index(idx)

    def _shot(self):
        self.app.capture()

    def _on_close(self):
        self.state.quit = True
        self.win.close()

    def _open_photo(self):
        d = os.path.abspath(getattr(self.app, "save_dir", "."))
        # sudo/root 运行时没有桌面用户的会话环境, QDesktopServices 会静默失败;
        # 且 gio open 的 D-Bus 激活往返实测有 ~7s 延迟, 故 root 下优先直启文件管理器
        if os.geteuid() == 0 and self._open_dir_direct(d):
            return
        if QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d)):
            return
        if self._open_dir_via_bus(d):
            return
        # 兜底: 全部失败时在状态栏显示照片目录路径
        self.photolb.setText(f"照片目录: {d}")

    @staticmethod
    def _open_dir_direct(d):
        # root 下直接以桌面用户身份拉起文件管理器 (绕过 gio/dbus 激活慢路径)
        import subprocess
        import glob
        import pwd
        import shutil
        if not shutil.which("nautilus"):
            return False
        for rt in sorted(glob.glob("/run/user/[0-9]*")):
            try:
                user = pwd.getpwuid(int(os.path.basename(rt))).pw_name
            except (KeyError, ValueError):
                continue
            try:
                subprocess.Popen(
                    ["runuser", "-u", user, "--", "env",
                     "XDG_RUNTIME_DIR=" + rt, "WAYLAND_DISPLAY=wayland-0",
                     "DBUS_SESSION_BUS_ADDRESS=unix:path=" + rt + "/bus",
                     "nautilus", "--new-window", d],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _open_dir_via_bus(d):
        # 借用桌面用户会话总线兜底 (Popen 非阻塞, 避免卡 UI)
        import subprocess
        import glob
        for bus in sorted(glob.glob("/run/user/*/bus")):
            try:
                subprocess.Popen(
                    ["xdg-open", d],
                    env=dict(os.environ,
                             DBUS_SESSION_BUS_ADDRESS="unix:path=" + bus),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
        return False

    # ---------------- 主循环 ----------------
    def run(self):
        self.win.show()
        self.timer.start(33)
        self.qtapp.exec()

    def _tick(self):
        st = self.state
        if st.quit:
            self.qtapp.quit()
            return
        QtCore = self.QtCore
        QtGui = self.QtGui
        with st.lock:
            frame = None if st.frame is None else st.frame.copy()
            fps, ms = st.fps, st.infer_ms
            mode, cnt = st.mode_label, st.photo_count
            last = st.last_photo
            events = list(st.events)

        # 视频帧 -> QImage -> 缩放显示
        if frame is not None:
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(rgb.data, fw, fh, fw * 3,
                               QtGui.QImage.Format_RGB888).copy()
            pm = QtGui.QPixmap.fromImage(img).scaled(
                self.video.width(), self.video.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.video.setPixmap(pm)

        self.status.setText(f"{mode}   {fps:.1f} fps / {ms:.0f} ms")
        self.photolb.setText(f"照片: {cnt}")

        # 拍照: 白闪 + 缩略图 + 记录
        if cnt != self._last_cnt:
            self._last_cnt = cnt
            self._flash_t = 4
        if self._flash_t > 0:
            self._flash_t -= 1
            self.flash.setGeometry(0, 0, self.video.width(),
                                   self.video.height())
            self.flash.show()
            self.flash.setStyleSheet(
                f"background: rgba(255,255,255,{0.75 * self._flash_t / 4:.2f});"
                "border-radius: 12px;")
        else:
            self.flash.hide()
        if last and self._flash_t == 3:      # 新照片落盘后刷新缩略图
            pm = QtGui.QPixmap(last)
            if not pm.isNull():
                self.thumb.setPixmap(pm.scaled(
                    self.thumb.width(), self.thumb.height(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))

        # 保存记录 (窗口标题式, 最近 3 条)
        tip = "  |  ".join(e for _, e in reversed(events[-3:]))
        self.win.setWindowTitle(f"风格化特效相机  {tip}" if tip
                                else "风格化特效相机")


# ------------------------------------------------------------------
# Tkinter UI (PySide6 不可用时回退)
# ------------------------------------------------------------------
class TkUI(BaseUI):
    def __init__(self, state, app, window_w=1024, modes=None):
        super().__init__(state, app)
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("风格化特效相机 Demo (RK3576)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        from PIL import Image, ImageTk
        self.PIL = (Image, ImageTk)
        self.window_w = window_w
        self.modes = modes or ["原始"]

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side="top", fill="both", expand=True)

        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="特效:").pack(side="left", padx=(8, 2))
        self.mode_var = tk.StringVar(value=self.modes[0])
        cb = ttk.Combobox(bar, textvariable=self.mode_var, state="readonly",
                          values=list(self.modes), width=14)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._on_mode)

        self.fps_var = tk.StringVar(value="FPS: -  特效: -ms")
        ttk.Label(bar, textvariable=self.fps_var).pack(side="left", padx=10)

        self.photo_var = tk.StringVar(value="照片: 0")
        ttk.Label(bar, textvariable=self.photo_var,
                  font=("Sans", 11, "bold")).pack(side="left", padx=8)

        btns = ttk.Frame(bar)
        btns.pack(side="right")
        shot = ttk.Button(btns, text="拍照 (空格)", command=self._shot)
        shot.pack(side="left", padx=4)
        shot.focus_set()
        self.root.bind("<space>", lambda e: self._shot())
        ttk.Button(btns, text="退出", command=self._on_close).pack(
            side="left", padx=4)

        log_frame = ttk.LabelFrame(self.root, text="保存记录", padding=4)
        log_frame.pack(side="bottom", fill="x")
        self.listbox = tk.Listbox(log_frame, height=4)
        self.listbox.pack(fill="x")

    def _on_mode(self, _evt=None):
        idx = self.modes.index(self.mode_var.get()) \
            if self.mode_var.get() in self.modes else 0
        self.app.set_mode_by_index(idx)

    def _shot(self):
        self.app.capture()

    def _on_close(self):
        self.state.quit = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.after(33, self._tick)
        self.root.mainloop()

    def _tick(self):
        if self.state.quit:
            # shutdown (SIGTERM/退出按钮) 后销毁窗口, 让 mainloop 正常返回
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        st = self.state
        with st.lock:
            frame = None if st.frame is None else st.frame.copy()
            fps, ms = st.fps, st.infer_ms
            mode = st.mode_label
            cnt = st.photo_count
            events = list(st.events)

        if frame is not None:
            Image, ImageTk = self.PIL
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (self.window_w, int(h * self.window_w / w)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self._photo)

        self.fps_var.set(f"FPS: {fps:.1f}  特效: {ms:.0f}ms  模式: {mode}")
        self.photo_var.set(f"照片: {cnt}")

        if self.listbox.size() != len(events):
            self.listbox.delete(0, "end")
            for t, e in reversed(events):
                self.listbox.insert("end", f"[{t}] {e}")

        self.root.after(33, self._tick)


# ------------------------------------------------------------------
# OpenCV 窗口 UI (无 tkinter/PIL 时回退)
# ------------------------------------------------------------------
class CvUI(BaseUI):
    def __init__(self, state, app, window_w=1024, modes=None):
        super().__init__(state, app)
        self.modes = modes or ["原始"]
        self.window_w = window_w
        self.win = "style_cam_demo (空格=拍照, s=切换特效, q=退出)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    def run(self):
        import time
        idx = 0
        while not self.state.quit:
            with self.state.lock:
                frame = None if self.state.frame is None \
                    else self.state.frame.copy()
                mode = self.state.mode_label
            if frame is None:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    self.state.quit = True
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (self.window_w, int(h * self.window_w / w)))
            cv2.putText(frame, f"{mode}   photos={self.state.photo_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2)
            cv2.imshow(self.win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.state.quit = True
            elif key == ord("s"):
                idx = (idx + 1) % len(self.modes)
                self.app.set_mode_by_index(idx)
            elif key == ord(" "):
                self.app.capture()
        cv2.destroyAllWindows()

    def close(self):
        cv2.destroyAllWindows()


class HeadlessUI(BaseUI):
    def run(self):
        import time
        last = ""
        while not self.state.quit:
            with self.state.lock:
                mode, cnt, fps = (self.state.mode_label,
                                  self.state.photo_count, self.state.fps)
            line = f"[headless] mode={mode} photos={cnt} fps={fps:.1f}"
            if line != last:
                print(line)
                last = line
            time.sleep(1)


def make_ui(state, app, ui_cfg, no_ui=False, modes=None):
    if no_ui:
        return HeadlessUI(state, app)
    try:
        import PySide6  # noqa
        return QtUI(state, app, ui_cfg.get("window_w", 1024), modes)
    except Exception as e:
        print(f"[ui] PySide6 不可用({e}), 回退到 Tkinter")
        try:
            import tkinter  # noqa
            import PIL  # noqa
            return TkUI(state, app, ui_cfg.get("window_w", 1024), modes)
        except Exception as e2:
            print(f"[ui] Tkinter 不可用({e2}), 回退到 OpenCV 窗口")
            try:
                if not os.environ.get("DISPLAY"):
                    raise RuntimeError("无 DISPLAY, 使用无界面模式")
                return CvUI(state, app, ui_cfg.get("window_w", 1024), modes)
            except Exception as e3:
                print(f"[ui] OpenCV 窗口不可用({e3}), 使用无界面模式")
                return HeadlessUI(state, app)
