#!/bin/bash
# fall_detect_demo 板端安装脚本 (幂等可重复执行)
# 支持两种部署方式:
#   A. 在线部署 (推荐, 板子可访问外网): sudo bash board_setup.sh
#   B. 离线部署: 把 cpython tarball / wheels_pkgs/ / librknnrt.so / demo/ 与
#      本脚本放同一目录 (make_bundle.ps1 产物) 后执行
set -e
PKG="$(cd "$(dirname "$0")" && pwd)"
PY=/opt/py312/bin/python3.12

# ---- Python 3.12 (离线包优先, 否则在线装) ----
NEED_PY=1
if [ -x "$PY" ]; then
    if "$PY" -m pip --version >/dev/null 2>&1; then
        echo "[deploy] Python 3.12 已存在, 跳过"
        NEED_PY=0
    else
        echo "[deploy] /opt/py312 存在但缺 pip, 重建..."
        rm -rf /opt/py312
    fi
fi
if [ "$NEED_PY" = 1 ]; then
    if [ -f "$PKG"/cpython-3.12-aarch64-install_only.tar.gz ]; then
        echo "[deploy] 离线安装 Python 3.12 -> /opt/py312 ..."
        mkdir -p /opt/py312
        tar -xzf "$PKG"/cpython-3.12-aarch64-install_only.tar.gz -C /opt/py312 --strip-components=1
    else
        echo "[deploy] 在线安装 Python 3.12 (via uv) -> /opt/py312 ..."
        if ! command -v uv >/dev/null 2>&1; then
            echo "[deploy] 安装 uv -> /usr/local/bin ..."
            curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/usr/local/bin" UV_PYTHON_INSTALL_DIR="/opt/uv/python" sh
        fi
        export UV_PYTHON_INSTALL_DIR="/opt/uv/python"
        # --seed: venv 内装上 pip, 后续 pip 安装依赖才可用
        uv venv --python 3.12 --seed /opt/py312
    fi
fi

# ---- Python 依赖 (离线 wheels 优先, 在线兜底) ----
DEPS="numpy opencv-python-headless pillow psutil pyyaml ruamel.yaml rknn-toolkit-lite2"
if [ -d "$PKG/wheels_pkgs" ]; then
    echo "[deploy] 离线安装 Python 依赖..."
    "$PY" -m pip install --no-index --find-links "$PKG/wheels_pkgs" $DEPS >/dev/null
else
    echo "[deploy] 在线安装 Python 依赖..."
    "$PY" -m pip install $DEPS >/dev/null
fi
echo "[deploy] Python 依赖 OK"

# ---- PySide6 界面 (可选, 安装失败自动回退 Tk) ----
if "$PY" -c "import PySide6" >/dev/null 2>&1; then
    echo "[deploy] PySide6 已存在, 跳过"
else
    echo "[deploy] 安装 PySide6 (现代深色 UI; 失败则回退 Tk)..."
    "$PY" -m pip install --no-cache-dir shiboken6 PySide6-Essentials >/dev/null 2>&1 \
        || echo "[deploy] 警告: PySide6 安装失败, UI 将回退 Tkinter"
fi

# ---- RKNN 运行时 ----
LIBSRC=""
[ -f "$PKG/librknnrt.so" ] && LIBSRC="$PKG/librknnrt.so"
[ -f "$PKG/assets/librknnrt.so" ] && LIBSRC="$PKG/assets/librknnrt.so"
if [ ! -f /usr/lib/librknnrt.so ]; then
    if [ -n "$LIBSRC" ]; then
        echo "[deploy] 安装 librknnrt.so ..."
        cp "$LIBSRC" /usr/lib/ && ldconfig
    else
        echo "[deploy] 警告: 缺少 librknnrt.so, 请从 rknn-toolkit2 (airockchip GitHub) \
runtime/Linux/librknn_api/aarch64/ 手动安装到 /usr/lib/"
    fi
else
    echo "[deploy] librknnrt.so 已存在, 跳过 (如需更新请先删除 /usr/lib/librknnrt.so)"
fi

# ---- 系统工具 ----
# libxcb-cursor0: Qt(xcb) 界面需要; 部分板卡 apt 源没有该包, 缺失时 UI 回退 Tk
for p in alsa-utils gpiod libxcb-cursor0; do
    if ! dpkg -s "$p" >/dev/null 2>&1; then
        echo "[deploy] apt 安装 $p ..."
        apt-get install -y "$p" || echo "[deploy] 警告: $p 安装失败 (无网络或源缺失, 不影响已装功能)"
    fi
done

# apt 源缺 libxcb-cursor0 时, 从 Debian 官方池直接装 deb (Qt 界面必需)
if ! dpkg -s libxcb-cursor0 >/dev/null 2>&1; then
    echo "[deploy] 从 Debian 池安装 libxcb-cursor0 ..."
    BASE=http://deb.debian.org/debian/pool/main/x/xcb-util-cursor
    F=$(curl -s "$BASE/" | grep -o 'libxcb-cursor0_[^"]*_arm64\.deb' | sort -V | tail -1)
    if [ -n "$F" ] && curl -sO "$BASE/$F"; then
        dpkg -i "$F" >/dev/null 2>&1 || echo "[deploy] 警告: libxcb-cursor0 安装失败, UI 回退 Tk"
        rm -f "/tmp/$F"
    else
        echo "[deploy] 警告: libxcb-cursor0 下载失败, UI 回退 Tk"
    fi
fi

# ---- demo 代码 (兼容离线包 demo/ 与 git 仓库 ../ 两种布局) ----
DEMO_SRC=""
[ -d "$PKG/demo" ] && DEMO_SRC="$PKG/demo"
[ -f "$PKG/../main.py" ] && DEMO_SRC="$PKG/.."
if [ -n "$DEMO_SRC" ]; then
    mkdir -p /opt/fall_detect_demo
    cp -r "$DEMO_SRC/." /opt/fall_detect_demo/
    chmod +x /opt/fall_detect_demo/run.sh
    rm -rf /opt/fall_detect_demo/__pycache__
    echo "[deploy] demo 代码已更新到 /opt/fall_detect_demo"
fi

mkdir -p /opt/fall_detect_demo/records
echo "[deploy] 板端安装完成"
