#!/bin/sh
# fall_detect_demo 启动包装 (root 终端用)
# 用法: ./run.sh            带界面 (板子接了显示器)
#       ./run.sh --no-ui    无界面
# 原理: 桌面会话是普通用户(q)的 Wayland, root 需借用其 XWayland 授权才能开窗口
export DISPLAY=:0
XAUTH=$(ls /run/user/$(id -u q)/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
[ -n "$XAUTH" ] && export XAUTHORITY="$XAUTH"
cd /opt/fall_detect_demo
exec /opt/py312/bin/python3.12 main.py "$@"
