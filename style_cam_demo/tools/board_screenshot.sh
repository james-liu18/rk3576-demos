#!/bin/bash
# 板端: 唤醒屏幕并截图
export XAUTH=$(ls /run/user/$(id -u q)/.mutter-Xwaylandauth.* | head -1)
export DISPLAY=:0
export XAUTHORITY="$XAUTH"
xset dpms force on 2>/dev/null
xset q 2>/dev/null | grep -A2 'DPMS is' | head -3
sleep 1
ffmpeg -y -loglevel error -f x11grab -video_size 1280x720 -i :0 -frames:v 1 /tmp/scr2.png
echo done
