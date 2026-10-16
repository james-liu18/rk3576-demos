#!/bin/bash
export XAUTH=$(ls /run/user/$(id -u q)/.mutter-Xwaylandauth.* | head -1)
export DISPLAY=:0
export XAUTHORITY="$XAUTH"
for W in $(xwininfo -root -tree | awk '/tk/ {print $1}'); do
  echo "try $W"
  xwd -id "$W" -silent -out /tmp/ui.xwd 2>/dev/null && \
    ffmpeg -y -loglevel error -i /tmp/ui.xwd /tmp/ui_win.png 2>/dev/null && break
done
ls -la /tmp/ui_win.png 2>/dev/null
