#!/bin/bash
# 克隆 CIPGAN 水墨 CycleGAN 仓库并查看结构 (编译机)
set -e
cd /tmp
if [ ! -d CIPGAN ]; then
    git clone --depth 1 https://github.com/Wenretium/CIPGAN.git 2>&1 | tail -2
fi
echo "---- 目录 ----"
ls /tmp/CIPGAN
echo "---- checkpoints ----"
find /tmp/CIPGAN -name "*.pth" -exec ls -l {} \;
echo "---- networks.py 生成器定义 ----"
grep -n "class ResnetGenerator" -A 60 /tmp/CIPGAN/models/networks.py | head -100
