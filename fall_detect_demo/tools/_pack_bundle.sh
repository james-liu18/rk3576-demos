#!/bin/bash
# 在编译机组装 fall_detect_bundle 并打包
set -e
B=~/fall_detect_bundle
rm -rf "$B"
mkdir -p "$B/demo"
cp ~/board_pkg/cpython-3.12-aarch64-install_only.tar.gz "$B/"
cp -r ~/board_pkg/wheels_pkgs "$B/"
cp ~/board_pkg/librknnrt.so "$B/"
cp /tmp/bundle_src/board_setup.sh "$B/"
cp /tmp/bundle_src/demo/*.py /tmp/bundle_src/demo/README.md \
   /tmp/bundle_src/demo/config.yaml /tmp/bundle_src/demo/run.sh "$B/demo/"
cp -r /tmp/bundle_src/demo/models /tmp/bundle_src/demo/assets \
      /tmp/bundle_src/demo/tools "$B/demo/"
chmod +x "$B/demo/run.sh"
tar -czf ~/fall_detect_bundle.tar.gz -C ~ fall_detect_bundle
ls -lh ~/fall_detect_bundle.tar.gz
tar -tzf ~/fall_detect_bundle.tar.gz | head -8
