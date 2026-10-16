#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 AnimeGANv2 系 ONNX 模型 (在有外网的编译机上运行).

用法: python3 tools/fetch_anime_models.py --outdir models/onnx

来源:
  - 卡通/新海诚: HF vumichien/AnimeGANv2_Hayao / AnimeGANv2_Shinkai
    (NHWC 512x512, [-1,1] 归一化, ~8-9MB)
  - 人像动漫: 优先在 HF 搜索 face_paint/portrait onnx, 找不到则回退
    GitHub AnimeGANv3 JP_face (人像动漫化), 再回退 Paprika (今敏风)
编译机证书校验有问题, 统一用不校验证书的 urllib.
"""
import argparse
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, context=CTX, timeout=timeout)


def download(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print(f"已存在: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
        return True
    print(f"下载 {url}")
    try:
        with get(url) as r, open(out, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        print(f"  失败: {e}")
        if os.path.exists(out):
            os.remove(out)
        return False
    size = os.path.getsize(out) / 1e6
    print(f"  -> {out} ({size:.1f} MB)")
    return size > 1.0


def hf_repo_files(repo_id):
    try:
        data = json.load(get(f"https://huggingface.co/api/models/{repo_id}",
                             timeout=30))
        return [s.get("rfilename", "") for s in data.get("siblings", [])]
    except Exception:
        return []


def find_hf_onnx(keywords, name_hints):
    """在 HF 搜索仓库, 返回第一个含目标 onnx 的 (repo, filename)."""
    for kw in keywords:
        try:
            data = json.load(get(
                "https://huggingface.co/api/models?search=%s&limit=50"
                % urllib.parse.quote(kw), timeout=30))
        except Exception:
            continue
        for m in data:
            rid = m.get("modelId")
            if not rid:
                continue
            for fn in hf_repo_files(rid):
                low = fn.lower()
                if low.endswith(".onnx") and any(h in low for h in name_hints):
                    return rid, fn
    return None, None


def find_github_release_asset(repo, asset_hints):
    """从 GitHub 最新 release 里找 onnx 资产, 返回下载 URL."""
    try:
        data = json.load(get(
            f"https://api.github.com/repos/{repo}/releases", timeout=30))
    except Exception:
        return None
    for rel in data:
        for a in rel.get("assets", []):
            n = a.get("name", "").lower()
            if n.endswith(".onnx") and any(h in n for h in asset_hints):
                return a.get("browser_download_url")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="models/onnx")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    results = {}

    # 1) 固定源: Hayao / Shinkai
    results["anime_hayao"] = download(
        "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/"
        "AnimeGANv2_Hayao.onnx",
        os.path.join(args.outdir, "anime_hayao.onnx"))
    results["anime_shinkai"] = download(
        "https://huggingface.co/vumichien/AnimeGANv2_Shinkai/resolve/main/"
        "AnimeGANv2_Shinkai.onnx",
        os.path.join(args.outdir, "anime_shinkai.onnx"))

    # 2) 人像动漫: face_paint onnx (HF 搜索) -> AnimeGANv3 JP_face (GitHub)
    out = os.path.join(args.outdir, "face_paint.onnx")
    if not (os.path.exists(out) and os.path.getsize(out) > 1_000_000):
        rid, fn = find_hf_onnx(
            ["face_paint animegan", "animegan2 face_paint", "face_paint"],
            ["face_paint", "portrait", "jp_face"])
        if rid:
            print(f"HF 人像模型: {rid}/{fn}")
            results["face_paint"] = download(
                f"https://huggingface.co/{rid}/resolve/main/{fn}", out)
        if not results.get("face_paint"):
            url = find_github_release_asset(
                "TachibanaYoshino/AnimeGANv3",
                ["jp_face", "hayao_36", "shinkai_37"])
            results["face_paint"] = download(url, out) if url else False
    else:
        results["face_paint"] = True

    # 3) 兜底: Paprika (今敏风, HF vumichien)
    out3 = os.path.join(args.outdir, "anime_paprika.onnx")
    if not results.get("face_paint") and not (
            os.path.exists(out3) and os.path.getsize(out3) > 1_000_000):
        results["anime_paprika"] = download(
            "https://huggingface.co/vumichien/AnimeGANv2_Paprika/resolve/main/"
            "AnimeGANv2_Paprika.onnx", out3)

    print("\n汇总:")
    ok = True
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAILED'}")
        ok = ok and v
    if not any(results.values()):
        sys.exit(2)
    print("完成 (个别失败可接受, 转换脚本会跳过缺失的)")


if __name__ == "__main__":
    main()
