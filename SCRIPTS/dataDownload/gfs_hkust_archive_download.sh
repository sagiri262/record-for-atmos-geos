#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://envf.ust.hk/dataop/data/model_input/gfs_0.25deg_archive"

# GFS的数据格式gfs.0p25.2025040900.f000.grib2
# gfs.0p25 表示 0.25 空间分辨率
# 2025040900 表示 2025年4月9日0时的起报时间
# f000表示起报的第几个文件

# 设定起始和结束时间
START_DATE="20260101"
END_DATE="20260103"

OUT_DIR="${1:-./gfs_archive}"

# Linux/GNU date 写法
# macOS 请安装 coreutils 后把 date 改成 gdate。
next_day() {
  date -u -d "$1 +1 day" +"%Y%m%d"
}

d="$START_DATE"

while [[ "$d" -le "$END_DATE" ]]; do
  yyyy="${d:0:4}"
  yyyymm="${d:0:6}"
  day_url="${BASE_URL}/${yyyy}/${yyyymm}/${d}/"
  day_out="${OUT_DIR}/${yyyy}/${yyyymm}/${d}"

  mkdir -p "$day_out"

  echo "==> Downloading ${day_url}"
  echo "    Save to ${day_out}"

  # -r       递归读取 Apache/Nginx 索引
  # -np      不进入父目录
  # -nd      不保留远程目录层级，因为我们自己按日期归档
  # -A       只接受目标日期的 GFS grib2 文件
  # -c       断点续传
  # --no-clobber 避免重复覆盖已完成文件
  # --no-proxy   超算需设置不走代理
  if ! wget \
    --no-proxy \
    -r -np -nd \
    -c --no-clobber \
    --tries=5 --timeout=30 --wait=1 \
    --user-agent="gfs-archive-downloader/1.0" \
    -A "gfs.0p25.${d}??.f???.grib2" \
    -P "$day_out" \
    "$day_url"; then
      echo "WARN: ${day_url} may not exist or download failed; skipped."
  fi

  d="$(next_day "$d")"
done

echo "==> Building checksum manifest..."
find "$OUT_DIR" -type f -name "gfs.0p25.*.grib2" -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${OUT_DIR}/SHA256SUMS.txt"

echo "Done."
echo "Archive root: ${OUT_DIR}"
echo "Checksum file: ${OUT_DIR}/SHA256SUMS.txt"