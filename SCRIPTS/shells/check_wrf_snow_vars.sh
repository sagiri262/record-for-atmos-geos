#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法: $0 wrfout_file [wrfout_file2 ...]"
  exit 1
fi

vars=(SNOWC SNOWH SWE SNOW_DEPTH SNOW ALBEDO SNOWNC RAINNC RAINC RAINSH SR)

for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "文件不存在: $f"
    continue
  fi

  echo "========================================"
  echo "文件: $f"
  echo "========================================"

  hdr=$(ncdump -h "$f")

  for var in "${vars[@]}"; do
    if printf '%s\n' "$hdr" | grep -Eq "^[[:space:]]*(byte|char|short|int|float|double)[[:space:]]+${var}\s*\("; then
      echo "${var}: found"
    else
      echo "${var}: no such variable"
    fi
  done

  echo
  echo "--- 推荐用于雨雪分离的变量检查 ---"
  for var in SNOWNC RAINNC RAINC RAINSH SR; do
    if printf '%s\n' "$hdr" | grep -Eq "^[[:space:]]*(byte|char|short|int|float|double)[[:space:]]+${var}\s*\("; then
      echo "${var}: found"
    else
      echo "${var}: no such variable"
    fi
  done
  echo

done
