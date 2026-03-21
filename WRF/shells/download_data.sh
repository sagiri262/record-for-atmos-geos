#!/bin/bash
set -euo pipefail

############################################
# 路径设置
############################################
DATA_DIR="/public/home/proj_kcchow/zhaoy/WRF_build/DATA"
GFS_ROOT="${DATA_DIR}/GFS"
GEOG_DIR="${DATA_DIR}/WPS_GEOG"
GEOG_TAR="${DATA_DIR}/geog_high_res_mandatory.tar.gz"

mkdir -p "${DATA_DIR}" "${GFS_ROOT}" "${GEOG_DIR}"

############################################
# 禁用代理
############################################
unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY all_proxy
WGET="wget --no-proxy"

############################################
# 下载源
############################################
GEOG_URL="https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz"
BASE_URL="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

############################################
# GFS 时间设置
# 可手动改 YYYYMMDD 和 CYCLE
############################################
YYYYMMDD=$(date -u +%Y%m%d)

HH=$(date -u +%H)
if   [ "$HH" -ge 18 ]; then
    CYCLE="18"
elif [ "$HH" -ge 12 ]; then
    CYCLE="12"
elif [ "$HH" -ge 06 ]; then
    CYCLE="06"
else
    CYCLE="00"
fi

# 也可以手动指定，例如：
# YYYYMMDD="20260319"
# CYCLE="00"

############################################
# GFS 预报时效设置
# 这里默认下载 0~72h，每 3 小时一个文件
############################################
FHR_START=0
FHR_END=72
FHR_STEP=3

GFS_DIR="${GFS_ROOT}/gfs_${YYYYMMDD}_${CYCLE}"
mkdir -p "${GFS_DIR}"

echo "======================================"
echo "开始下载时间 : $(date '+%F %T %Z')"
echo "DATA_DIR     : ${DATA_DIR}"
echo "GEOG_DIR     : ${GEOG_DIR}"
echo "GFS_DIR      : ${GFS_DIR}"
echo "GFS 日期     : ${YYYYMMDD}"
echo "GFS 时次     : ${CYCLE}"
echo "======================================"

############################################
# 1) 下载 WPS 静态地理数据
############################################
if [ ! -f "${GEOG_TAR}" ]; then
    echo "[1/3] 下载 WPS 静态地理数据..."
    ${WGET} -c -O "${GEOG_TAR}" "${GEOG_URL}"
else
    echo "[1/3] 静态地理数据压缩包已存在，跳过下载: ${GEOG_TAR}"
fi

############################################
# 2) 解压 WPS 静态地理数据
############################################
if [ -z "$(ls -A "${GEOG_DIR}" 2>/dev/null)" ]; then
    echo "[2/3] 解压静态地理数据到 ${GEOG_DIR} ..."
    tar -xzf "${GEOG_TAR}" -C "${GEOG_DIR}" --strip-components=1 || \
    tar -xzf "${GEOG_TAR}" -C "${GEOG_DIR}"
else
    echo "[2/3] ${GEOG_DIR} 非空，跳过解压"
fi

############################################
# 3) 下载 GFS 0.25° 数据
############################################
echo "[3/3] 下载 GFS 0.25° 数据..."
cd "${GFS_DIR}"

for ((fhr=${FHR_START}; fhr<=${FHR_END}; fhr+=${FHR_STEP})); do
    fff=$(printf "%03d" "${fhr}")
    fname="gfs.t${CYCLE}z.pgrb2.0p25.f${fff}"
    url="${BASE_URL}/gfs.${YYYYMMDD}/${CYCLE}/atmos/${fname}"

    if [ -f "${fname}" ]; then
        echo "  已存在: ${fname}"
    else
        echo "  下载: ${fname}"
        ${WGET} -c --tries=10 --timeout=60 "${url}"
    fi
done

echo "======================================"
echo "下载完成时间 : $(date '+%F %T %Z')"
echo "静态数据目录 : ${GEOG_DIR}"
echo "GFS 数据目录 : ${GFS_DIR}"
echo "======================================"