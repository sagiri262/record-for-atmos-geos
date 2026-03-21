#!/usr/bin/bash

set -euo pipefail

# 固定路径
WPS_DIR = "/public/home/proj_kcchow/zhaoy/WRF_build/WRFV3"
WRF_DIR = "/public/home/proj_kcchow/zhaoy/WRF_build/WPS"
DATA_DIR = "/public/home/proj_kcchow/zhaoy/WRF_build/DATA"

VATABLE_SRC = "${WPS_DIR}/ungrib/Variable_Tables/Vtable.GFS"

RUN_REAL_CMD = "./real.exe"
RUN_WRF_CMD  = "./wrf.exe"


# 下载 GFS 静态地理数据
BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

UTC_TIME = $(date -u +%Y%m%d%H)
YYYYMMDD = $(date -u +%Y%m%d)


HH = $(date -u +%H)
if   ["$HH" -ge 18]; then CYCLE = "18"
elif ["$HH" -ge 12]; then CYCLE = "12"
elif ["$HH" -ge 06]; then CYCLE = "06"
else CYCLE="00"
fi


# 预报时效
FHR_START = 3
FHR_END   = 336
STEP      =3


GFS_DIR="${DATA_DIR}/GFS_${YYYYMMDD}${CYCLE}"
GEOG_DIR="${DATA_DIR}/WPS_GEOG"
STATIC_TAR="${DATA_DIR}/geog_high_res_mandatory.tar.gz"

mkdir -p "${DATA_DIR}" "${GFS_DIR}" "${GEOG_DIR}"

echo "======================================"
echo "开始时间: $(date '+%F %T %Z')"
echo "GFS日期:  ${YYYYMMDD}"
echo "GFS时次:  ${CYCLE}"
echo "数据目录: ${DATA_DIR}"
echo "======================================"

# 下载静态地理数据
# UCAR 官方最高分辨率 mandatory 包
GEOG_URL="https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz"

if [ ! -f "${STATIC_TAR}" ]; then
    echo "[1/7] 下载 WPS 静态地理数据..."
    ${WGET} -c -O "${STATIC_TAR}" "${GEOG_URL}"
else
    echo "[1/7] 已存在静态包，跳过下载: ${STATIC_TAR}"
fi

if [ ! -f "${GEOG_DIR}/index" ] && [ -z "$(ls -A "${GEOG_DIR}" 2>/dev/null)" ]; then
    echo "[1/7] 解压静态地理数据到 ${GEOG_DIR} ..."
    tar -xzf "${STATIC_TAR}" -C "${GEOG_DIR}" --strip-components=1 || \
    tar -xzf "${STATIC_TAR}" -C "${GEOG_DIR}"
else
    echo "[1/7] 静态地理数据目录非空，跳过解压"
fi

# 下载 GFS 数据
echo "[2/7] 下载 GFS 0.25° 数据..."
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


