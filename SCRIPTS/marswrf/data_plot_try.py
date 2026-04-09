#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MarsWRF 柱沙尘光学厚度 (CDOD) 纬度-太阳黄经图
=====================================================
横轴：火星太阳黄经 Ls (0–360°)
纵轴：纬度 (-90°S ~ 90°N)
色填：CDOD 日均值 / 时次均值
白线：CO2ICE 极冠轮廓线

换算关系：
  CDOD = TAU_OD2D * 610 / 700 / 2 / 1.3

变量说明（来自 MarsWRF 输出）：
  TAU_OD2D  : 2D 柱积分沙尘光学厚度（时间, 南北, 东西）
  CO2ICE    : 地表 CO2 冰含量，用于标注极冠边界（时间, 南北, 东西）
  L_S       : 火星太阳黄经（标量/时间维）
  XLAT      : 纬度（南北, 东西）或（时间, 南北, 东西）

使用方法：
  python marswrf_cdod_ls_lat.py --file /path/to/wrfout_d01_*
  python marswrf_cdod_ls_lat.py --file /path/to/dir_containing_wrfout_files
  python marswrf_cdod_ls_lat.py --file /path/to/single_wrfout_file

依赖：
  pip install numpy matplotlib netCDF4
"""

import argparse
import glob
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from netCDF4 import Dataset

# =========================================================
# 0. 导入上级目录中的 wrf_read_data.py
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

try:
    from wrf_read_data import WRFDataReader
    HAS_WRF_READER = True
except Exception:
    HAS_WRF_READER = False


# =========================================================
# 1. 常量：TAU_OD2D -> CDOD 换算系数
# =========================================================
CDOD_FACTOR = 610.0 / 700.0 / 2.0 / 1.3


# =========================================================
# 2. 工具函数
# =========================================================
def collect_wrf_files(path_pattern):
    """
    支持三种输入：
      1) 目录：自动查找目录下 wrfout_d01_*
      2) 通配符路径：直接 glob
      3) 单文件路径：直接返回该文件
    """
    if os.path.isdir(path_pattern):
        files = sorted(
            [
                os.path.join(path_pattern, name)
                for name in os.listdir(path_pattern)
                if name.startswith("wrfout_d01_")
                and os.path.isfile(os.path.join(path_pattern, name))
            ]
        )
        return files

    # 先按 glob 试
    files = sorted(glob.glob(path_pattern))
    if files:
        return files

    # 如果是单文件且存在
    if os.path.isfile(path_pattern):
        return [path_pattern]

    # 再尝试 WRFDataReader（兼容你的原工程）
    if HAS_WRF_READER:
        reader = WRFDataReader(path_pattern)
        files = reader.get_files()
        return sorted(files)

    return []


def get_lat_1d(ds):
    """
    从 XLAT 中提取一维纬度坐标。
    兼容：
      XLAT(time, south_north, west_east)
      XLAT(south_north, west_east)
    """
    if "XLAT" not in ds.variables:
        raise KeyError("文件中不存在变量 XLAT")

    xlat = ds.variables["XLAT"][:]

    if xlat.ndim == 3:
        lat2d = xlat[0, :, :]
    elif xlat.ndim == 2:
        lat2d = xlat[:, :]
    else:
        raise ValueError(f"XLAT 维度异常: {xlat.shape}")

    lat_1d = np.mean(lat2d, axis=1)  # 对经向求平均，得到每条纬线的代表纬度
    return np.asarray(lat_1d, dtype=float)


def get_time_length(ds, varname):
    """
    获取变量的时间长度，要求变量形状至少为 (time, y, x)
    """
    if varname not in ds.variables:
        raise KeyError(f"文件中不存在变量 {varname}")
    shape = ds.variables[varname].shape
    if len(shape) < 3:
        raise ValueError(f"{varname} 维度异常，期望至少 3 维，实际为 {shape}")
    return shape[0]


def get_ls_value(ds, t):
    """
    读取第 t 个时次的 L_S，并规整到 0~360
    """
    if "L_S" not in ds.variables:
        raise KeyError("文件中不存在变量 L_S")

    ls_var = ds.variables["L_S"]

    if ls_var.ndim == 0:
        ls_val = float(ls_var[...])
    elif ls_var.ndim == 1:
        ls_val = float(ls_var[t])
    else:
        # 极少数情况若维度更复杂，取第一个可索引值
        ls_val = float(np.ravel(ls_var[t])[0])

    return ls_val % 360.0


def sort_lat_profile(lat_1d, profile_1d):
    """
    为了插值稳定，保证纬度坐标是单调递增的
    """
    idx = np.argsort(lat_1d)
    return lat_1d[idx], profile_1d[idx]


# =========================================================
# 3. 命令行参数
# =========================================================
parser = argparse.ArgumentParser(description="MarsWRF CDOD Ls-Lat diagram")
parser.add_argument(
    "--file", "-f",
    default="wrfout_d01_*",
    help="NetCDF 文件路径（支持目录 / 单文件 / 通配符），默认 wrfout_d01_*"
)
parser.add_argument(
    "--ls_bins",
    type=int,
    default=72,
    help="Ls 分箱数（默认72，即每5°一箱）"
)
parser.add_argument(
    "--lat_bins",
    type=int,
    default=36,
    help="纬度分箱数（默认36，即每5°一箱）"
)
parser.add_argument(
    "--co2_threshold",
    type=float,
    default=1.0,
    help="CO2ICE 极冠轮廓阈值（kg/m²），默认1.0"
)
parser.add_argument(
    "--vmin",
    type=float,
    default=0.0,
    help="色标最小值"
)
parser.add_argument(
    "--vmax",
    type=float,
    default=1.2,
    help="色标最大值（CDOD 建议可比 TAU_OD2D 小一些，默认 1.2）"
)
parser.add_argument(
    "--cmap",
    default="jet",
    help="色图，默认 jet"
)
parser.add_argument(
    "--out",
    default="cdod_ls_lat.png",
    help="输出图片路径"
)
args = parser.parse_args()


# =========================================================
# 4. 收集文件
# =========================================================
files = collect_wrf_files(args.file)

if not files:
    raise FileNotFoundError(f"没有找到 wrfout_d01_* 文件：{args.file}")

print(f"[INFO] 共找到 {len(files)} 个文件")
print(f"[INFO] 第一个文件: {files[0]}")
print(f"[INFO] 最后一个文件: {files[-1]}")
print(f"[INFO] CDOD 换算系数: {CDOD_FACTOR:.10f}")


# =========================================================
# 5. 读取并汇总数据
# =========================================================
all_ls = []         # 每个时次的 Ls
all_cdod_lat = []   # 每个时次对应的纬向平均 CDOD
all_co2_lat = []    # 每个时次对应的纬向平均 CO2ICE
lat_1d = None       # 原始纬度坐标

for fpath in files:
    print(f"[INFO] 处理文件: {os.path.basename(fpath)}")

    with Dataset(fpath, "r") as ds:
        # 只在第一个文件中读取纬度坐标
        if lat_1d is None:
            lat_1d = get_lat_1d(ds)

        ntimes = get_time_length(ds, "TAU_OD2D")

        for t in range(ntimes):
            # 读取 Ls
            ls_val = get_ls_value(ds, t)

            # 读取 TAU_OD2D 并换算为 CDOD
            tau2d = np.asarray(ds.variables["TAU_OD2D"][t, :, :], dtype=float)
            cdod2d = tau2d * CDOD_FACTOR
            cdod_zonal = np.nanmean(cdod2d, axis=1)

            # 读取 CO2ICE
            co2_2d = np.asarray(ds.variables["CO2ICE"][t, :, :], dtype=float)
            co2_zonal = np.nanmean(co2_2d, axis=1)

            all_ls.append(ls_val)
            all_cdod_lat.append(cdod_zonal)
            all_co2_lat.append(co2_zonal)

all_ls = np.array(all_ls, dtype=float)               # (N,)
all_cdod_lat = np.array(all_cdod_lat, dtype=float)   # (N, SN)
all_co2_lat = np.array(all_co2_lat, dtype=float)     # (N, SN)

if all_cdod_lat.ndim != 2:
    raise ValueError(f"all_cdod_lat 维度异常: {all_cdod_lat.shape}")

n_sn = all_cdod_lat.shape[1]

print(f"[INFO] 共读取 {len(all_ls)} 个时次，纬度格点数 = {n_sn}")
print(f"[INFO] Ls 范围：{np.nanmin(all_ls):.2f} ~ {np.nanmax(all_ls):.2f}")


# =========================================================
# 6. 按 Ls 分箱求均值 -> 二维网格 (lat_bins, ls_bins)
# =========================================================
ls_edges = np.linspace(0.0, 360.0, args.ls_bins + 1)
lat_edges = np.linspace(-90.0, 90.0, args.lat_bins + 1)

ls_centers = 0.5 * (ls_edges[:-1] + ls_edges[1:])
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

cdod_grid = np.full((args.lat_bins, args.ls_bins), np.nan, dtype=float)
co2_grid = np.full((args.lat_bins, args.ls_bins), np.nan, dtype=float)

for i_ls in range(args.ls_bins):
    # 最后一箱包含右端点 360，避免极少数边界值漏掉
    if i_ls < args.ls_bins - 1:
        mask = (all_ls >= ls_edges[i_ls]) & (all_ls < ls_edges[i_ls + 1])
    else:
        mask = (all_ls >= ls_edges[i_ls]) & (all_ls <= ls_edges[i_ls + 1])

    if not np.any(mask):
        continue

    cdod_mean_sn = np.nanmean(all_cdod_lat[mask, :], axis=0)  # (SN,)
    co2_mean_sn = np.nanmean(all_co2_lat[mask, :], axis=0)    # (SN,)

    lat_sorted, cdod_sorted = sort_lat_profile(lat_1d, cdod_mean_sn)
    _, co2_sorted = sort_lat_profile(lat_1d, co2_mean_sn)

    cdod_grid[:, i_ls] = np.interp(lat_centers, lat_sorted, cdod_sorted)
    co2_grid[:, i_ls] = np.interp(lat_centers, lat_sorted, co2_sorted)


# =========================================================
# 7. 绘图
# =========================================================
fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

cmap = plt.get_cmap(args.cmap)
norm = mcolors.Normalize(vmin=args.vmin, vmax=0.35)

im = ax.pcolormesh(
    ls_centers,
    lat_centers,
    cdod_grid,
    cmap=cmap,
    norm=norm,
    shading="auto"
)

# CO2 极冠轮廓
ax.contour(
    ls_centers,
    lat_centers,
    co2_grid,
    levels=[args.co2_threshold],
    colors="white",
    linewidths=1.2,
    linestyles="-"
)

# 坐标轴
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)
ax.set_xticks(np.arange(0, 361, 45))
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_xlabel("Solar Longitude  $L_s$ (°)", fontsize=11)
ax.set_ylabel("Latitude (°)", fontsize=11)

ax.xaxis.set_minor_locator(mticker.MultipleLocator(15))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.tick_params(which="both", direction="in", top=True, right=True)

# 维持你原来的纬度标签风格
ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"])

# 色标
cbar = fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    pad=0.18,
    fraction=0.04,
    aspect=40
)
cbar.set_label("Column Dust Optical Depth (CDOD)", fontsize=9)

if 0.35 > args.vmin:
    cbar.set_ticks(np.linspace(args.vmin, 0.35, 7))

# 标题
ax.set_title(
    "MarsWRF Mean Column Dust Optical Depth (CDOD)\n"
    f"(CO₂ ice cap contour = {args.co2_threshold} kg m⁻²)",
    fontsize=11,
    pad=8
)

plt.tight_layout()
plt.savefig(args.out, bbox_inches="tight")
print(f"[INFO] 图片已保存：{args.out}")
plt.show()