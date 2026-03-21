# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, binary_erosion

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from wrf import (
    getvar, interplevel, latlon_coords, to_np,
    get_cartopy, cartopy_xlim, cartopy_ylim
)

# =========================================================
# 0. 导入上级目录中的 wrf_read_data.py
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from wrf_read_data import WRFDataReader


# =========================================================
# 1. 参数设置
# =========================================================
wrf_path = "/Volumes/Lexar/WRF_Data/WRF_second_try/wrfout_d01_*"
reader = WRFDataReader(wrf_path)

# 获取排序后的文件列表
wrf_files = reader.get_files()

# 选择要绘制的文件
file_idx = -1      # -1 表示最后一个文件
ncfile = Dataset(wrf_files[file_idx])

# 如果你想画“时段降水”，建议用当前文件减去前一个文件
use_period_precip = True   # True: 时段降水；False: 累积降水

# 500 hPa 标准气压层
target_lev = 500  # hPa

# 稀疏风矢量步长
skip = 8

# 风切变线识别参数（可调）
smooth_sigma = 1.2          # 平滑程度
shear_percentile = 88       # 形变强度阈值百分位
conv_percentile = 35        # 辐合阈值百分位（越小越偏向收敛区）


# =========================================================
# 2. 读取三维变量并插值到 500 hPa
# =========================================================
pressure = getvar(ncfile, "pressure")
z = getvar(ncfile, "z", units="dm")              # 位势高度，单位 dam（更适合天气图）
ua = getvar(ncfile, "ua", units="m s-1")
va = getvar(ncfile, "va", units="m s-1")
tc = getvar(ncfile, "tc")                        # 摄氏度

z500 = interplevel(z, pressure, target_lev)
u500 = interplevel(ua, pressure, target_lev)
v500 = interplevel(va, pressure, target_lev)
t500 = interplevel(tc, pressure, target_lev)

lats, lons = latlon_coords(z500)
lats_np = to_np(lats)
lons_np = to_np(lons)

proj = get_cartopy(z500)


# =========================================================
# 3. 读取降水
# =========================================================
# WRF 常用累计降水 = RAINC + RAINNC
rainc_now = getvar(ncfile, "RAINC")
rainnc_now = getvar(ncfile, "RAINNC")
rain_now = rainc_now + rainnc_now

if use_period_precip and len(wrf_files) >= 2:
    # 当前时次相对前一个 wrfout 的时段降水
    prev_idx = file_idx - 1 if file_idx != 0 else 0
    ncfile_prev = Dataset(wrf_files[prev_idx])

    rainc_prev = getvar(ncfile_prev, "RAINC")
    rainnc_prev = getvar(ncfile_prev, "RAINNC")
    rain_prev = rainc_prev + rainnc_prev

    rain_plot = to_np(rain_now - rain_prev)
    rain_plot = np.where(rain_plot < 0, 0, rain_plot)   # 防止少数重启导致负值
else:
    # 直接画累计降水
    rain_plot = to_np(rain_now)


# =========================================================
# 4. 计算风切变线（近似诊断）
# =========================================================
# 说明：
# “风切变线”自动识别并没有唯一标准。
# 这里采用：
# 1) 500hPa 风场形变（deformation）较大
# 2) 同时位于相对辐合区
# 3) 对满足条件区域的边界作线
# 这是比较实用的一种近似画法。

u = gaussian_filter(to_np(u500), smooth_sigma)
v = gaussian_filter(to_np(v500), smooth_sigma)

# 简化处理：在规则网格上用 np.gradient 求梯度
# 对天气图展示足够实用
dudy, dudx = np.gradient(u)
dvdy, dvdx = np.gradient(v)

# 水平辐散（负值代表辐合）
div = dudx + dvdy

# 形变强度（总形变）
deform = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)

# 阈值
shear_thr = np.nanpercentile(deform, shear_percentile)
conv_thr = np.nanpercentile(div, conv_percentile)

# 剪切线候选区：形变强 + 收敛
shear_mask = (deform >= shear_thr) & (div <= conv_thr)

# 去掉零碎像元，使线更平滑
shear_mask = gaussian_filter(shear_mask.astype(float), 1.0) > 0.35

# 取边界
shear_edge = shear_mask ^ binary_erosion(shear_mask)


# =========================================================
# 5. 降水配色：<5 mm 透明，其余红到蓝（红小蓝大）
# =========================================================
# 第一个区间 [0, 5) 透明
rain_levels = [0, 5, 10, 25, 50, 100, 250]

# 颜色顺序：5-10 红，随后逐步过渡到蓝，高值蓝色
rain_colors = [
    (1.0, 1.0, 1.0, 0.0),   # 0-5 mm：透明
    "#d73027",              # 5-10
    "#fc8d59",              # 10-25
    "#fee090",              # 25-50
    "#91bfdb",              # 50-100
    "#4575b4"               # 100-250
]
rain_cmap = mcolors.ListedColormap(rain_colors)
rain_norm = mcolors.BoundaryNorm(rain_levels, rain_cmap.N)


# =========================================================
# 6. 开始绘图
# =========================================================
fig = plt.figure(figsize=(13, 10))
ax = plt.axes(projection=proj)

# 地理要素
ax.coastlines(resolution="50m", linewidth=0.8)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
ax.add_feature(cfeature.LAKES.with_scale("50m"), alpha=0.4)
ax.add_feature(cfeature.RIVERS.with_scale("50m"), alpha=0.4)

# 设置范围
ax.set_xlim(cartopy_xlim(z500))
ax.set_ylim(cartopy_ylim(z500))

# ---------------------------------------------------------
# 6.1 降水填色
# ---------------------------------------------------------
cf = ax.contourf(
    lons_np, lats_np, rain_plot,
    levels=rain_levels,
    cmap=rain_cmap,
    norm=rain_norm,
    extend="max",
    transform=ccrs.PlateCarree(),
    zorder=1
)

cbar = plt.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
cbar.set_label("Precipitation (mm)")

# ---------------------------------------------------------
# 6.2 500 hPa 位势高度线（黑色细线）
# ---------------------------------------------------------
# 常见天气图习惯可用每 4 dagpm 或 8 dagpm
z500_np = to_np(z500)
zmin = int(np.nanmin(z500_np) // 4 * 4)
zmax = int(np.nanmax(z500_np) // 4 * 4 + 4)
hgt_levels = np.arange(zmin, zmax + 1, 4)

cs_hgt = ax.contour(
    lons_np, lats_np, z500_np,
    levels=hgt_levels,
    colors="black",
    linewidths=0.8,   # <<< 等压线/位势高度线粗细在这里调
    transform=ccrs.PlateCarree(),
    zorder=3
)
ax.clabel(cs_hgt, fmt="%d", fontsize=8)

# ---------------------------------------------------------
# 6.3 500 hPa 等温线（绿色）
# ---------------------------------------------------------
t500_np = to_np(t500)
t_levels = np.arange(-60, 21, 4)

cs_tmp = ax.contour(
    lons_np, lats_np, t500_np,
    levels=t_levels,
    colors="green",
    linewidths=1.0,   # <<< 等温线粗细在这里调
    linestyles="-",
    transform=ccrs.PlateCarree(),
    zorder=4
)
ax.clabel(cs_tmp, fmt="%d", fontsize=8, colors="green")

# ---------------------------------------------------------
# 6.4 风切变线（深红色）
# ---------------------------------------------------------
# 用 shear_edge 或 shear_mask 的 0.5 等值线来表示
ax.contour(
    lons_np, lats_np, shear_edge.astype(float),
    levels=[0.5],
    colors=["darkred"],
    linewidths=2.0,   # <<< 风切变线粗细在这里调
    transform=ccrs.PlateCarree(),
    zorder=5
)

# ---------------------------------------------------------
# 6.5 风向箭头（黑色）
# ---------------------------------------------------------
u500_np = to_np(u500)
v500_np = to_np(v500)

qv = ax.quiver(
    lons_np[::skip, ::skip],
    lats_np[::skip, ::skip],
    u500_np[::skip, ::skip],
    v500_np[::skip, ::skip],
    color="black",
    pivot="middle",
    scale=500,          # <<< 箭头长度比例可在这里调
    width=0.0022,       # <<< 箭杆粗细在这里调
    headwidth=3.5,      # <<< 箭头头部宽度
    headlength=4.5,     # <<< 箭头头部长度
    transform=ccrs.PlateCarree(),
    zorder=6
)

# ---------------------------------------------------------
# 6.6 标题
# ---------------------------------------------------------
time_str = str(to_np(getvar(ncfile, "times")))
title_text = (
    f"500 hPa Synoptic Analysis\n"
    f"Time: {time_str}"
)
plt.title(title_text, fontsize=14)

plt.tight_layout()
plt.show()