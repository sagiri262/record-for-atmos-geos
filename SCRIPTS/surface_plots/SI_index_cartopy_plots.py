# -*- coding: utf-8 -*-
"""
Matplotlib + Cartopy 两联图
------------------------------------------------------------
左图：WRF 原始范围上的海平面气压场（SLP）
右图：杭州附近 [118E, 122E, 28.5N, 31.5N] 的 Showalter Index (SI)

本版按你的要求做了这些修改：
1. 左右图高度保持一致
2. 左图宽度比右图大两个单位（width_ratios = [8, 6]）
3. 左图纬度标注间隔改成 5 度，经度标注间隔保持不变
4. 左图内部格网间隔保持不变
5. 右图经纬网设置保持不变，只把经纬度标注字体调小
6. 图号 (a)(b) 放在各子图左上角，10 pt，加粗
7. 右图直接用裁剪后的 WRF 原始二维经纬网绘图，不再做 griddata

注意：
- 这版 SI 先用占位露点近似：
      dewpoint = temperature - 2 K
  只是为了把图稳定跑出来。
- 如果你后面提供真实湿度变量（QVAPOR / RH / td），再把这段换成正式业务版。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy.calc as mpcalc

from netCDF4 import Dataset
from metpy.units import units

from wrf import getvar, latlon_coords, to_np
from wrf_read_data import WRFDataReader

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# 0. 可调参数区
# ============================================================

# 数据路径
wrf_path = "/Volumes/Lexar/WRF_Data/WRF_second_try/wrfout_d01_*"

# 右图区域 [west, east, south, north]，单位：度
region_right = [118.0, 122.0, 28.5, 31.5]

# 杭州中心点，单位：度
hangzhou_lon = 120.0
hangzhou_lat = 30.0

# 整图大小
# figsize 单位：英寸；1 inch = 2.54 cm
fig_w_cm = 16.0
fig_h_cm = 8.0
fig_w_in = fig_w_cm / 2.54
fig_h_in = fig_h_cm / 2.54

# 左右子图宽度比例：左图比右图大两个单位
LEFT_WIDTH_RATIO = 8
RIGHT_WIDTH_RATIO = 6

# 经纬度标注字体大小
TICK_LABEL_SIZE = 6

# 左图经纬度“标注”间隔，单位：度
LEFT_LON_LABEL = 2.0   # 经度标注保持不变
LEFT_LAT_LABEL = 5.0   # 纬度标注改成 5 度

# 左图内部格网间隔，单位：度（保持不变）
LEFT_LON_GRID = 2.0
LEFT_LAT_GRID = 2.0

# 右图经纬网间隔，单位：度（保持不变）
RIGHT_LON_MAJOR = 1.0
RIGHT_LAT_MAJOR = 1.0

# 左图气压场等值线设置，单位：hPa
SLP_CONTOUR_INTERVAL = 2.0
SLP_LABEL_INTERVAL = 4.0

# 右图 SI 等值线设置，单位：°C（温差）
SI_CONTOUR_INTERVAL = 1.0
SI_LABEL_INTERVAL = 2.0

# 输出文件
out_png = "wrf_pressure_si_cartopy.png"
out_pdf = "wrf_pressure_si_cartopy.pdf"


# ============================================================
# 1. 读 WRF 文件
# ============================================================
reader = WRFDataReader(wrf_path)
wrf_files = reader.get_files()

if len(wrf_files) == 0:
    raise FileNotFoundError(f"没有找到 WRF 文件：{wrf_path}")

ncfile = Dataset(wrf_files[0])


# ============================================================
# 2. 左图变量：SLP + 经纬度
# ============================================================
slp = getvar(ncfile, "slp", timeidx=0)  # 单位通常为 hPa
lat2d, lon2d = latlon_coords(slp)

slp_np = np.ma.filled(to_np(slp), np.nan)
lat2d_np = np.ma.filled(to_np(lat2d), np.nan)
lon2d_np = np.ma.filled(to_np(lon2d), np.nan)

# 左图原始地理四至范围
west = float(np.nanmin(lon2d_np))
east = float(np.nanmax(lon2d_np))
south = float(np.nanmin(lat2d_np))
north = float(np.nanmax(lat2d_np))
region_left = [west, east, south, north]


# ============================================================
# 3. 裁剪右图区域
# ============================================================
mask_right = (
    (lon2d_np >= region_right[0]) & (lon2d_np <= region_right[1]) &
    (lat2d_np >= region_right[2]) & (lat2d_np <= region_right[3])
)

if not np.any(mask_right):
    raise ValueError("右图区域在当前 WRF 域内没有任何格点，请检查 region_right 或数据范围")

jj, ii = np.where(mask_right)
j0, j1 = jj.min(), jj.max()
i0, i1 = ii.min(), ii.max()

lon_r = lon2d_np[j0:j1 + 1, i0:i1 + 1]
lat_r = lat2d_np[j0:j1 + 1, i0:i1 + 1]


# ============================================================
# 4. 右图 SI 所需三维变量
# ============================================================
pres3d_hpa = getvar(ncfile, "pressure", timeidx=0)   # hPa
temp3d_c = getvar(ncfile, "tc", timeidx=0)           # degC

pres3d_hpa_np = np.ma.filled(to_np(pres3d_hpa), np.nan)[:, j0:j1 + 1, i0:i1 + 1]
temp3d_c_np = np.ma.filled(to_np(temp3d_c), np.nan)[:, j0:j1 + 1, i0:i1 + 1]

# 挂单位
p = pres3d_hpa_np * units.hPa
t = temp3d_c_np * units.degC


# ============================================================
# 5. 计算右图 SI
# ------------------------------------------------------------
# 使用 MetPy 自带的 showalter_index
# dewpoint 先用占位近似：比气温低 2 K
# ============================================================
nz, ny, nx = p.shape
si = np.full((ny, nx), np.nan, dtype=float)

ok_count = 0
skip_count = 0
err_count = 0

for j in range(ny):
    for i in range(nx):
        p_prof = p[:, j, i]
        t_prof = t[:, j, i]

        p_vals = np.asarray(p_prof.magnitude)
        t_vals = np.asarray(t_prof.magnitude)

        # 缺测过滤
        good = np.isfinite(p_vals) & np.isfinite(t_vals)
        if good.sum() < 3:
            skip_count += 1
            continue

        p1 = p_prof[good]
        t1 = t_prof[good]

        # pressure 从大到小排序：近地面 -> 高空
        order = np.argsort(p1.magnitude)[::-1]
        p1 = p1[order]
        t1 = t1[order]

        # Showalter Index 至少需要覆盖 850 hPa 和 500 hPa
        if not (np.nanmax(p1.magnitude) >= 850 and np.nanmin(p1.magnitude) <= 500):
            skip_count += 1
            continue

        try:
            # 转 Kelvin，避免摄氏温差单位坑
            t1k = t1.to("kelvin")

            # 占位近似露点：比气温低 2 K
            td1k = t1k - 2.0 * units.kelvin

            si_val = mpcalc.showalter_index(p1, t1k, td1k)

            # 转成温差单位并取标量
            si[j, i] = np.asarray(si_val.to("delta_degC").magnitude).item()
            ok_count += 1

        except Exception as e:
            err_count += 1
            if err_count <= 5:
                print(f"[SI error] j={j}, i={i}, err={e}")
            continue

print("SI valid count =", np.isfinite(si).sum())
print("SI ok_count    =", ok_count)
print("SI skip_count  =", skip_count)
print("SI err_count   =", err_count)

if not np.isfinite(si).any():
    raise ValueError("右图 SI 全是 NaN，请检查气压/温度廓线或 dewpoint 近似写法")


# ============================================================
# 6. 开始绘图：Matplotlib + Cartopy
# ============================================================
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=300)

gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[LEFT_WIDTH_RATIO, RIGHT_WIDTH_RATIO],  # 左图比右图大两个单位
    wspace=0.12,  # 子图间距，越大间距越大
)

ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax2 = fig.add_subplot(gs[0, 1], projection=proj)


# ============================================================
# 7. 左图：原始 WRF 范围气压场
# ============================================================
ax1.set_extent(region_left, crs=proj)

# 底图填色
pcm1 = ax1.pcolormesh(
    lon2d_np,
    lat2d_np,
    slp_np,
    cmap="coolwarm",
    shading="auto",
    transform=proj,
)

# 等值线
pmin = np.nanmin(slp_np)
pmax = np.nanmax(slp_np)

levels_left = np.arange(
    np.floor(pmin / SLP_CONTOUR_INTERVAL) * SLP_CONTOUR_INTERVAL,
    np.ceil(pmax / SLP_CONTOUR_INTERVAL) * SLP_CONTOUR_INTERVAL + SLP_CONTOUR_INTERVAL,
    SLP_CONTOUR_INTERVAL,
)

cs1 = ax1.contour(
    lon2d_np,
    lat2d_np,
    slp_np,
    levels=levels_left,
    colors="black",
    linewidths=0.45,
    transform=proj,
)

# 只在实际画出的等值线里筛选出需要标注的层级，避免 clabel 报错
levels_left_label = [
    lev for lev in cs1.levels
    if np.isclose((lev / SLP_LABEL_INTERVAL) - round(lev / SLP_LABEL_INTERVAL), 0)
]

if len(levels_left_label) > 0:
    ax1.clabel(
        cs1,
        levels=levels_left_label,
        inline=True,
        fmt="%d",
        fontsize=6,
    )

# 海岸线 / 国界 / 陆海底图
ax1.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="none", zorder=0)
ax1.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
ax1.coastlines(resolution="10m", linewidth=0.5)
ax1.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)

# 左图内部格网：只画格网，不画文字
gl1 = ax1.gridlines(
    crs=proj,
    draw_labels=False,
    linewidth=0.4,
    color="gray",
    alpha=0.7,
    linestyle="-",
)

# 内部格网间隔保持不变
gl1.xlocator = mticker.MultipleLocator(LEFT_LON_GRID)
gl1.ylocator = mticker.MultipleLocator(LEFT_LAT_GRID)

# 单独设置坐标轴刻度和经纬度文字
# 经度标注保持不变；纬度标注改为每 5 度
xticks_left = np.arange(
    np.floor(region_left[0] / LEFT_LON_LABEL) * LEFT_LON_LABEL,
    np.ceil(region_left[1] / LEFT_LON_LABEL) * LEFT_LON_LABEL + LEFT_LON_LABEL,
    LEFT_LON_LABEL,
)
yticks_left = np.arange(
    np.floor(region_left[2] / LEFT_LAT_LABEL) * LEFT_LAT_LABEL,
    np.ceil(region_left[3] / LEFT_LAT_LABEL) * LEFT_LAT_LABEL + LEFT_LAT_LABEL,
    LEFT_LAT_LABEL,
)

ax1.set_xticks(xticks_left, crs=proj)
ax1.set_yticks(yticks_left, crs=proj)
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())

# 经纬度标注字体调小
ax1.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)

# 边框线宽
for spine in ax1.spines.values():
    spine.set_linewidth(0.8)

# 图号放左上角，10 pt，加粗
ax1.text(
    0.02, 0.98,
    "(a)",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    fontweight="bold",
    zorder=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=1.0),
)

# 左图色标
cbar1 = fig.colorbar(
    pcm1,
    ax=ax1,
    orientation="horizontal",
    fraction=0.055,
    pad=0.08,
)
cbar1.set_label("Pressure (hPa)", fontsize=8)
cbar1.ax.tick_params(labelsize=7)


# ============================================================
# 8. 右图：杭州附近 SI
# ============================================================
ax2.set_extent(region_right, crs=proj)

pcm2 = ax2.pcolormesh(
    lon_r,
    lat_r,
    si,
    cmap="RdYlBu_r",
    shading="auto",
    transform=proj,
)

# 等值线
simin = np.nanmin(si)
simax = np.nanmax(si)

levels_right = np.arange(
    np.floor(simin / SI_CONTOUR_INTERVAL) * SI_CONTOUR_INTERVAL,
    np.ceil(simax / SI_CONTOUR_INTERVAL) * SI_CONTOUR_INTERVAL + SI_CONTOUR_INTERVAL,
    SI_CONTOUR_INTERVAL,
)

cs2 = ax2.contour(
    lon_r,
    lat_r,
    si,
    levels=levels_right,
    colors="black",
    linewidths=0.35,
    transform=proj,
)

levels_right_label = [
    lev for lev in cs2.levels
    if np.isclose((lev / SI_LABEL_INTERVAL) - round(lev / SI_LABEL_INTERVAL), 0)
]

if len(levels_right_label) > 0:
    ax2.clabel(
        cs2,
        levels=levels_right_label,
        inline=True,
        fmt="%d",
        fontsize=6,
    )

# 海岸线 / 国界 / 陆海底图
ax2.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="none", zorder=0)
ax2.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
ax2.coastlines(resolution="10m", linewidth=0.5)
ax2.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)

# 右图网格和标注保持不变，只把字体调小
gl2 = ax2.gridlines(
    crs=proj,
    draw_labels=True,
    linewidth=0.4,
    color="gray",
    alpha=0.7,
    linestyle="-",
)

gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {"size": TICK_LABEL_SIZE}
gl2.ylabel_style = {"size": TICK_LABEL_SIZE}
gl2.xlocator = mticker.MultipleLocator(RIGHT_LON_MAJOR)
gl2.ylocator = mticker.MultipleLocator(RIGHT_LAT_MAJOR)

for spine in ax2.spines.values():
    spine.set_linewidth(0.8)

# 杭州中心点
ax2.plot(
    hangzhou_lon,
    hangzhou_lat,
    marker="o",
    markersize=4,
    color="red",
    markeredgecolor="black",
    markeredgewidth=0.4,
    transform=proj,
)

# 杭州文字标注
# x 加大 -> 更往右
# y 加大 -> 更往上
ax2.text(
    hangzhou_lon + 0.08,
    hangzhou_lat,
    "Hangzhou",
    fontsize=8,
    fontweight="bold",
    ha="left",
    va="center",
    transform=proj,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=1.5),
)

# 图号放左上角，10 pt，加粗
ax2.text(
    0.02, 0.98,
    "(b)",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    fontweight="bold",
    zorder=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=1.0),
)

# 右图色标
cbar2 = fig.colorbar(
    pcm2,
    ax=ax2,
    orientation="horizontal",
    fraction=0.055,
    pad=0.08,
)
cbar2.set_label("SI (°C)", fontsize=8)
cbar2.ax.tick_params(labelsize=7)


# ============================================================
# 9. 布局与输出
# ============================================================
plt.tight_layout()

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()