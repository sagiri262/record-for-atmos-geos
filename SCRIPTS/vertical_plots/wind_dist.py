# -*- coding: utf-8 -*-
"""
基于 wrfout_d01_* 某一天的数据，逐个时次输出三联图：

(a) 近地面风场：
    - 底图：10 m 风速
    - 小箭头：10 m 风向/风矢量
    - 黑色细线：海平面气压 SLP

(b) 500 hPa 风场：
    - 底图：500 hPa 风速
    - 弯曲箭头：500 hPa streamlines
    - 黑色细线：500 hPa 位势高度
    - 虚线：120E 剖面位置

(c) 120E 经向垂直剖面：
    - 填色：w 垂直速度
    - 黑色箭头：v-w 垂直面风场（其中 w 仅用于显示时放大）
    - 黑色细线：等压线
    - 灰色填充：地形（使用 WRF ter/HGT，即与模式一致的 DEM）

输出：
- 同一天所有 wrf 文件都输出到同一个文件夹
- 文件名按 wrfout 的时间命名

依赖：
pip install netCDF4 wrf-python cartopy scipy matplotlib numpy
"""

import os
import re
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from netCDF4 import Dataset
from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from wrf import (
    getvar,
    interplevel,
    latlon_coords,
    to_np,
)

# =========================================================
# 0. 中文字体设置（macOS）
# =========================================================
candidates = [
    "Hiragino Sans GB",
    "PingFang SC",
    "STHeiti",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
]

installed = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((name for name in candidates if name in installed), None)

if font_name is None:
    raise RuntimeError(
        "没有找到可用的中文字体。请先安装一个中文字体，例如 Noto Sans CJK SC。"
    )

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [font_name]
mpl.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 导入上级目录中的 wrf_read_data.py
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from wrf_read_data import WRFDataReader

# =========================================================
# 2. 参数设置
# =========================================================
wrf_path = "/Volumes/Lexar/WRF_Data/WRF_second_try/wrfout_d01_*"
reader = WRFDataReader(wrf_path)

# 获取排序后的文件列表
wrf_files = reader.get_files()

# -----------------------------
# 需要处理的日期
# -----------------------------
target_day = "2022-11-26"   # 修改成你要处理的日期
lon_section = 120.0         # 图3经向剖面的经度
output_dir = os.path.join(current_dir, f"wrf_three_panels_{target_day.replace('-', '')}")
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 3. 工具函数
# =========================================================
def nice_ceil(value, step):
    """把 value 向上取整到 step 的倍数。"""
    if step == 0:
        return value
    return np.ceil(value / step) * step


def extract_time_str(filepath):
    """从 wrfout 文件名中提取时间串。"""
    base = os.path.basename(filepath)
    return base.replace("wrfout_d01_", "")


def filter_files_by_day(files, day_str):
    """筛选某一天的 wrf 文件。"""
    out = []
    for f in files:
        if day_str in os.path.basename(f):
            out.append(f)
    return sorted(out)


def add_map_features(ax, extent):
    """给地图轴添加高分辨率海岸线等。"""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, edgecolor="black")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}


def regrid_to_regular_lonlat(lons2d, lats2d, data2d, nx=220, ny=220):
    """
    将曲线网格数据插值到规则经纬度网格，便于 streamplot。
    """
    lon_min = np.nanmin(lons2d)
    lon_max = np.nanmax(lons2d)
    lat_min = np.nanmin(lats2d)
    lat_max = np.nanmax(lats2d)

    lon_reg = np.linspace(lon_min, lon_max, nx)
    lat_reg = np.linspace(lat_min, lat_max, ny)
    lon2d_reg, lat2d_reg = np.meshgrid(lon_reg, lat_reg)

    points = np.column_stack((lons2d.ravel(), lats2d.ravel()))
    values = data2d.ravel()

    data_linear = griddata(points, values, (lon2d_reg, lat2d_reg), method="linear")
    data_nearest = griddata(points, values, (lon2d_reg, lat2d_reg), method="nearest")
    data_out = np.where(np.isfinite(data_linear), data_linear, data_nearest)

    return lon_reg, lat_reg, lon2d_reg, lat2d_reg, data_out


def height_km_to_pressure_hpa(z_km):
    """
    标准大气近似：高度(km) -> 压力(hPa)
    仅用于图3右侧副坐标轴。
    实际压力结构仍以剖面中的黑色等压线为准。
    """
    z_m = np.asarray(z_km) * 1000.0
    p = np.empty_like(z_m, dtype=float)

    # 对流层 (0~11 km)
    mask1 = z_m <= 11000.0
    p[mask1] = 1013.25 * (1.0 - 2.25577e-5 * z_m[mask1]) ** 5.25588

    # 11 km 以上简单指数近似
    mask2 = ~mask1
    p[mask2] = 226.321 * np.exp(-(z_m[mask2] - 11000.0) / 6341.62)
    return p


def pressure_hpa_to_height_km(p_hpa):
    """
    标准大气近似：压力(hPa) -> 高度(km)
    仅用于图3右侧副坐标轴。
    """
    p = np.asarray(p_hpa, dtype=float)
    z_m = np.empty_like(p, dtype=float)

    mask1 = p >= 226.321
    z_m[mask1] = (1.0 - (p[mask1] / 1013.25) ** (1.0 / 5.25588)) / 2.25577e-5

    mask2 = ~mask1
    z_m[mask2] = 11000.0 - 6341.62 * np.log(p[mask2] / 226.321)

    return z_m / 1000.0


def section_along_fixed_lon(lons2d, lats2d, lon_target, var3d_dict, terrain2d):
    """
    沿固定经度 lon_target，按每个 south_north 行选取最接近 lon_target 的网格点，
    构造一个近似经向剖面。

    返回：
        sec_lat          : (ny,)
        sec_lon          : (ny,)
        terrain_sec      : (ny,)
        vars_sec[name]   : (nz, ny)
    """
    lons_np = np.asarray(lons2d)
    lats_np = np.asarray(lats2d)

    lon_min = np.nanmin(lons_np)
    lon_max = np.nanmax(lons_np)
    if not (lon_min <= lon_target <= lon_max):
        raise ValueError(
            f"指定剖面经度 {lon_target}E 不在当前模式区域经度范围内: [{lon_min:.2f}, {lon_max:.2f}]"
        )

    ny, nx = lons_np.shape
    jj = np.arange(ny)

    # 每一行选取最接近目标经度的格点
    i_sec = np.argmin(np.abs(lons_np - lon_target), axis=1)

    sec_lat = lats_np[jj, i_sec]
    sec_lon = lons_np[jj, i_sec]
    terrain_sec = terrain2d[jj, i_sec]

    vars_sec = {}
    for key, arr3d in var3d_dict.items():
        # arr3d shape = (nz, ny, nx)
        vars_sec[key] = arr3d[:, jj, i_sec]

    # 按纬度从大到小排序：左北右南
    order = np.argsort(sec_lat)[::-1]
    sec_lat = sec_lat[order]
    sec_lon = sec_lon[order]
    terrain_sec = terrain_sec[order]

    for key in vars_sec:
        vars_sec[key] = vars_sec[key][:, order]

    return sec_lat, sec_lon, terrain_sec, vars_sec


def plot_one_wrf_file(wrf_file, output_dir, lon_section=120.0):
    """
    对单个 wrfout 文件作图并保存。
    """
    time_str = extract_time_str(wrf_file)
    print(f"正在处理: {time_str}")

    ncfile = Dataset(wrf_file)

    # -----------------------------
    # 读取变量
    # -----------------------------
    slp = getvar(ncfile, "slp")                           # hPa
    ter = getvar(ncfile, "ter")                           # m
    uv10 = getvar(ncfile, "uvmet10", units="m s-1")       # (2, y, x)
    pressure = getvar(ncfile, "pressure")                 # hPa
    z = getvar(ncfile, "z", units="m")                    # m
    uvmet = getvar(ncfile, "uvmet", units="m s-1")        # (2, z, y, x)
    va = getvar(ncfile, "va", units="m s-1")              # (z, y, x)
    wa = getvar(ncfile, "wa", units="m s-1")              # (z, y, x)

    lats, lons = latlon_coords(slp)

    lats_np = to_np(lats)
    lons_np = to_np(lons)

    slp_np = to_np(slp)
    ter_np = to_np(ter)

    u10_np = to_np(uv10[0])
    v10_np = to_np(uv10[1])
    ws10_np = np.hypot(u10_np, v10_np)

    pres_np = to_np(pressure)
    z_np = to_np(z)
    va_np = to_np(va)
    wa_np = to_np(wa)

    u3d_np = to_np(uvmet[0])
    v3d_np = to_np(uvmet[1])

    # 500 hPa
    u500 = to_np(interplevel(uvmet[0], pressure, 500.0))
    v500 = to_np(interplevel(uvmet[1], pressure, 500.0))
    z500 = to_np(interplevel(z, pressure, 500.0))         # m
    ws500 = np.hypot(u500, v500)

    # -----------------------------
    # 地图范围
    # -----------------------------
    lon_min, lon_max = np.nanmin(lons_np), np.nanmax(lons_np)
    lat_min, lat_max = np.nanmin(lats_np), np.nanmax(lats_np)
    dlon = lon_max - lon_min
    dlat = lat_max - lat_min

    extent = [
        lon_min - 0.03 * dlon,
        lon_max + 0.03 * dlon,
        lat_min - 0.03 * dlat,
        lat_max + 0.03 * dlat,
    ]

    # -----------------------------
    # 图3：120E 经向剖面
    # -----------------------------
    sec_lat, sec_lon, terrain_sec, vars_sec = section_along_fixed_lon(
        lons_np,
        lats_np,
        lon_section,
        var3d_dict={
            "z": z_np,
            "p": pres_np,
            "v": va_np,
            "w": wa_np,
        },
        terrain2d=ter_np,
    )

    z_sec = vars_sec["z"] / 1000.0     # km
    p_sec = vars_sec["p"]              # hPa
    v_sec = vars_sec["v"]              # m/s
    w_sec = vars_sec["w"]              # m/s

    # 剖面填色：w
    w_abs = np.nanpercentile(np.abs(w_sec), 98)
    w_max = max(0.5, nice_ceil(w_abs, 0.2))
    w_levels = np.linspace(-w_max, w_max, 17)

    # 等压线
    p_levels = [100, 200, 300, 500, 700, 850, 900, 950, 1000]

    # 右轴最大高度
    y_max_km = min(18.0, max(8.0, np.nanpercentile(z_sec, 99)))

    # -----------------------------
    # 图2 streamplot 需要规则经纬网格
    # -----------------------------
    lon_reg, lat_reg, lon2d_reg, lat2d_reg, u500_reg = regrid_to_regular_lonlat(
        lons_np, lats_np, u500, nx=220, ny=220
    )
    _, _, _, _, v500_reg = regrid_to_regular_lonlat(
        lons_np, lats_np, v500, nx=220, ny=220
    )
    _, _, _, _, ws500_reg = regrid_to_regular_lonlat(
        lons_np, lats_np, ws500, nx=220, ny=220
    )
    _, _, _, _, z500_reg = regrid_to_regular_lonlat(
        lons_np, lats_np, z500, nx=220, ny=220
    )

    # -----------------------------
    # 画图
    # -----------------------------
    fig = plt.figure(figsize=(30, 8))
    gs = fig.add_gridspec(1, 3, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[0, 2])

    # ========== (a) 10m 风场 + SLP ==========
    add_map_features(ax1, extent)

    ws10_top = max(2.0, nice_ceil(np.nanpercentile(ws10_np, 98), 2.0))
    levels1 = np.linspace(0, ws10_top, 13)

    cf1 = ax1.contourf(
        lons_np,
        lats_np,
        ws10_np,
        levels=levels1,
        cmap="YlGnBu",
        extend="max",
        transform=ccrs.PlateCarree(),
    )

    # SLP 等值线
    slp_min = np.nanmin(slp_np)
    slp_max = np.nanmax(slp_np)
    slp_start = np.floor(slp_min / 2.0) * 2.0
    slp_end = np.ceil(slp_max / 2.0) * 2.0
    slp_levels = np.arange(slp_start, slp_end + 0.1, 2.0)

    cs1 = ax1.contour(
        lons_np,
        lats_np,
        slp_np,
        levels=slp_levels,
        colors="k",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
    )
    ax1.clabel(cs1, fmt="%.0f", fontsize=8, inline=True)

    # 小箭头
    skip = max(1, int(min(ws10_np.shape) / 25))
    ax1.quiver(
        lons_np[::skip, ::skip],
        lats_np[::skip, ::skip],
        u10_np[::skip, ::skip],
        v10_np[::skip, ::skip],
        transform=ccrs.PlateCarree(),
        color="k",
        scale=300,
        width=0.0022,
        headwidth=3.5,
        headlength=4.0,
    )

    # 剖面线
    ax1.plot(
        [lon_section, lon_section],
        [lat_min, lat_max],
        linestyle="--",
        linewidth=1.2,
        color="magenta",
        transform=ccrs.PlateCarree(),
    )

    ax1.text(
        0.02, 0.98, "(a)",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )

    cbar1 = fig.colorbar(cf1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.90)
    cbar1.set_label("10 m 风速 (m s$^{-1}$)")

    # ========== (b) 500 hPa 风场 streamlines ==========
    add_map_features(ax2, extent)

    ws500_top = max(10.0, nice_ceil(np.nanpercentile(ws500_reg, 98), 2.0))
    levels2 = np.linspace(0, ws500_top, 13)

    cf2 = ax2.contourf(
        lon2d_reg,
        lat2d_reg,
        ws500_reg,
        levels=levels2,
        cmap="plasma",
        extend="max",
        transform=ccrs.PlateCarree(),
    )

    # 500 hPa 位势高度
    z500_levels = np.arange(
        np.floor(np.nanmin(z500_reg) / 60.0) * 60.0,
        np.ceil(np.nanmax(z500_reg) / 60.0) * 60.0 + 1,
        60.0,
    )

    cs2 = ax2.contour(
        lon2d_reg,
        lat2d_reg,
        z500_reg,
        levels=z500_levels,
        colors="k",
        linewidths=0.55,
        transform=ccrs.PlateCarree(),
    )
    ax2.clabel(cs2, fmt="%.0f", fontsize=8, inline=True)

    # 弯曲箭头：streamlines
    ax2.streamplot(
        lon_reg,
        lat_reg,
        u500_reg,
        v500_reg,
        density=1.6,
        color="k",
        linewidth=0.8,
        arrowsize=1.0,
        transform=ccrs.PlateCarree(),
    )

    ax2.plot(
        [lon_section, lon_section],
        [lat_min, lat_max],
        linestyle="--",
        linewidth=1.2,
        color="magenta",
        transform=ccrs.PlateCarree(),
    )

    ax2.text(
        0.02, 0.98, "(b)",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )

    cbar2 = fig.colorbar(cf2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.90)
    cbar2.set_label("500 hPa 风速 (m s$^{-1}$)")

    # ========== (c) 120E 经向垂直剖面 ==========
    # X/Y 都用 2D，便于直接用真实高度场作图
    X = np.tile(sec_lat[None, :], (z_sec.shape[0], 1))
    Y = z_sec

    cf3 = ax3.contourf(
        X,
        Y,
        w_sec,
        levels=w_levels,
        cmap="RdBu_r",
        extend="both",
    )

    # 等压线（真实压力场）
    cs3 = ax3.contour(
        X,
        Y,
        p_sec,
        levels=[lv for lv in p_levels if np.nanmin(p_sec) <= lv <= np.nanmax(p_sec)],
        colors="k",
        linewidths=0.55,
    )
    ax3.clabel(cs3, fmt="%d", fontsize=8, inline=True)

    # 地形填充（WRF ter，即模式 DEM）
    ax3.fill_between(
        sec_lat,
        0.0,
        terrain_sec / 1000.0,
        color="0.5",
        alpha=0.6,
        zorder=5,
    )

    # 垂直面风场：v-w
    # 注意：为了让箭头可见，w 仅在显示时放大
    w_display_factor = 50.0
    skipx = max(1, int(sec_lat.size / 24))
    skipz = max(1, int(z_sec.shape[0] / 22))

    q = ax3.quiver(
        X[::skipz, ::skipx],
        Y[::skipz, ::skipx],
        v_sec[::skipz, ::skipx],
        w_sec[::skipz, ::skipx] * w_display_factor,
        angles="xy",
        scale_units="xy",
        scale=100,
        width=0.0022,
        color="k",
        zorder=6,
    )

    ax3.quiverkey(
        q,
        X=0.98,
        Y=1.03,
        U=10,
        label="v = 10 m/s，w 显示放大 ×50",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 9},
    )

    ax3.set_ylim(0, y_max_km)

    # 左北右南
    ax3.set_xlim(np.nanmax(sec_lat), np.nanmin(sec_lat))

    ax3.set_xlabel("纬度 (°)")
    ax3.set_ylabel("高度 (km)")

    # 右侧压力副坐标（标准大气近似）
    secax = ax3.secondary_yaxis(
        "right",
        functions=(height_km_to_pressure_hpa, pressure_hpa_to_height_km),
    )
    secax.set_ylabel("近似气压 (hPa)")
    secax.set_yticks([1000, 850, 700, 500, 300, 200, 100])

    # 横轴刻度
    n_xticks = 6
    xticks = np.linspace(np.nanmax(sec_lat), np.nanmin(sec_lat), n_xticks)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([f"{x:.1f}" for x in xticks])

    ax3.text(
        0.02, 0.98, "(c)",
        transform=ax3.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )

    cbar3 = fig.colorbar(cf3, ax=ax3, orientation="horizontal", pad=0.08, shrink=0.90)
    cbar3.set_label("垂直速度 w (m s$^{-1}$)")

    # 不加总标题；只在图下方加时间信息
    fig.text(
        0.5,
        0.01,
        f"{time_str}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    # 保存
    out_name = f"{time_str}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ncfile.close()

    print(f"已保存: {out_path}")


# =========================================================
# 4. 主程序：处理指定日期全部文件
# =========================================================
day_files = filter_files_by_day(wrf_files, target_day)

if len(day_files) == 0:
    raise FileNotFoundError(f"没有找到日期 {target_day} 对应的 wrfout 文件。")

print(f"日期 {target_day} 共找到 {len(day_files)} 个文件")
print(f"输出目录: {output_dir}")

for wf in day_files:
    plot_one_wrf_file(wf, output_dir=output_dir, lon_section=lon_section)

print("全部完成。")