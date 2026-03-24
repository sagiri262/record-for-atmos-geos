# -*- coding: utf-8 -*-
"""
从指定 wrfout 文件中读取某一时刻的冰雪相关变量，完成：
1. 检查关键变量是否存在
2. 分离雨和雪
3. 绘制积雪覆盖区域图
4. 若可获得积雪深度，则绘制积雪深度空间分布图
5. 可选绘制液态降水 / 冻结降水 / 反照率分布图

说明：
- 优先使用 SNOWNC 作为冻结降水
- 若没有 SNOWNC，则退回使用 SR * 总降水 近似冻结降水
- 积雪深度优先使用 SNOWH / SNOW_DEPTH
- 若没有深度变量，则尝试用 SWE 或 SNOW 估算积雪深度
"""

import os
import sys
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from wrf import getvar, to_np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from datetime import datetime


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

# 指定目标文件名
target_basename = "wrfout_d01_2022-11-26_18_00_00"

# 查找目标文件
target_file = None
for f in wrf_files:
    if os.path.basename(f) == target_basename:
        target_file = f
        break

if target_file is None:
    raise FileNotFoundError(f"没有找到目标文件: {target_basename}")

print(f"读取文件: {target_file}")
ncfile = Dataset(target_file)

# 时间索引：如果 wrfout 里只有一个时间，一般用 0 或 -1 都可以
timeidx = -1

# 若用 SWE 推算积雪深度，雪密度取值（kg/m^3）
rho_snow = 100.0

# 输出目录
out_dir = "../FIGS"
os.makedirs(out_dir, exist_ok=True)


# =========================================================
# 3. 工具函数
# =========================================================
def read2d_from_var(nc, name, timeidx=-1):
    """
    从 nc.variables 中读取二维场：
    - 若变量是 (Time, south_north, west_east)，取指定 timeidx
    - 若变量本身就是二维，则直接读取
    """
    if name not in nc.variables:
        return None

    var = nc.variables[name]

    if var.ndim == 2:
        return np.asarray(var[:, :], dtype=np.float64)
    elif var.ndim >= 3:
        return np.asarray(var[timeidx, :, :], dtype=np.float64)
    else:
        return np.asarray(var[...], dtype=np.float64)


def get_latlon_2d(nc):
    """
    优先用 wrf.getvar 读取 lat/lon
    """
    lats = getvar(nc, "lat")
    lons = getvar(nc, "lon")
    return np.asarray(to_np(lats), dtype=np.float64), np.asarray(to_np(lons), dtype=np.float64)


def print_var_status(nc):
    """
    检查关键变量是否存在
    """
    key_vars = [
        "SNOWC",
        "SNOWH",
        "SWE",
        "SNOW_DEPTH",
        "SNOW",
        "ALBEDO",
        "SNOWNC",
        "RAINC",
        "RAINNC",
        "RAINSH",
        "SR",
    ]

    print("========== 冰雪相关变量检查结果 ==========")
    for name in key_vars:
        if name in nc.variables:
            print(f"{name}: found")
        else:
            print(f"{name}: no such variable...")
    print("========================================\n")


def choose_swe(nc, timeidx=-1):
    """
    优先使用 SWE
    若没有 SWE，则尝试使用 SNOW（很多 wrfout 用它表示雪水当量）
    """
    for name in ("SWE", "SNOW"):
        arr = read2d_from_var(nc, name, timeidx)
        if arr is not None:
            return name, arr
    return None, None


def choose_snow_depth(nc, timeidx=-1, rho_snow=100.0):
    """
    积雪深度优先级：
    1. SNOWH
    2. SNOW_DEPTH
    3. SWE/SNOW 推算

    返回：
    depth_name, depth_m, is_estimated
    """
    snowh = read2d_from_var(nc, "SNOWH", timeidx)
    if snowh is not None:
        return "SNOWH", snowh, False

    snow_depth = read2d_from_var(nc, "SNOW_DEPTH", timeidx)
    if snow_depth is not None:
        return "SNOW_DEPTH", snow_depth, False

    swe_name, swe = choose_swe(nc, timeidx)
    if swe is not None:
        # SWE 若按 kg/m^2 计，则 depth = SWE / rho_snow，单位 m
        depth_m = swe / rho_snow
        return f"{swe_name}_估算深度", depth_m, True

    return None, None, False


def separate_rain_snow(nc, timeidx=-1):
    """
    雨雪分离：
    total_precip = RAINC + RAINNC + RAINSH
    frozen_precip 优先用 SNOWNC
    若没有 SNOWNC，则 frozen_precip = total_precip * SR
    liquid_precip = total_precip - frozen_precip

    注意：
    这些常常是累计量。如果你想算某个时段降水，应对相邻两个时次做差分。
    """
    rainc = read2d_from_var(nc, "RAINC", timeidx)
    rainnc = read2d_from_var(nc, "RAINNC", timeidx)
    rainsh = read2d_from_var(nc, "RAINSH", timeidx)
    snownc = read2d_from_var(nc, "SNOWNC", timeidx)
    sr = read2d_from_var(nc, "SR", timeidx)

    total_precip = None
    parts = [x for x in (rainc, rainnc, rainsh) if x is not None]
    if len(parts) > 0:
        total_precip = np.sum(parts, axis=0)

    frozen_precip = None
    frozen_source = None

    if snownc is not None:
        frozen_precip = np.asarray(snownc, dtype=np.float64)
        frozen_source = "SNOWNC"
    elif (total_precip is not None) and (sr is not None):
        frozen_precip = np.clip(total_precip * sr, 0.0, None)
        frozen_source = "SR × 总降水"

    liquid_precip = None
    if (total_precip is not None) and (frozen_precip is not None):
        liquid_precip = np.clip(total_precip - frozen_precip, 0.0, None)
    elif total_precip is not None:
        liquid_precip = np.asarray(total_precip, dtype=np.float64)

    return {
        "RAINC": rainc,
        "RAINNC": rainnc,
        "RAINSH": rainsh,
        "SNOWNC": snownc,
        "SR": sr,
        "total_precip": total_precip,
        "frozen_precip": frozen_precip,
        "liquid_precip": liquid_precip,
        "frozen_source": frozen_source,
    }


def get_snow_cover_mask(nc, timeidx=-1, rho_snow=100.0):
    """
    积雪覆盖区域判定优先级：
    1. SNOWC
    2. SNOWH
    3. SNOW_DEPTH
    4. SWE / SNOW
    5. frozen_precip
    """
    snowc = read2d_from_var(nc, "SNOWC", timeidx)
    if snowc is not None:
        return "SNOWC", snowc > 0.5

    snowh = read2d_from_var(nc, "SNOWH", timeidx)
    if snowh is not None:
        return "SNOWH", snowh > 1.0e-4

    snow_depth = read2d_from_var(nc, "SNOW_DEPTH", timeidx)
    if snow_depth is not None:
        return "SNOW_DEPTH", snow_depth > 1.0e-4

    swe_name, swe = choose_swe(nc, timeidx)
    if swe is not None:
        return swe_name, swe > 0.1

    sep = separate_rain_snow(nc, timeidx)
    if sep["frozen_precip"] is not None:
        return sep["frozen_source"], sep["frozen_precip"] > 0.1

    return None, None


def plot_field(lons, lats, field, title, cbar_label, out_png,
               cmap="viridis", vmin=None, vmax=None):
    """
    带海岸线和国界线的二维空间分布图
    """
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)

    # 根据数据范围设置显示区域
    lon_min = np.nanmin(lons)
    lon_max = np.nanmax(lons)
    lat_min = np.nanmin(lats)
    lat_max = np.nanmax(lats)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # 底图要素
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="whitesmoke")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="lightcyan")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, linestyle="-")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="black", linewidth=0.4)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor="blue", linewidth=0.3)

    # 绘制数据场
    pm = ax.pcolormesh(
        lons, lats, field,
        transform=proj,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    # 色标
    cbar = plt.colorbar(pm, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label(cbar_label, fontsize=11)

    # 经纬网
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title, fontsize=13)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"图已保存: {out_png}")
    plt.close(fig)


def plot_mask(lons, lats, mask, title, out_png):
    plot_data = np.where(mask, 1.0, np.nan)
    plot_field(
        lons=lons,
        lats=lats,
        field=plot_data,
        title=title,
        cbar_label="积雪覆盖",
        out_png=out_png,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

# =========================================================
# 4. 读取经纬度与变量检查
# =========================================================
lats2d, lons2d = get_latlon_2d(ncfile)
print_var_status(ncfile)


# =========================================================
# 5. 雨雪分离
# =========================================================
sep = separate_rain_snow(ncfile, timeidx=timeidx)

print("========== 雨雪分离结果 ==========")
print(f"冻结降水来源: {sep['frozen_source']}")
print(f"液态降水是否可用: {'是' if sep['liquid_precip'] is not None else '否'}")
print(f"冻结降水是否可用: {'是' if sep['frozen_precip'] is not None else '否'}")

if sep["total_precip"] is not None:
    print(f"总降水范围: {np.nanmin(sep['total_precip']):.3f} ~ {np.nanmax(sep['total_precip']):.3f}")
if sep["liquid_precip"] is not None:
    print(f"液态降水范围: {np.nanmin(sep['liquid_precip']):.3f} ~ {np.nanmax(sep['liquid_precip']):.3f}")
if sep["frozen_precip"] is not None:
    print(f"冻结降水范围: {np.nanmin(sep['frozen_precip']):.3f} ~ {np.nanmax(sep['frozen_precip']):.3f}")
print("=================================\n")


# =========================================================
# 6. 获取积雪覆盖区域
# =========================================================
cover_source, cover_mask = get_snow_cover_mask(ncfile, timeidx=timeidx, rho_snow=rho_snow)

print("========== 积雪覆盖判定 ==========")
print(f"积雪覆盖来源: {cover_source}")
print("=================================\n")


# =========================================================
# 7. 获取积雪深度
# =========================================================
depth_source, snow_depth_m, depth_is_estimated = choose_snow_depth(
    ncfile, timeidx=timeidx, rho_snow=rho_snow
)

print("========== 积雪深度结果 ==========")
print(f"积雪深度来源: {depth_source}")
print(f"是否由 SWE/SNOW 估算: {'是' if depth_is_estimated else '否'}")
if snow_depth_m is not None:
    valid = snow_depth_m[np.isfinite(snow_depth_m)]
    if valid.size > 0:
        print(f"积雪深度范围: {np.nanmin(valid):.4f} ~ {np.nanmax(valid):.4f} m")
print("=================================\n")


# =========================================================
# 8. 可选读取反照率
# =========================================================
albedo = read2d_from_var(ncfile, "ALBEDO", timeidx=timeidx)


# =========================================================
# 9. 绘图：液态降水
# =========================================================
time_str = target_basename.replace("wrfout_d01_", "")

if sep["liquid_precip"] is not None:
    liquid_plot = np.where(
        np.isfinite(sep["liquid_precip"]) & (sep["liquid_precip"] > 0),
        sep["liquid_precip"],
        np.nan
    )
    out_png = os.path.join(out_dir, "liquid_precip.png")
    plot_field(
        lons2d, lats2d, liquid_plot,
        title=f"液态降水空间分布\n时间: {time_str}",
        cbar_label="液态降水",
        out_png=out_png,
        cmap="Greens"
    )
else:
    print("无法绘制液态降水图：缺少降水变量。")


# =========================================================
# 10. 绘图：冻结降水
# =========================================================
if sep["frozen_precip"] is not None:
    frozen_plot = np.where(
        np.isfinite(sep["frozen_precip"]) & (sep["frozen_precip"] > 0),
        sep["frozen_precip"],
        np.nan
    )
    out_png = os.path.join(out_dir, "frozen_precip.png")
    plot_field(
        lons2d, lats2d, frozen_plot,
        title=f"冻结降水空间分布（来源: {sep['frozen_source']}）\n时间: {time_str}",
        cbar_label="冻结降水",
        out_png=out_png,
        cmap="PuBu"
    )
else:
    print("无法绘制冻结降水图：没有 SNOWNC，也没有可用的 SR。")


# =========================================================
# 11. 绘图：积雪覆盖区域
# =========================================================
if cover_mask is not None:
    out_png = os.path.join(out_dir, "snow_cover_region.png")
    plot_mask(
        lons2d, lats2d, cover_mask,
        title=f"积雪覆盖区域分布图（来源: {cover_source}）\n时间: {time_str}",
        out_png=out_png
    )
else:
    print("无法绘制积雪覆盖区域图：没有可用的雪相关变量。")


# =========================================================
# 12. 绘图：积雪深度
# =========================================================
if snow_depth_m is not None:
    depth_plot = np.where(
        np.isfinite(snow_depth_m) & (snow_depth_m > 0),
        snow_depth_m,
        np.nan
    )

    suffix = "（由 SWE/SNOW 估算）" if depth_is_estimated else ""
    out_png = os.path.join(out_dir, "snow_depth_distribution.png")
    plot_field(
        lons2d, lats2d, depth_plot,
        title=f"积雪深度空间分布图（来源: {depth_source}）{suffix}\n时间: {time_str}",
        cbar_label="积雪深度（m）",
        out_png=out_png,
        cmap="Blues"
    )
else:
    print("无法绘制积雪深度图：没有 SNOWH / SNOW_DEPTH，也无法通过 SWE(SNOW) 估算。")


# =========================================================
# 13. 可选：绘图 反照率
# =========================================================
if albedo is not None:
    albedo_plot = np.where(np.isfinite(albedo), albedo, np.nan)
    out_png = os.path.join(out_dir, "albedo.png")
    plot_field(
        lons2d, lats2d, albedo_plot,
        title=f"地表反照率空间分布图\n时间: {time_str}",
        cbar_label="反照率",
        out_png=out_png,
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0
    )


# =========================================================
# 14. 结束
# =========================================================
ncfile.close()
print("\n全部处理完成。")