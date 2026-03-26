# -*- coding: utf-8 -*-
"""
读取第一个 wrfout_d01_* 文件，分别生成 3 张单独图：
1) 近地面风场
2) 500 hPa 风流场
3) 120E 经向垂直剖面（w 填色 + 风矢量 + 风速等值线 + 等压线 + 地形）

说明：
- 第三张图中的“风速”等值线，这里定义为剖面风速模：
    speed_sec = sqrt(v^2 + w^2)
  若你只想画经向风速 |v|，把下面 speed_sec 那一行改掉即可。
"""

import os
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
wrf_files = reader.get_files()

if len(wrf_files) == 0:
    raise FileNotFoundError("没有找到 wrfout_d01_* 文件。")

target_file = wrf_files[0]   # 只处理第一个文件
target_basename = os.path.basename(target_file)
time_str = target_basename.replace("wrfout_d01_", "")

print(f"读取文件: {target_file}")

output_dir = os.path.join(current_dir, "wrf_single_figures")
os.makedirs(output_dir, exist_ok=True)

lon_section = 120.0  # 第三张图的剖面经度

# =========================================================
# 3. 工具函数
# =========================================================
def nice_ceil(value, step):
    if step == 0:
        return value
    return np.ceil(value / step) * step


def add_map_features(ax, extent):
    '''
    学习绘制图像的参数
    extent 表示四至范围， 加入ccrs

    coastlines表示海岸线，分辨率 resolution，线宽 0.8cm，颜色设置为黑色
    颜色设置要怎么做？比如红、黑、渐变等等
    
    '''
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, edgecolor="black")

    """
    绘制格网线
    """
    
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    """
    
    """

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}


def regrid_to_regular_lonlat(lons2d, lats2d, data2d, nx=220, ny=220):
    """
    经纬度
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
    计算纵坐标，从wrfout文件里提取z坐标高度
    转换到 km 或 m 尺度下
    使用掩膜 mask 
    11 km 以下和11 km 以上的大气层，温度随高度变化规律不同，所以气压公式也不同，要分段算。

    """
    z_m = np.asarray(z_km) * 1000.0
    p = np.empty_like(z_m, dtype=float)

    # 判断高度高于或低于 11km
    mask1 = z_m <= 11000.0
    # 这一行是在算 11 km 以下（对流层） 的气压。
    p[mask1] = 1013.25 * (1.0 - 2.25577e-5 * z_m[mask1]) ** 5.25588

    mask2 = ~mask1
    # 这一行是在算 11 km 以上 的气压。
    p[mask2] = 226.321 * np.exp(-(z_m[mask2] - 11000.0) / 6341.62)
    return p


def pressure_hpa_to_height_km(p_hpa):
    p = np.asarray(p_hpa, dtype=float)
    z_m = np.empty_like(p, dtype=float)

    mask1 = p >= 226.321
    z_m[mask1] = (1.0 - (p[mask1] / 1013.25) ** (1.0 / 5.25588)) / 2.25577e-5

    mask2 = ~mask1
    z_m[mask2] = 11000.0 - 6341.62 * np.log(p[mask2] / 226.321)

    return z_m / 1000.0


def section_along_fixed_lon(lons2d, lats2d, lon_target, var3d_dict, terrain2d):
    lons_np = np.asarray(lons2d)
    lats_np = np.asarray(lats2d)

    lon_min = np.nanmin(lons_np)
    lon_max = np.nanmax(lons_np)
    if not (lon_min <= lon_target <= lon_max):
        raise ValueError(
            f"指定剖面经度 {lon_target}E 不在当前区域范围 [{lon_min:.2f}, {lon_max:.2f}] 内。"
        )

    ny, nx = lons_np.shape
    jj = np.arange(ny)

    i_sec = np.argmin(np.abs(lons_np - lon_target), axis=1)

    sec_lat = lats_np[jj, i_sec]
    sec_lon = lons_np[jj, i_sec]
    terrain_sec = terrain2d[jj, i_sec]

    vars_sec = {}
    for key, arr3d in var3d_dict.items():
        vars_sec[key] = arr3d[:, jj, i_sec]

    order = np.argsort(sec_lat)[::-1]   # 左北右南
    sec_lat = sec_lat[order]
    sec_lon = sec_lon[order]
    terrain_sec = terrain_sec[order]

    for key in vars_sec:
        vars_sec[key] = vars_sec[key][:, order]

    return sec_lat, sec_lon, terrain_sec, vars_sec


# =========================================================
# 4. 读取数据
# =========================================================
ncfile = Dataset(target_file)

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
z500 = to_np(interplevel(z, pressure, 500.0))
ws500 = np.hypot(u500, v500)

# 区域范围
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

# =========================================================
# 读取三维降水粒子变量（用于第三张图）
# =========================================================
# 优先使用“真正降水粒子”：
#   QRAIN + QSNOW + QGRAUP
# 如果某些变量不存在，就自动跳过
precip_candidates = ["QRAIN", "QSNOW", "QGRAUP"]
used_precip_vars = [v for v in precip_candidates if v in ncfile.variables]

if len(used_precip_vars) == 0:
    raise RuntimeError(
        "当前 wrfout 中没有找到 QRAIN/QSNOW/QGRAUP，无法绘制垂直降水分布。"
    )

print("\n第三张图填色将使用以下三维降水粒子变量：")
print(used_precip_vars)

precip3d = None
for vname in used_precip_vars:
    arr = np.asarray(ncfile.variables[vname][0, :, :, :], dtype=np.float64)  # kg/kg
    if precip3d is None:
        precip3d = arr.copy()
    else:
        precip3d += arr

# 图3剖面
sec_lat, sec_lon, terrain_sec, vars_sec = section_along_fixed_lon(
    lons_np,
    lats_np,
    lon_section,
    var3d_dict={
        "z": z_np,
        "p": pres_np,
        "v": va_np,
        "w": wa_np,
        "precip": precip3d,
    },
    terrain2d=ter_np,
)

z_sec = vars_sec["z"] / 1000.0    # km
p_sec = vars_sec["p"]             # hPa
v_sec = vars_sec["v"]             # m/s
w_sec = vars_sec["w"]             # m/s

precip_sec_kgkg = vars_sec["precip"]
precip_sec = precip_sec_kgkg * 1e3

# 第三图新增：风速等值线
# 这里按剖面内风速模定义；若只想看经向风速，把这一行改为 np.abs(v_sec)
speed_sec = np.sqrt(v_sec**2 + w_sec**2)

# 打印统计量
valid_precip = precip_sec[precip_sec > 0]

precip_masked = np.ma.masked_less_equal(precip_sec, 0.0)

# 构造一个 “垂直输送诊断量”
precip_vflux = w_sec * precip_sec_kgkg

print("\n第三张图剖面降水粒子统计（QRAIN+QSNOW+QGRAUP）：")
if valid_precip.size > 0:
    print(f"最大值: {np.nanmax(valid_precip):.6f} g/kg")
    print(f"最小值: {np.nanmin(valid_precip):.6f} g/kg")
    print(f"平均值: {np.nanmean(valid_precip):.6f} g/kg")
else:
    print("该剖面上降水粒子全为 0。")

valid_flux = precip_vflux[np.isfinite(precip_vflux)]
print("\n第三张图诊断垂直输送量 w*qprecip 统计：")
print(f"最大值: {np.nanmax(valid_flux):.8e}")
print(f"最小值: {np.nanmin(valid_flux):.8e}")
print(f"平均值: {np.nanmean(valid_flux):.8e}")

print("\n第三张图剖面风速统计：")
print(f"最大值: {np.nanmax(speed_sec):.3f} m/s")
print(f"最小值: {np.nanmin(speed_sec):.3f} m/s")
print(f"平均值: {np.nanmean(speed_sec):.3f} m/s")

# 图2插值到规则经纬度网格，便于 streamplot
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

# =========================================================
# 5. 图1：近地面风场
# =========================================================
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
add_map_features(ax1, extent)

ws10_top = max(2.0, nice_ceil(np.nanpercentile(ws10_np, 98), 2.0))
levels1 = np.linspace(0, ws10_top, 13)

cf1 = ax1.contourf(
    lons_np, lats_np, ws10_np,
    levels=levels1,
    cmap="YlGnBu",
    extend="max",
    transform=ccrs.PlateCarree(),
)

slp_min = np.nanmin(slp_np)
slp_max = np.nanmax(slp_np)
slp_start = np.floor(slp_min / 2.0) * 2.0
slp_end = np.ceil(slp_max / 2.0) * 2.0
slp_levels = np.arange(slp_start, slp_end + 0.1, 2.0)

cs1 = ax1.contour(
    lons_np, lats_np, slp_np,
    levels=slp_levels,
    colors="k",
    linewidths=0.5,
    transform=ccrs.PlateCarree(),
)
ax1.clabel(cs1, fmt="%.0f", fontsize=8, inline=True)

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

cbar1 = fig1.colorbar(cf1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.9)
cbar1.set_label("10 m 风速 (m s$^{-1}$)")

fig1_path = os.path.join(output_dir, f"{time_str}_panel_a_surface_wind.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"已保存: {fig1_path}")

# =========================================================
# 6. 图2：500 hPa 风流场
# =========================================================
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
add_map_features(ax2, extent)

ws500_top = max(10.0, nice_ceil(np.nanpercentile(ws500_reg, 98), 2.0))
levels2 = np.linspace(0, ws500_top, 13)

cf2 = ax2.contourf(
    lon2d_reg, lat2d_reg, ws500_reg,
    levels=levels2,
    cmap="plasma",
    extend="max",
    transform=ccrs.PlateCarree(),
)

z500_levels = np.arange(
    np.floor(np.nanmin(z500_reg) / 60.0) * 60.0,
    np.ceil(np.nanmax(z500_reg) / 60.0) * 60.0 + 1,
    60.0,
)

cs2 = ax2.contour(
    lon2d_reg, lat2d_reg, z500_reg,
    levels=z500_levels,
    colors="k",
    linewidths=0.55,
    transform=ccrs.PlateCarree(),
)
ax2.clabel(cs2, fmt="%.0f", fontsize=8, inline=True)

ax2.streamplot(
    lon_reg, lat_reg,
    u500_reg, v500_reg,
    density=1.6,
    color="k",
    linewidth=0.8,
    arrowsize=1.0,
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

cbar2 = fig2.colorbar(cf2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.9)
cbar2.set_label("500 hPa 风速 (m s$^{-1}$)")

fig2_path = os.path.join(output_dir, f"{time_str}_panel_b_500hpa_stream.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"已保存: {fig2_path}")

# =========================================================
# 7. 图3：120E 经向垂直剖面
# =========================================================
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(1, 1, 1)

X = np.tile(sec_lat[None, :], (z_sec.shape[0], 1))
Y = z_sec

# -----------------------------
# 1. 背景填色：降水粒子混合比（只画 > 0）
# -----------------------------
positive_precip = precip_sec[precip_sec > 0]

if positive_precip.size > 0:
    precip_top = np.nanpercentile(positive_precip, 98)
    precip_top = max(0.05, nice_ceil(precip_top, 0.05))
    precip_levels = np.linspace(0.05, precip_top, 14)

    cf3 = ax3.contourf(
        X,
        Y,
        precip_masked,
        levels=precip_levels,
        cmap="YlGnBu",
        extend="max",
    )

    cbar3 = fig3.colorbar(cf3, ax=ax3, orientation="horizontal", pad=0.08, shrink=0.9)
    cbar3.set_label("降水粒子混合比 (g kg$^{-1}$)")
else:
    cf3 = None
    ax3.text(
        0.5, 0.92, "该剖面无降水粒子",
        transform=ax3.transAxes,
        ha="center", va="center",
        fontsize=12, color="gray"
    )

# -----------------------------
# 2. 等压线（保持不变）
# -----------------------------
p_levels = [100, 200, 300, 500, 700, 850, 900, 950, 1000]
valid_p_levels = [lv for lv in p_levels if np.nanmin(p_sec) <= lv <= np.nanmax(p_sec)]

cs3 = ax3.contour(
    X,
    Y,
    p_sec,
    levels=valid_p_levels,
    colors="k",
    linewidths=0.55,
)
ax3.clabel(cs3, fmt="%d", fontsize=8, inline=True)

# -----------------------------
# 3. 风速等值线（保持不变）
#    speed_sec 已经在前面算好
#    如果你前面用的是：
#    speed_sec = np.sqrt(v_sec**2 + w_sec**2)
#    那这里就直接沿用
# -----------------------------
spd_top = max(5.0, nice_ceil(np.nanpercentile(speed_sec, 98), 5.0))
spd_levels = np.arange(5.0, spd_top + 0.1, 5.0)

cs_spd = ax3.contour(
    X,
    Y,
    speed_sec,
    levels=spd_levels,
    colors="darkgreen",
    linewidths=0.8,
)
ax3.clabel(cs_spd, fmt="%.0f", fontsize=8, inline=True)

# -----------------------------
# 4. 地形（保持不变）
# -----------------------------
ax3.fill_between(
    sec_lat,
    0.0,
    terrain_sec / 1000.0,
    color="0.5",
    alpha=0.6,
    zorder=5,
)

# -----------------------------
# 5. 垂直面风场 v-w（保持不变）
# -----------------------------
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

# -----------------------------
# 6. 坐标轴设置（保持不变）
# -----------------------------
y_max_km = min(18.0, max(8.0, np.nanpercentile(z_sec, 99)))
ax3.set_ylim(0, y_max_km)

# 左北右南
ax3.set_xlim(np.nanmax(sec_lat), np.nanmin(sec_lat))

ax3.set_xlabel("纬度 (°)")
ax3.set_ylabel("高度 (km)")

secax = ax3.secondary_yaxis(
    "right",
    functions=(height_km_to_pressure_hpa, pressure_hpa_to_height_km),
)
secax.set_ylabel("近似气压 (hPa)")
secax.set_yticks([1000, 850, 700, 500, 300, 200, 100])

xticks = np.linspace(np.nanmax(sec_lat), np.nanmin(sec_lat), 6)
ax3.set_xticks(xticks)
ax3.set_xticklabels([f"{x:.1f}" for x in xticks])

ax3.text(
    0.02, 0.98, "(c)",
    transform=ax3.transAxes,
    ha="left", va="top",
    fontsize=13, fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
)

fig3_path = os.path.join(output_dir, f"{time_str}_panel_c_120E_precip_section.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig3)
print(f"已保存: {fig3_path}")