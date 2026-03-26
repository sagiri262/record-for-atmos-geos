# -*- coding: utf-8 -*-
"""
基于所有 wrfout_d01_* 文件，计算整个模拟时段的时间平均 120E 经向剖面图：

1. 背景填色：时间平均降水粒子混合比 (QRAIN + QSNOW + QGRAUP)
2. 黑色细线：时间平均等压线
3. 绿色等值线：时间平均风速等值线
4. 黑色箭头：时间平均 v-w 垂直面风场
5. 灰色填充：地形

输出：
- 一张大图，表示 wrf 模拟时段的平均剖面
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np

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

lon_section = 120.0

output_dir = os.path.join(current_dir, "wrf_precip_section_all_times")
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 3. 工具函数
# =========================================================
def nice_ceil(value, step):
    if step == 0:
        return value
    return np.ceil(value / step) * step


def height_km_to_pressure_hpa(z_km):
    z_m = np.asarray(z_km) * 1000.0
    p = np.empty_like(z_m, dtype=float)

    mask1 = z_m <= 11000.0
    p[mask1] = 1013.25 * (1.0 - 2.25577e-5 * z_m[mask1]) ** 5.25588

    mask2 = ~mask1
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
    """
    沿固定经度 lon_target 提取近似经向剖面：
    在每个 south_north 行上，选取最接近目标经度的格点。
    """
    lons_np = np.asarray(lons2d)
    lats_np = np.asarray(lats2d)

    lon_min = np.nanmin(lons_np)
    lon_max = np.nanmax(lons_np)

    if not (lon_min <= lon_target <= lon_max):
        raise ValueError(
            f"指定剖面经度 {lon_target}E 不在当前区域经度范围内: [{lon_min:.2f}, {lon_max:.2f}]"
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

    # 左北右南
    order = np.argsort(sec_lat)[::-1]
    sec_lat = sec_lat[order]
    sec_lon = sec_lon[order]
    terrain_sec = terrain_sec[order]

    for key in vars_sec:
        vars_sec[key] = vars_sec[key][:, order]

    return sec_lat, sec_lon, terrain_sec, vars_sec


# =========================================================
# 4. 循环所有 wrfout，提取剖面并累加
# =========================================================
precip_list = []
p_list = []
v_list = []
w_list = []
z_list = []

sec_lat_ref = None
terrain_ref = None
used_precip_vars_ref = None

print(f"共找到 {len(wrf_files)} 个 wrfout 文件")
print(f"剖面经度: {lon_section}E")

for idx, target_file in enumerate(wrf_files, start=1):
    time_str = os.path.basename(target_file).replace("wrfout_d01_", "")
    print(f"\n[{idx}/{len(wrf_files)}] 处理: {time_str}")

    ncfile = Dataset(target_file)

    # 基础变量
    pressure = getvar(ncfile, "pressure")      # hPa
    z = getvar(ncfile, "z", units="m")         # m
    va = getvar(ncfile, "va", units="m s-1")   # m/s
    wa = getvar(ncfile, "wa", units="m s-1")   # m/s
    ter = getvar(ncfile, "ter")                # m

    slp = getvar(ncfile, "slp")
    lats, lons = latlon_coords(slp)

    lats_np = to_np(lats)
    lons_np = to_np(lons)

    pres_np = np.asarray(to_np(pressure), dtype=np.float64)
    z_np = np.asarray(to_np(z), dtype=np.float64)
    va_np = np.asarray(to_np(va), dtype=np.float64)
    wa_np = np.asarray(to_np(wa), dtype=np.float64)
    ter_np = np.asarray(to_np(ter), dtype=np.float64)

    # 三维降水粒子变量
    precip_candidates = ["QRAIN", "QSNOW", "QGRAUP"]
    used_precip_vars = [v for v in precip_candidates if v in ncfile.variables]

    if len(used_precip_vars) == 0:
        ncfile.close()
        raise RuntimeError(
            f"{time_str}: 当前 wrfout 中没有找到 QRAIN/QSNOW/QGRAUP。"
        )

    if used_precip_vars_ref is None:
        used_precip_vars_ref = used_precip_vars

    precip3d = None
    for vname in used_precip_vars:
        arr = np.asarray(ncfile.variables[vname][0, :, :, :], dtype=np.float64)  # kg/kg
        if precip3d is None:
            precip3d = arr.copy()
        else:
            precip3d += arr

    # 提取经向剖面
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

    z_sec = vars_sec["z"] / 1000.0          # km
    p_sec = vars_sec["p"]                   # hPa
    v_sec = vars_sec["v"]                   # m/s
    w_sec = vars_sec["w"]                   # m/s
    precip_sec = vars_sec["precip"] * 1e3   # g/kg

    if sec_lat_ref is None:
        sec_lat_ref = sec_lat.copy()
        terrain_ref = terrain_sec.copy()
        ref_shape = precip_sec.shape
    else:
        if precip_sec.shape != ref_shape:
            ncfile.close()
            raise ValueError(
                f"{time_str}: 剖面形状与前一个文件不一致，"
                f"当前 {precip_sec.shape}，参考 {ref_shape}"
            )

    precip_list.append(precip_sec)
    p_list.append(p_sec)
    v_list.append(v_sec)
    w_list.append(w_sec)
    z_list.append(z_sec)

    ncfile.close()

print("\n所有 wrfout 文件剖面提取完成。")


# =========================================================
# 5. 计算时间平均
# =========================================================
precip_mean = np.nanmean(np.stack(precip_list, axis=0), axis=0)
p_mean = np.nanmean(np.stack(p_list, axis=0), axis=0)
v_mean = np.nanmean(np.stack(v_list, axis=0), axis=0)
w_mean = np.nanmean(np.stack(w_list, axis=0), axis=0)
z_mean = np.nanmean(np.stack(z_list, axis=0), axis=0)

# 风速等值线（保持之前逻辑）
speed_mean = np.sqrt(v_mean**2 + w_mean**2)

# 只画 > 0 的降水粒子
precip_mean_masked = np.ma.masked_less_equal(precip_mean, 0.0)

# 统计量
valid_precip = precip_mean[precip_mean > 0]

print("\n时间平均剖面降水粒子统计（QRAIN+QSNOW+QGRAUP）：")
if valid_precip.size > 0:
    print(f"最大值: {np.nanmax(valid_precip):.6f} g/kg")
    print(f"最小值: {np.nanmin(valid_precip):.6f} g/kg")
    print(f"平均值: {np.nanmean(valid_precip):.6f} g/kg")
else:
    print("时间平均后整个剖面降水粒子都为 0。")

print("\n时间平均剖面风速统计：")
print(f"最大值: {np.nanmax(speed_mean):.3f} m/s")
print(f"最小值: {np.nanmin(speed_mean):.3f} m/s")
print(f"平均值: {np.nanmean(speed_mean):.3f} m/s")


# =========================================================
# 6. 绘制时间平均大图
# =========================================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

X = np.tile(sec_lat_ref[None, :], (z_mean.shape[0], 1))
Y = z_mean

# 1) 背景：时间平均降水粒子混合比
positive_precip = precip_mean[precip_mean > 0]

if positive_precip.size > 0:
    precip_top = np.nanpercentile(positive_precip, 98)
    precip_top = max(0.05, nice_ceil(precip_top, 0.05))
    precip_levels = np.linspace(0.05, precip_top, 14)

    cf = ax.contourf(
        X,
        Y,
        precip_mean_masked,
        levels=precip_levels,
        cmap="YlGnBu",
        extend="max",
    )

    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, shrink=0.9)
    cbar.set_label("时间平均降水粒子混合比 (g kg$^{-1}$)")
else:
    ax.text(
        0.5, 0.92, "时间平均后该剖面无降水粒子",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12, color="gray"
    )

# 2) 时间平均等压线
p_levels = [100, 200, 300, 500, 700, 850, 900, 950, 1000]
valid_p_levels = [lv for lv in p_levels if np.nanmin(p_mean) <= lv <= np.nanmax(p_mean)]

cs_p = ax.contour(
    X,
    Y,
    p_mean,
    levels=valid_p_levels,
    colors="k",
    linewidths=0.55,
)
ax.clabel(cs_p, fmt="%d", fontsize=8, inline=True)

# 3) 时间平均风速等值线
spd_top = max(5.0, nice_ceil(np.nanpercentile(speed_mean, 98), 5.0))
spd_levels = np.arange(5.0, spd_top + 0.1, 5.0)

cs_spd = ax.contour(
    X,
    Y,
    speed_mean,
    levels=spd_levels,
    colors="darkgreen",
    linewidths=0.8,
)
ax.clabel(cs_spd, fmt="%.0f", fontsize=8, inline=True)

# 4) 地形
ax.fill_between(
    sec_lat_ref,
    0.0,
    terrain_ref / 1000.0,
    color="0.5",
    alpha=0.6,
    zorder=5,
)

# 5) 时间平均垂直面风场
w_display_factor = 50.0
skipx = max(1, int(sec_lat_ref.size / 24))
skipz = max(1, int(z_mean.shape[0] / 22))

q = ax.quiver(
    X[::skipz, ::skipx],
    Y[::skipz, ::skipx],
    v_mean[::skipz, ::skipx],
    w_mean[::skipz, ::skipx] * w_display_factor,
    angles="xy",
    scale_units="xy",
    scale=100,
    width=0.0022,
    color="k",
    zorder=6,
)

ax.quiverkey(
    q,
    X=0.98,
    Y=1.03,
    U=10,
    label="平均 v = 10 m/s，平均 w 显示放大 ×50",
    labelpos="E",
    coordinates="axes",
    fontproperties={"size": 9},
)

# 6) 坐标轴
y_max_km = min(18.0, max(8.0, np.nanpercentile(z_mean, 99)))
ax.set_ylim(0, y_max_km)

# 左北右南
ax.set_xlim(np.nanmax(sec_lat_ref), np.nanmin(sec_lat_ref))

ax.set_xlabel("纬度 (°)")
ax.set_ylabel("高度 (km)")

secax = ax.secondary_yaxis(
    "right",
    functions=(height_km_to_pressure_hpa, pressure_hpa_to_height_km),
)
secax.set_ylabel("近似气压 (hPa)")
secax.set_yticks([1000, 850, 700, 500, 300, 200, 100])

xticks = np.linspace(np.nanmax(sec_lat_ref), np.nanmin(sec_lat_ref), 6)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:.1f}" for x in xticks])

# 不加大标题，只写说明
ax.text(
    0.02, 0.98,
    "(c) 模拟时段平均剖面",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=13, fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
)

fig.text(
    0.5,
    0.01,
    f"平均时段：{os.path.basename(wrf_files[0]).replace('wrfout_d01_', '')}  ~  "
    f"{os.path.basename(wrf_files[-1]).replace('wrfout_d01_', '')}",
    ha="center",
    va="bottom",
    fontsize=12,
)

out_path = os.path.join(output_dir, "time_mean_panel_c_120E_precip_section.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n已保存时间平均大图: {out_path}")
print(f"降水粒子变量使用: {used_precip_vars_ref}")
print("全部完成。")