'''
本代码的目标是实现：
1、实现风速大小的经向分布
2、实现底层地形剖面的实现

暂时如上
'''
import os
import re
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from wrf import (
    getvar, vertcross, interpline, CoordPair,
    to_np, latlon_coords
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

start_time = datetime.strptime("2022-11-28_00:00:00", "%Y-%m-%d_%H:%M:%S")
end_time   = datetime.strptime("2022-12-02_21:00:00", "%Y-%m-%d_%H:%M:%S")

target_lon = 120.0  # 目标经线

# 统一垂直插值层，单位 m
z_levels = np.arange(0, 20001, 250)
z_km = z_levels / 1000.0

# 等值线设置
ua_levels = np.arange(-40, 41, 5)         # 纬向风 m/s
theta_levels = np.arange(260, 421, 5)     # 位温 K
temp_levels = np.arange(-60, 31, 2)       # 温度 degC，填色


# =========================================================
# 2. 时间解析函数
# =========================================================
def parse_wrf_time_from_filename(fname):
    """
    支持：
    wrfout_d01_2022-11-30_18:00:00
    wrfout_d01_2022-11-30_18_00_00
    wrfout_d01_2022-11-30_180000
    """
    base = os.path.basename(fname)
    tstr = base.replace("wrfout_d01_", "")
    tstr = tstr.replace("\uf03a", ":")

    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})[:_](\d{2})[:_](\d{2})", tstr)
    if not m:
        raise ValueError(f"无法从文件名解析时间: {fname}")

    date_part, hh, mm, ss = m.groups()
    standard = f"{date_part}_{hh}:{mm}:{ss}"
    return datetime.strptime(standard, "%Y-%m-%d_%H:%M:%S")


# =========================================================
# 3. 获取文件并按时间筛选
# =========================================================
reader = WRFDataReader(wrf_path)
wrf_files = sorted(reader.get_files())

selected_files = []
for f in wrf_files:
    try:
        t = parse_wrf_time_from_filename(f)
        if start_time <= t <= end_time:
            selected_files.append(f)
    except Exception as e:
        print(f"跳过无法解析时间的文件: {f}, error={e}")

if not selected_files:
    raise FileNotFoundError("没有找到指定时间范围内的 wrfout 文件。")

print(f"选中的文件数: {len(selected_files)}")
for f in selected_files:
    print(os.path.basename(f))


# =========================================================
# 4. 用第一个文件自动确定最接近 120E 的剖面线
# =========================================================
nc0 = Dataset(selected_files[0])

ter0 = getvar(nc0, "ter", timeidx=0)
lats0, lons0 = latlon_coords(ter0)

lats0 = to_np(lats0)
lons0 = to_np(lons0)

# 找最接近 120E 的列
dist = np.abs(lons0 - target_lon)
j0, i0 = np.unravel_index(np.argmin(dist), dist.shape)

section_lon = float(np.nanmean(lons0[:, i0]))
lat_min = float(np.nanmin(lats0[:, i0]))
lat_max = float(np.nanmax(lats0[:, i0]))

start_point = CoordPair(lat=lat_min, lon=section_lon)
end_point   = CoordPair(lat=lat_max, lon=section_lon)

print(f"实际剖面经线: {section_lon:.3f}E")
print(f"纬度范围: {lat_min:.3f} ~ {lat_max:.3f}")

nc0.close()


# =========================================================
# 5. 循环做剖面并时间平均
# =========================================================
sum_temp = None
sum_ua = None
sum_theta = None

valid_temp_count = None
valid_ua_count = None
valid_theta_count = None

lat_vals = None
ter_km = None

for wrf_file in selected_files:
    print(f"处理: {os.path.basename(wrf_file)}")
    nc = Dataset(wrf_file)

    # 变量读取
    temp  = getvar(nc, "tc", timeidx=0)       # degC
    ua    = getvar(nc, "ua", timeidx=0)       # m/s
    theta = getvar(nc, "theta", timeidx=0)    # K
    z     = getvar(nc, "z", timeidx=0)        # m
    ter   = getvar(nc, "ter", timeidx=0)      # m

    # 在统一高度层上做剖面
    temp_cross = vertcross(
        temp, z,
        wrfin=nc,
        start_point=start_point,
        end_point=end_point,
        latlon=True,
        meta=True,
        levels=z_levels
    )

    ua_cross = vertcross(
        ua, z,
        wrfin=nc,
        start_point=start_point,
        end_point=end_point,
        latlon=True,
        meta=True,
        levels=z_levels
    )

    theta_cross = vertcross(
        theta, z,
        wrfin=nc,
        start_point=start_point,
        end_point=end_point,
        latlon=True,
        meta=True,
        levels=z_levels
    )

    # 地形只取一次
    if ter_km is None:
        ter_line = interpline(
            ter,
            wrfin=nc,
            start_point=start_point,
            end_point=end_point
        )
        ter_km = to_np(ter_line) / 1000.0

    # 横轴纬度只取一次
    if lat_vals is None:
        xy_locs = to_np(theta_cross.coords["xy_loc"])
        lat_vals = np.array([pt.lat for pt in xy_locs])

    # 转数组
    temp2d = np.asarray(to_np(temp_cross), dtype=float)
    ua2d = np.asarray(to_np(ua_cross), dtype=float)
    theta2d = np.asarray(to_np(theta_cross), dtype=float)

    temp2d[~np.isfinite(temp2d)] = np.nan
    ua2d[~np.isfinite(ua2d)] = np.nan
    theta2d[~np.isfinite(theta2d)] = np.nan

    # 初始化
    if sum_temp is None:
        sum_temp = np.zeros_like(temp2d)
        sum_ua = np.zeros_like(ua2d)
        sum_theta = np.zeros_like(theta2d)

        valid_temp_count = np.zeros_like(temp2d)
        valid_ua_count = np.zeros_like(ua2d)
        valid_theta_count = np.zeros_like(theta2d)

    # 累加
    temp_mask = np.isfinite(temp2d)
    ua_mask = np.isfinite(ua2d)
    theta_mask = np.isfinite(theta2d)

    sum_temp[temp_mask] += temp2d[temp_mask]
    sum_ua[ua_mask] += ua2d[ua_mask]
    sum_theta[theta_mask] += theta2d[theta_mask]

    valid_temp_count[temp_mask] += 1
    valid_ua_count[ua_mask] += 1
    valid_theta_count[theta_mask] += 1

    nc.close()

# 时间平均
temp_mean = np.where(valid_temp_count > 0, sum_temp / valid_temp_count, np.nan)
ua_mean = np.where(valid_ua_count > 0, sum_ua / valid_ua_count, np.nan)
theta_mean = np.where(valid_theta_count > 0, sum_theta / valid_theta_count, np.nan)


# =========================================================
# 6. 画图
# =========================================================
x2d, y2d = np.meshgrid(lat_vals, z_km)

fig, ax = plt.subplots(figsize=(12, 8))

# 填色：温度
cf = ax.contourf(
    x2d, y2d, temp_mean,
    levels=temp_levels,
    cmap="turbo",
    extend="both"
)

# 红线：纬向风
cs_ua = ax.contour(
    x2d, y2d, ua_mean,
    levels=ua_levels,
    colors="red",
    linewidths=1.0
)
ax.clabel(cs_ua, fmt="%d", fontsize=8)

# 蓝线：位温
cs_th = ax.contour(
    x2d, y2d, theta_mean,
    levels=theta_levels,
    colors="blue",
    linewidths=1.0
)
ax.clabel(cs_th, fmt="%d", fontsize=8)

# 地形
ax.fill_between(lat_vals, 0, ter_km, color="black", zorder=20)

# 坐标与标题
ax.set_xlabel("Latitude (deg)")
ax.set_ylabel("Height above geoid (km)")
ax.set_ylim(0, 20)

ax.set_title(
    "Time-mean meridional cross section near 120E\n"
    "2022-11-28 00:00:00 to 2022-12-02 21:00:00\n"
    "Temperature (shaded, degC), Zonal Wind (red, m/s), Potential Temperature (blue, K)"
)

# 色标
cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label("Temperature (degC)")

plt.tight_layout()
plt.show()