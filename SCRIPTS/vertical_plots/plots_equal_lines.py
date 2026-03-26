# -*- coding: utf-8 -*-
"""
从指定 wrfout 文件中读取某一时刻、某一格点的垂直廓线，
绘制 Skew-T / log-P 图，并显示：
1. 等压线
2. 等温线
3. 干绝热线
4. 湿绝热线
5. 等饱和比湿线（等混合比线）

横坐标：温度（°C）
纵坐标：气压（hPa，对数坐标）
"""

import os
import sys
import numpy as np

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy, to_np

from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# macOS 上优先尝试的中文字体
candidates = [
    "Hiragino Sans GB",
    "PingFang SC",
    "STHeiti",
    "Arial Unicode MS",
]

installed = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((name for name in candidates if name in installed), None)

if font_name is None:
    raise RuntimeError(
        "没有找到可用的中文字体。请先安装一个中文字体，例如 Noto Sans CJK SC。"
    )

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [font_name]
mpl.rcParams["axes.unicode_minus"] = False  # 顺手避免负号乱码

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


# =========================================================
# 2. 选择一个垂直廓线位置
# =========================================================
# 方法1：指定经纬度找最近格点
use_latlon = False

if use_latlon:
    lat0 = 30.0
    lon0 = 114.0
    xy = ll_to_xy(ncfile, lat0, lon0)
    i = int(to_np(xy[0]))
    j = int(to_np(xy[1]))
else:
    # 默认取区域中心点
    lats = getvar(ncfile, "lat")
    lons = getvar(ncfile, "lon")
    ny, nx = lats.shape
    j = ny // 2
    i = nx // 2

# 该格点经纬度
lats = getvar(ncfile, "lat")
lons = getvar(ncfile, "lon")
lat_pt = float(to_np(lats[j, i]))
lon_pt = float(to_np(lons[j, i]))

print(f"选取格点: i={i}, j={j}, lat={lat_pt:.3f}, lon={lon_pt:.3f}")


# =========================================================
# 3. 读取变量
# =========================================================
# pressure: hPa
# tc      : 摄氏温度
# 横坐标
# 纵坐标
pressure = getvar(ncfile, "pressure")
tc = getvar(ncfile, "tc")

# 比湿
# QVAPOR: kg/kg
qvapor = ncfile.variables["QVAPOR"][0, :, :, :]

# 垂直高度
height_agl = getvar(ncfile, "height_agl", units="m")

# 风速廓线
uvmet_wspd_wdir = getvar(ncfile, "uvmet_wspd_wdir", units="m s-1")

# =========================================================
# 4. 提取单点垂直廓线
# =========================================================
p_prof = np.asarray(to_np(pressure[:, j, i]), dtype=np.float64)  # hPa
t_prof = np.asarray(to_np(tc[:, j, i]), dtype=np.float64)        # degC
w_prof = np.asarray(qvapor[:, j, i], dtype=np.float64)           # kg/kg
z_prof = np.asarray(to_np(height_agl[:, j, i]), dtype=np.float64)

# m/s
ws_prof = np.asarray(to_np(uvmet_wspd_wdir[0, :, j ,i]), dtype=np.float64) 
# deg
wd_prof = np.asarray(to_np(uvmet_wspd_wdir[1, :, j, i]), dtype=np.float64)

# 过滤无效值
mask = np.isfinite(p_prof) & np.isfinite(t_prof) & np.isfinite(w_prof) & np.isfinite(ws_prof) & np.isfinite(wd_prof) & np.isfinite(z_prof)
p_prof = p_prof[mask]
t_prof = t_prof[mask]
w_prof = w_prof[mask]
z_prof = z_prof[mask]
ws_prof = ws_prof[mask]
wd_prof = wd_prof[mask]

# 去掉非正气压
mask2 = p_prof > 0
p_prof = p_prof[mask2]
t_prof = t_prof[mask2]
w_prof = w_prof[mask2]

# 按气压从大到小排序（地面 -> 高空）
sort_idx = np.argsort(p_prof)[::-1]
p_prof = p_prof[sort_idx]
t_prof = t_prof[sort_idx]
w_prof = w_prof[sort_idx]
z_prof = z_prof[sort_idx]
ws_prof = ws_prof[sort_idx]
wd_prof = wd_prof[sort_idx]


# =========================================================
# 5. 用混合比计算露点
# =========================================================
# QVAPOR 通常就是水汽混合比 r (kg/kg)
# e = r * p / (epsilon + r)
epsilon = 0.622
e_prof = (w_prof * p_prof) / (epsilon + w_prof)   # hPa
e_prof = np.asarray(e_prof, dtype=np.float64)

# 显式转成带单位的 Quantity，避免 masked array/xarray 单位识别问题
td_prof = mpcalc.dewpoint(units.Quantity(e_prof, "hPa")).to("degC").m


# =========================================================
# 6. 转为带单位的数组
# =========================================================
p = units.Quantity(p_prof, "hPa")
T = units.Quantity(t_prof, "degC")
Td = units.Quantity(td_prof, "degC")


 # =========================================================
# 7. 绘图
# =========================================================
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# 设置范围
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 50)

# -----------------------------
# 1. 先画背景热力线
# -----------------------------
# 等压线（水平网格线）
skew.ax.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)

# 等温线方向上的参考网格
skew.ax.grid(True, which='major', axis='x', linestyle='--', color='lightgray', alpha=0.35)

# 干绝热线
skew.plot_dry_adiabats(color='orange', alpha=0.7, linewidth=0.8)

# 湿绝热线
skew.plot_moist_adiabats(color='blue', alpha=0.7, linewidth=0.8)

# 等混合比线（常对应你说的等饱和比湿线）
skew.plot_mixing_lines(color='green', alpha=0.7, linewidth=0.8)

# 0°C 等温线
zero_line = skew.ax.axvline(0, color='cyan', linestyle='--', linewidth=1.2)

# -----------------------------
# 2. 再画实际廓线
# -----------------------------
temp_line, = skew.plot(p, T, color='red', linewidth=2.2, label='环境温度')
dew_line,  = skew.plot(p, Td, color='purple', linewidth=2.2, label='露点温度')

# -----------------------------
# 3. 标题与坐标
# -----------------------------
time_str = target_basename.replace("wrfout_d01_", "")
skew.ax.set_title(
    f"WRF Skew-T / log-P\n"
    f"Time: {time_str}    Grid: (i={i}, j={j})    "
    f"Lat/Lon=({lat_pt:.3f}, {lon_pt:.3f})",
    fontsize=12
)

skew.ax.set_xlabel("Temperature (°C)")
skew.ax.set_ylabel("Pressure (hPa)")

# -----------------------------
# 4. 手动构造图例
# -----------------------------
legend_elements = [
    Line2D([0], [0], color='red', lw=2.2, label='环境温度'),
    Line2D([0], [0], color='purple', lw=2.2, label='露点温度'),
    Line2D([0], [0], color='orange', lw=1.2, label='干绝热线'),
    Line2D([0], [0], color='blue', lw=1.2, label='湿绝热线'),
    Line2D([0], [0], color='green', lw=1.2, label='等饱和比湿线/等混合比线'),
    Line2D([0], [0], color='gray', lw=1.2, linestyle='--', label='等压线'),
    Line2D([0], [0], color='lightgray', lw=1.2, linestyle='--', label='等温线参考网格'),
    Line2D([0], [0], color='cyan', lw=1.2, linestyle='--', label='0°C等温线'),
]

skew.ax.legend(
    handles=legend_elements,
    loc='best',
    fontsize=10,
    frameon=True
)

# 保存
out_png = f"../FIGS/skewt.png"
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"图已保存: {out_png}")

plt.show()


# 单独绘制风速垂直廓线

fig, ax = plt.subplots(figsize=(7, 9))

ax.plot(ws_prof, z_prof, color="tab:red", linewidth=2.2, marker="o", markersize=4)

ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Height AGL (m)")
ax.set_title(
    f"WRF 风速垂直廓线（高度坐标）\n"
    # f"Time: {time_str}    Grid: (i={i}, j={j})    "
    # f"Lat/Lon=({lat_pt:.3f}, {lon_pt:.3f})"
)

ax.grid(True, linestyle="--", alpha=0.4)

out_png = "../FIGS/wind_profile_height.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"图已保存: {out_png}")
plt.show()