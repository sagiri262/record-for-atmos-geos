import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from wrf_read_data import WRFDataReader
from netCDF4 import Dataset
from wrf import (
    to_np, getvar, get_cartopy,
    cartopy_xlim, cartopy_ylim, latlon_coords,
    CoordPair, vertcross
)

# =========================
# 1. 读取数据
# =========================
wrf_path = "/Volumes/Lexar/WRF_Data/wrfout_d01_*"

reader = WRFDataReader(wrf_path)
wrf_files = reader.get_files()

# 打开第一个文件示例
ncfile = Dataset(wrf_files[0])

# =========================
# 2. 读取水汽和高度变量
# =========================
# 三维水汽混合比，kg/kg -> g/kg
qvapor = getvar(ncfile, "QVAPOR") * 1000.0

# 几何高度，单位 km
z = getvar(ncfile, "z", units="km")

# 子图1：取最低模式层水汽，做水平分布
qvapor_low = qvapor[0, :, :]

# 经纬度与投影
lat, lon = latlon_coords(qvapor_low)
cart_proj = get_cartopy(qvapor_low)

# =========================
# 3. 计算 120E 经线垂直剖面
# =========================
# 先检查 120E 是否落在模拟域内
lon_min = float(np.nanmin(to_np(lon)))
lon_max = float(np.nanmax(to_np(lon)))

if not (lon_min <= 120.0 <= lon_max):
    raise ValueError(
        f"120E 不在当前模拟区域经度范围内，当前范围为 [{lon_min:.2f}, {lon_max:.2f}]"
    )

lat_min = float(np.nanmin(to_np(lat)))
lat_max = float(np.nanmax(to_np(lat)))

start_point = CoordPair(lat=lat_min, lon=120.0)
end_point   = CoordPair(lat=lat_max, lon=120.0)

qv_cross = vertcross(
    qvapor, z,
    wrfin=ncfile,
    start_point=start_point,
    end_point=end_point,
    latlon=True,
    autolevels=100,
    meta=True
)

# 垂直坐标（km）
z_cross = to_np(qv_cross.coords["vertical"])

# 垂直坐标最大值 = 文件最大高度 + 5 km
zmax_data = float(np.nanmax(to_np(z)))
zmax_plot = zmax_data + 5.0

# 剖面横坐标标签
xy_loc = to_np(qv_cross.coords["xy_loc"])
x_ticks = np.arange(len(xy_loc))
x_labels = [str(p) for p in xy_loc]

# 避免横坐标标签过密
step = max(1, len(x_ticks) // 8)
x_ticks_plot = x_ticks[::step]
x_labels_plot = x_labels[::step]

# =========================
# 4. 色阶设置
# =========================
# 按你的要求使用 RdBu
vmin = 0.0
vmax = max(
    float(np.nanmax(to_np(qvapor_low))),
    float(np.nanmax(to_np(qv_cross)))
)
levels = np.linspace(vmin, vmax, 21)

# =========================
# 5. 绘制两个子图
# =========================
fig = plt.figure(figsize=(16, 6))

# ---------- 子图1：水汽水平分布 ----------
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())

cf1 = ax1.contourf(
    to_np(lon), to_np(lat), to_np(qvapor_low),
    levels=levels,
    cmap="RdBu",
    transform=ccrs.PlateCarree()
)

cs1 = ax1.contour(
    to_np(lon), to_np(lat), to_np(qvapor_low),
    levels=levels[::2],
    colors="black",
    linewidths=0.4,
    transform=ccrs.PlateCarree()
)
ax1.clabel(cs1, inline=True, fontsize=7, fmt="%.1f")

# 不再添加 coastline/borders/NaturalEarthFeature
# 只保留范围、网格线和坐标
ax1.set_xlim([lon_min, lon_max])
ax1.set_ylim([float(np.nanmin(to_np(lat))), float(np.nanmax(to_np(lat)))])

gl = ax1.gridlines(
    draw_labels=True,
    color="gray",
    linestyle="dotted",
    linewidth=0.5
)
gl.right_labels = False
gl.top_labels = False
gl.x_inline = False

ax1.set_title("WRF水汽水平分布（最低模式层, g/kg）", fontsize=13)

cbar1 = plt.colorbar(cf1, ax=ax1, shrink=0.95, pad=0.03)
cbar1.set_label("Water Vapor Mixing Ratio (g/kg)")

# ---------- 子图2：120E 水汽垂直剖面 ----------
ax2 = fig.add_subplot(1, 2, 2)

cf2 = ax2.contourf(
    np.arange(qv_cross.shape[-1]),
    z_cross,
    to_np(qv_cross),
    levels=levels,
    cmap="RdBu"
)

cs2 = ax2.contour(
    np.arange(qv_cross.shape[-1]),
    z_cross,
    to_np(qv_cross),
    levels=levels[::2],
    colors="black",
    linewidths=0.4
)
ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.1f")

ax2.set_xticks(x_ticks_plot)
ax2.set_xticklabels(x_labels_plot, rotation=30, fontsize=8)
ax2.set_xlabel("120°E 经线上的纬度位置")
ax2.set_ylabel("Height (km)")
ax2.set_ylim(0, zmax_plot)

ax2.set_title("WRF水汽垂直剖面（120°E）", fontsize=13)

cbar2 = plt.colorbar(cf2, ax=ax2, shrink=0.95, pad=0.03)
cbar2.set_label("Water Vapor Mixing Ratio (g/kg)")

plt.tight_layout()
plt.savefig("wrf_qvapor_horizontal_and_120E_cross.png", dpi=300, bbox_inches="tight")
print("图已保存为 wrf_qvapor_horizontal_and_120E_cross.png")

ncfile.close()
reader.close()