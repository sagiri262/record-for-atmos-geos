# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import cfgrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime

FILE    = "/home/zy/record-for-atmos-geos/coldwave/datasets/data.grib"
FIG_DIR = "/home/zy/record-for-atmos-geos/coldwave/figs"
os.makedirs(FIG_DIR, exist_ok=True)
OUT     = os.path.join(FIG_DIR, "coldwave_mslp_wind_t2m.png")

t0 = datetime(2025, 10, 17, 0)   # UTC

# ---------- helpers ----------
def find_dataset_with_var_and_time(grib_file, short_name, var_candidates):
    """
    在 cfgrib.open_datasets 拆分后的子数据集中，找到包含 var_candidates 中任一变量
    且具有时间轴 (time/valid_time/forecastTime 或 time+step) 的那个，并返回：
      ds, varname, time_coord_name
    """
    dsets = cfgrib.open_datasets(
        grib_file,
        backend_kwargs={"filter_by_keys": {"shortName": short_name}}
    )

    for ds in dsets:
        # 找变量名
        varname = next((v for v in var_candidates if v in ds.variables), None)
        if varname is None:
            continue

        # 找到时间坐标名
        if "time" in ds.coords:
            tcoord = "time"
        elif "valid_time" in ds.coords:
            tcoord = "valid_time"
        elif "forecastTime" in ds.coords:
            tcoord = "forecastTime"
        elif "time" in ds and "step" in ds.coords:
            # 构造有效时间并挂到坐标上
            ds = ds.assign_coords(valid_time=(ds["time"] + ds["step"]))
            tcoord = "valid_time"
        else:
            # 无法识别时间轴
            continue

        return ds, varname, tcoord

    raise RuntimeError(
        f"Cannot find dataset for shortName='{short_name}' with vars {var_candidates} "
        f"that also has a time-like coordinate."
    )

# ---------- open all needed fields robustly ----------
# 2 m temperature (K) -> °C
ds_t2m, var_t2m, tcoord_t2m = find_dataset_with_var_and_time(FILE, "2t", ["t2m", "2t"])
# mean sea level pressure (Pa) -> hPa
ds_msl, var_msl, tcoord_msl = find_dataset_with_var_and_time(FILE, "msl", ["msl"])
# 10 m wind
ds_u10, var_u10, tcoord_u10 = find_dataset_with_var_and_time(FILE, "10u", ["10u", "u10"])
ds_v10, var_v10, tcoord_v10 = find_dataset_with_var_and_time(FILE, "10v", ["10v", "v10"])

# 以 t2m 的时间轴为准，选离 t0 最近的时刻
t_vals = ds_t2m.coords[tcoord_t2m].values
itime  = int(np.argmin(np.abs(t_vals - np.datetime64(t0))))
t_sel  = ds_t2m.coords[tcoord_t2m][itime]
print("Using time:", np.datetime_as_string(t_sel.values if hasattr(t_sel, "values") else t_sel, unit="h"))

# 对齐各场到同一时刻
t2m = (ds_t2m[var_t2m].sel({tcoord_t2m: t_sel}, method="nearest") - 273.15)   # °C
msl =  (ds_msl[var_msl].sel({tcoord_msl:  t_sel}, method="nearest") / 100.0) # hPa
u10 =   ds_u10[var_u10].sel({tcoord_u10: t_sel}, method="nearest")
v10 =   ds_v10[var_v10].sel({tcoord_v10: t_sel}, method="nearest")

# 网格
lon = t2m.longitude.values
lat = t2m.latitude.values
Lon, Lat = np.meshgrid(lon, lat)

# ---------- plot ----------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 若终端报出 killed，表明内存不够用，可先改 dpi 试跑
# 假设 Lon, Lat 和 t2m 已经定义好，并且你正在使用 Cartopy 绘图
fig = plt.figure(figsize=(20, 15), dpi=600)  
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=proj)

extent = [60, 150, 0, 70]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# 国际通用海岸线和国界
ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.5)
ax.coastlines(resolution="50m", linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black")

# 底图：2 m 温度（配色参考示例，采用 rainbow；范围 -35~45°C）
levels_t = np.arange(-35, 46, 1)
cf = ax.contourf(Lon, Lat, t2m.values, levels=levels_t,
                 cmap="rainbow", extend="neither", transform=ccrs.PlateCarree())  # 去掉箭头

# 创建 colorbar，调整 fraction 和 pad 参数
cbar = plt.colorbar(cf, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

# 设置 colorbar 标签
cbar.set_label("2 m Temperature (°C)", fontsize=14)

# 调整 colorbar 高度与图框一致
# 获取当前图框的纵向尺寸
height_inch = fig.get_figheight()  # 获取图形高度 (英寸)
fig_height_ratio = height_inch / ax.get_position().height  # 获取图框纵向比例

# 设置 colorbar 高度与图框纵向一致
cbar.ax.set_aspect(fig_height_ratio)  # 使 colorbar 高度与图框高度相同

# 等压线：从 1000 hPa 起每 5 hPa；每 15 hPa 红线并标注
levels_all = np.arange(1000, 1101, 2)
levels_red = np.arange(1000, 1101, 15)
cs = ax.contour(Lon, Lat, msl.values, levels=levels_all,
                colors="grey", linewidths=0.9, transform=ccrs.PlateCarree())
ax.clabel(cs, fmt="%d hPa", inline=True, fontsize=10, colors="grey")
csr = ax.contour(Lon, Lat, msl.values, levels=levels_red,
                 colors="red", linewidths=1.4, transform=ccrs.PlateCarree())
ax.clabel(csr, fmt="%d hPa", inline=True, fontsize=12, colors="red")

# 风矢量：白色
skip = (slice(None, None, 8), slice(None, None, 8))
ax.quiver(Lon[skip], Lat[skip], u10.values[skip], v10.values[skip],
          color="black", transform=ccrs.PlateCarree(), scale=550, width=0.002)

# 设置经纬度标注
xticks = np.arange(int(extent[0]/5)*5, extent[1]+1, 5)
yticks = np.arange(int(extent[2]/5)*5, extent[3]+1, 5)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True)
ax.tick_params(axis="y", right=False, left=True, labelleft=True)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(45);  lbl.set_fontsize(8)
for lbl in ax.get_yticklabels():
    lbl.set_rotation(-45); lbl.set_fontsize(8)

# 设置标题
ax.set_title(
    f"2 m Temperature (shaded), MSLP (contours) & 10 m Wind — {t0:%Y-%m-%d %H:%M} UTC",
    fontsize=18
)

# 保存图像
plt.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Saved to: {OUT}")