import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import matplotlib
from matplotlib.ticker import FixedLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ——— 中文字体设置 ———
# 请确认系统中安装了该字体，或替换为系统已有的中文字体名称
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# ——— 参数设置 ———
file_path  = "/home/zy/record-for-atmos-geos/surface_pressure/typhoon/ragasa.grib"
target_bjt = datetime(2025,9,24,17)
target_utc = target_bjt - timedelta(hours=8)
output_png = "plot_20250924_17BJT.png"

# ——— 读取数据 ———
ds  = xr.open_dataset(file_path, engine="cfgrib")
print(ds)
sst = ds["sst"]
u10 = ds["u10"]
v10 = ds["v10"]

# 获取最近时间索引
time_vals = sst.time.values
time_idx  = int(np.argmin(np.abs(time_vals - np.datetime64(target_utc))))
print("Using index", time_idx, "for UTC time", target_utc)

# 掩膜：保留海洋区域
sea_mask = ~np.isnan(sst.isel(time=time_idx))

# ——— 绘图 ———
fig = plt.figure(figsize=(14,6))

# 设置投影为 Albers 等面积
proj = ccrs.AlbersEqualArea(central_longitude=110, central_latitude=25,
                             standard_parallels=(20,40))

# 设置经纬度范围（你可根据实际区域修改）
extent = [90, 140, 0, 50]  # 经度从90°E到140°E，纬度从0°到50°N
# 左图
ax1 = fig.add_subplot(1,2,1, projection=proj)
ax1.set_extent(extent, crs=ccrs.PlateCarree())
ax1.coastlines(resolution='50m', color='black')
ax1.add_feature(cfeature.LAND, facecolor='lightgray')
ax1.set_title(f"BJT {target_bjt.strftime('%Y-%m-d %H:%M')} 海温＋风场")

# 加网格线与标注
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    xlocs=np.arange(extent[0], extent[1]+1, 20),
                    ylocs=np.arange(extent[2], extent[3]+1, 15),
                    linewidth=0.5, color='gray', linestyle='--')
gl1.top_labels    = False
gl1.right_labels  = False
gl1.xformatter    = LongitudeFormatter()
gl1.yformatter    = LatitudeFormatter()
gl1.xlabel_style  = {'size':10}
gl1.ylabel_style  = {'size':10}

lon = sst.longitude.values
lat = sst.latitude .values
Lon, Lat = np.meshgrid(lon, lat)

data_sst = sst.isel(time=time_idx).values
data_sst = np.where(sea_mask, data_sst, np.nan)
cf1 = ax1.contourf(Lon, Lat, data_sst,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', levels=20, extend='both')
u_vec = u10.isel(time=time_idx).values
v_vec = v10.isel(time=time_idx).values
skip  = (slice(None, None, 10), slice(None, None, 10))
ax1.quiver(Lon[skip], Lat[skip], u_vec[skip], v_vec[skip],
           transform=ccrs.PlateCarree(), color='k', scale=400)

# 右图
ax2 = fig.add_subplot(1,2,2, projection=proj)
ax2.set_extent(extent, crs=ccrs.PlateCarree())
ax2.coastlines(resolution='50m', color='black')
ax2.add_feature(cfeature.LAND, facecolor='lightgray')
ax2.set_title(f"UTC {target_utc.strftime('%Y-%m-d %H:%M')} 风速＋风场")

gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    xlocs=np.arange(extent[0], extent[1]+1, 20),
                    ylocs=np.arange(extent[2], extent[3]+1, 15),
                    linewidth=0.5, color='gray', linestyle='--')
gl2.top_labels    = False
gl2.right_labels  = False
gl2.xformatter    = LongitudeFormatter()
gl2.yformatter    = LatitudeFormatter()
gl2.xlabel_style  = {'size':10}
gl2.ylabel_style  = {'size':10}

wind_speed = np.sqrt(u_vec**2 + v_vec**2)
ws_data    = np.where(sea_mask, wind_speed, np.nan)
cf2 = ax2.contourf(Lon, Lat, ws_data,
                   transform=ccrs.PlateCarree(),
                   cmap='viridis', levels=20, extend='both')
ax2.quiver(Lon[skip], Lat[skip], u_vec[skip], v_vec[skip],
           transform=ccrs.PlateCarree(), color='k', scale=400)

# 添加 colorbars
cbar1 = fig.colorbar(cf1, ax=ax1, orientation='vertical', pad=0.02, shrink=0.8)
cbar1.set_label('海温 (单位)')
cbar2 = fig.colorbar(cf2, ax=ax2, orientation='vertical', pad=0.02, shrink=0.8)
cbar2.set_label('风速 (m/s)')

plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.show()
print(f"Saved plot to {output_png}")
