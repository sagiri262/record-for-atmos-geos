import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
from datetime import datetime, timedelta

# ——— 参数设置 ———
file_path = "/home/zy/record-for-atmos-geos/surface_pressure/typhoon/ragasa.grib"
start_time = datetime(2025,9,17,0)
end_time   = datetime(2025,9,26,23)
time_interval = timedelta(hours=1)  # 若数据为小时间隔
output_gif = "ragasa_wind_sst_20250917_26.gif"

# ——— 读取 GRIB 文件 ———
ds = xr.open_dataset(file_path, engine="cfgrib")
print(ds)  # 查看有哪些变量，例如 'sst', 'u10', 'v10', 'msl' 等

# 假设变量名分别为 'sst'、'u10'、'v10'、'msl'
sst  = ds["sst"]
u10  = ds["u10"]
v10  = ds["v10"]
msl = ds["msl"]

# 限定时间区间
ds_sel = ds.sel(time=slice(start_time, end_time))
sst   = ds_sel["sst"]
u10   = ds_sel["u10"]
v10   = ds_sel["v10"]
msl  = ds_sel["msl"]

# 掩膜：只保留海洋
sea_mask = ~np.isnan(sst.isel(time=0))

# ——— 绘图函数 ———
def plot_frame(ax, idx):
    t = sst.time.values[idx]
    ax.clear()
    ax.set_global()
    ax.coastlines(resolution='50m', color='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    lon = sst.longitude.values
    lat = sst.latitude .values
    Lon, Lat = np.meshgrid(lon, lat)
    
    # 底图：用气压 msl (颜色从蓝→红 表示气压从低到高)
    data_pr = msl.isel(time=idx).values
    data_pr = np.where(sea_mask, data_pr, np.nan)
    cf = ax.contourf(Lon, Lat, data_pr,
                     transform=ccrs.PlateCarree(),
                     cmap='coolwarm', levels=20, extend='both', zorder=1)
    
    # 风箭头
    u_vec = u10.isel(time=idx).values
    v_vec = v10.isel(time=idx).values
    skip = (slice(None, None, 10), slice(None, None, 10))
    ax.quiver(Lon[skip], Lat[skip], u_vec[skip], v_vec[skip],
              transform=ccrs.PlateCarree(), color='k', zorder=2, scale=400)
    ax.set_title(f"UTC {np.datetime_as_string(t, unit='h')}")
    return cf

# ——— 创建动画 ———
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
times = range(len(sst.time))
anim  = animation.FuncAnimation(fig, lambda i: plot_frame(ax,i),
                                frames=times, interval=200)
anim.save(output_gif, writer='pillow', dpi=150)
print(f"Saved GIF to {output_gif}")
