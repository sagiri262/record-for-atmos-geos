import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob
import cartopy
import cartopy.crs as ccrs
import numpy as np
import imageio
from netCDF4 import Dataset
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import ListedColormap
from meteva.base.tool.plot_tools import add_china_map_2basemap
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)


SCRIPT_DIR = Path(__file__).resolve().parent
WRF_RESULT_DIR = SCRIPT_DIR.parents[2] / "WRF_result" / "era_test"


def plot_plain(wrf_files, titles):
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    colors = ['red', 'green']    
    for i, (files, title) in enumerate(zip(wrf_files, titles)):
        lons, lats, slps = [], [], []
        for file in sorted(files):
            ncfile = Dataset(file)
            slp = getvar(ncfile, "slp")
            #smooth2d(field: Any, passes: Any, cenweight: float = 2, meta: bool = True)
            smooth_slp = smooth2d(slp, 3, cenweight=4)
            # 找到最低气压点
            min_idx = np.unravel_index(np.argmin(to_np(smooth_slp)), smooth_slp.shape)
            lon = to_np(ncfile['XLONG'][0,:,:])[min_idx]
            lat = to_np(ncfile['XLAT'][0,:,:])[min_idx]
            lons.append(lon)
            lats.append(lat)
            slps.append(to_np(smooth_slp)[min_idx])
        # 绘制路径
        ax.plot(lons, lats, color=colors[i], markers='o', linestyle='-', 
                linewidth=2, markersize=6, label=title, transform=ccrs.PlateCarree())
    """添加地图要素
    包括：海岸线、边境线、格网线、四至范围
    对应：coastlines  add_feature   gridlines   set_extent

    """
    ax.coastlines(resolution='10m')
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    # 四至范围
    ax.set_extent([111, 121, 36, 44])
    
    plt.title('Pressure Distribution')
    plt.legend()
    plt.savefig('test_colorbar.png', dpi=300, bbox_inches='tight')
    plt.close()

orig_files = sorted(glob.glob('wrfout_d02_2025-04-09*'))

plot_plain(orig_files, 'Pressure Distribution')


"""
沙尘数据四至范围：[111, 121, 36, 44]
测试数据四至范围：[110, 125, 21, 43]
[West, East, South, North]
"""
def setup_map(ax, extent):
    # 加入中国地图图层
    add_china_map_2basemap(ax, name="river", edgecolor='k', lw=0.5, encoding='gbk')
    add_china_map_2basemap(ax, name="nation", edgecolor='k', lw=0.5, encoding='gbk')
    add_china_map_2basemap(ax, name="province", edgecolor='k', lw=0.5, encoding='gbk')
    # 设置四至范围
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # 配置格网点
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='none', 
                      alpha=0.5, linestyle='--', x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # 旋转标签
    gl.rotate_labels = False
    # 东西向
    gl.xlocator = mticker.FixedLocator(np.arange(111, 125, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(21, 44, 2))
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    return ax


def read_plot_field(ncfile, var_name):
    """Return scalar data and optional wind vectors for plotting."""
    if not isinstance(var_name, str):
        raise TypeError("var_name must be a string, such as 'slp', 'R', or 'UV'.")

    name = var_name.lower()
    if name in ('r', 'rain', 'rainfall', 'precip'):
        data = (
            to_np(getvar(ncfile, "RAINC"))
            + to_np(getvar(ncfile, "RAINNC"))
            + to_np(getvar(ncfile, "RAINSH"))
        )
        return data, None, None, 'Rainfall(mm)'

    if name in ('uv', 'wind', 'uv10', 'wind10'):
        uv = to_np(getvar(ncfile, "uvmet10", units="m s-1"))
        u = uv[0, :, :]
        v = uv[1, :, :]
        data = np.hypot(u, v)
        return data, u, v, 'Wind Speed(m/s)'

    field = getvar(ncfile, var_name)
    if name == 'slp':
        field = smooth2d(field, 3, cenweight=4)
    return to_np(field), None, None, var_name


# 绘制 GIF 动图
def wrfout_gif(wrf_files, var_name, gif_name, extent, levels=None, plot_type='contourf', title='', frame_dir='frames', duration=0.5, vector_step=10):
    os.makedirs(frame_dir, exist_ok=True)
    wrf_files = sorted(wrf_files)
    if not wrf_files:
        raise FileNotFoundError(
            "No WRF output files found for GIF generation. "
            "Check the wrfout file path or glob pattern."
        )

    pics = []

    for i, file in enumerate(wrf_files):
        ncfile = Dataset(file)
        lon = np.array(ncfile['XLONG'])[0,:,:]
        lat = np.array(ncfile['XLAT'])[0,:,:]

        data, u, v, cbar_label = read_plot_field(ncfile, var_name)
        if data.ndim != 2:
            raise ValueError(f"{var_name} is {data.ndim}D. wrfout_gif currently expects a 2D field.")

        if levels is None:
            vmin = np.floor(np.nanmin(data) / 2) * 2
            vmax = np.ceil(np.nanmax(data) / 2) * 2
            use_levels = np.arange(vmin, vmax + 2, 2)
        else:
            use_levels = levels

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        if plot_type == 'contour':
            cs = ax.contour(lon, lat, data, levels=use_levels, colors='r', linewidths=0.4, transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
            if u is not None and v is not None:
                step = max(1, int(vector_step))
                ax.quiver(lon[::step, ::step], lat[::step, ::step], u[::step, ::step], v[::step, ::step],
                          transform=ccrs.PlateCarree(), color='k', scale=500, width=0.002)
        elif plot_type == 'barbs':
            cf = ax.contourf(lon, lat, data, levels=use_levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='max')
            cbar = fig.colorbar(cf, ax=ax, ticks=use_levels[::3], shrink=0.65)
            cbar.set_label(cbar_label, size=12)
            if u is None or v is None:
                raise ValueError("plot_type='barbs' requires a vector variable such as 'UV'.")
            step = max(1, int(vector_step))
            ax.barbs(lon[::step, ::step], lat[::step, ::step], u[::step, ::step], v[::step, ::step],
                     transform=ccrs.PlateCarree(), length=5, linewidth=0.4)
        else:
            cf = ax.contourf(lon, lat, data, levels=use_levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='max')
            cbar = fig.colorbar(cf, ax=ax, ticks=use_levels[::3], shrink=0.65)
            cbar.set_label(cbar_label, size=12)
            if u is not None and v is not None:
                step = max(1, int(vector_step))
                ax.quiver(lon[::step, ::step], lat[::step, ::step], u[::step, ::step], v[::step, ::step],
                          transform=ccrs.PlateCarree(), color='k', scale=500, width=0.002)

        ax = setup_map(ax, extent)
        plt.title(title if title else var_name, fontsize=14, pad=20)

        pic = os.path.join(frame_dir, f'{i:03d}_{var_name}.png')
        plt.savefig(pic, dpi=200, bbox_inches='tight')
        plt.close()
        pics.append(pic)
        ncfile.close()

    if not pics:
        raise RuntimeError("No frame images were generated; GIF creation was skipped.")

    imageio.mimsave(gif_name, [imageio.imread(pic) for pic in pics], duration=duration)


# 设置colorbar
rgb = ([237, 237, 237], [209, 209, 209], [173, 173, 173], [131, 131, 131], [93, 93, 93],
        [151, 198, 223], [111, 176, 214], [49, 129, 189], [26, 104, 174], [8, 79, 153],
        [62, 168, 91], [110, 193, 115], [154, 214, 149], [192, 230, 185], [223, 242, 217],
        [255, 255, 164], [255, 243, 0], [255, 183, 0], [255, 123, 0], [255, 62, 0],
        [255, 2, 0], [196, 0, 0], [136, 0, 0])
colors = np.array(rgb) / 255.
cmap = ListedColormap(colors)
# rain_levels = [0.1, 1, 2, 5, 7.5, 10, 13, 16, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250]
rain_levels = [0.1, 1, 2, 5, 7.5, 10, 13, 16, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
rain_levels = [value * 5 for value in rain_levels]

ncfile = Dataset(WRF_RESULT_DIR / 'wrfout_d01_2022-11-28_22:00:00')
lon = np.array(ncfile['XLONG'])[0,:,:]
lat = np.array(ncfile['XLAT'])[0,:,:]
# wrfout计算降水的方法
R = (to_np(getvar(ncfile, "RAINC")) + to_np(getvar(ncfile, "RAINNC")) + to_np(getvar(ncfile, "RAINSH")))
slp = getvar(ncfile, "slp")
smooth_slp = smooth2d(slp, 3, cenweight=4)

# 绘制第一幅子图，降水
fig1, ax1 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
cf1 = ax1.contourf(lon, lat, R, levels=rain_levels, cmap=cmap,
                    transform=ccrs.PlateCarree(), extend='max')
# 设置地图
ax1 = setup_map(ax1, [116, 124, 25, 35])
cbar1 = fig1.colorbar(cf1, ax=ax1, ticks=rain_levels[::3], shrink=0.65)
cbar1.set_label('Rainfall(mm)', size=12)
plt.title('Accumulated Rainfall', fontsize=14, pad=20)
plt.savefig('rainfall_plot.png', dpi=300, bbox_inches='tight')

# =========================
# 绘制海平面气压分布图 SLP
# =========================

slp_np = to_np(smooth_slp)
# 根据当前数据自动生成气压分级，间隔 2 hPa
slp_min = np.floor(np.nanmin(slp_np) / 2) * 2
slp_max = np.ceil(np.nanmax(slp_np) / 2) * 2
slp_levels = np.arange(slp_min, slp_max + 2, 2)
fig2, ax2 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# 填色气压图
"""cf2 = ax2.contourf(lon, lat, slp_np, levels=slp_levels, cmap='coolwarm',
                   transform=ccrs.PlateCarree(), extend='both')"""
# 叠加等压线
cs2 = ax2.contour(lon, lat, slp_np, levels=slp_levels, colors='r',
                  linewidths=0.4, transform=ccrs.PlateCarree())
# 标注等压线数值
ax2.clabel(cs2, inline=True, fontsize=8, fmt='%d')
# 地图底图、边界、经纬度
ax2 = setup_map(ax2, [110, 125, 21, 43])
# 色标
"""
cbar2 = fig2.colorbar(cf2, ax=ax2, ticks=slp_levels[::2], shrink=0.65)
cbar2.set_label('Sea Level Pressure (hPa)', size=12)
"""
plt.title('Sea Level Pressure', fontsize=14, pad=20)
plt.savefig('pressure_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 输出GIF图
wrf_files = sorted(glob.glob(str(WRF_RESULT_DIR / 'wrfout_d01_2022-11*')))
wrfout_gif(wrf_files, 'R', 'RainfallDistribute.gif', [116, 124, 25, 35], levels=rain_levels, title='Rainfall Distribution')
