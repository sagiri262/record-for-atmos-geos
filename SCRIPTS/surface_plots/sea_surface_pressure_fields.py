import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs
import glob
import cartopy.feature as cfeature
from wrf_read_data import WRFDataReader
from netCDF4 import Dataset
from cartopy.feature import NaturalEarthFeature
from pathlib import Path
from typing import List, Union, Optional
from wrf import (
    to_np, getvar, smooth2d, get_cartopy,
    cartopy_xlim, cartopy_ylim, latlon_coords
)



'''root = Path('/home/zy/WRF/WRFV4.6.0/test/em_real')

wrf_files_dir = root / 
'''
# 批量读取文件
# wrf_files = sorted(glob.glob("wrf_"))


'''
通用 WRF 结果数据读取工具
使用场景包括
1、只有一个 wrfout 文件
2、逻辑合并读取一连串的 wrfout 文件
3、列出文件的变量、维度、时间信息等必要信息
4、使用 xarray 提取基本变量
5、可选提取 WRF 诊断变量 

说明：
    - 默认使用 xarray.open_mfdataset 逻辑合并
    - 对 WRF 多文件，默认按文件顺序沿 Time 维拼接
    - 不写死路径，适合嵌入其他脚本
'''

# =========================
# 1. 用读取模块统一管理文件
# =========================
# wrf_path = "/home/zy/WRF/WRFV4.6.0/test/em_real/wrfout_d01_*"

wrf_path = "/Volumes/Lexar/WRF_Data/wrfout_d01_*"

reader = WRFDataReader(wrf_path)

# 获取排序后的文件列表
wrf_files = reader.get_files()

# 打开第一个文件示例
ncfile = Dataset(wrf_files[0])


# =========================
# 2. 读取变量
# =========================
slp = getvar(ncfile, "slp")
smooth_slp = smooth2d(slp, 3, cenweight=4)

# 获取经纬度和投影
lats, lons = latlon_coords(slp)
cart_proj = get_cartopy(slp)


# =========================
# 3. 绘图
# =========================
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=cart_proj)

try:
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, edgecolor="black")
except Exception:
    pass

# ax.add_feature(states, linewidth=0.5, edgecolor="black")
ax.coastlines(resolution="50m", linewidth=0.8)

contours = plt.contour(
    to_np(lons), to_np(lats), to_np(smooth_slp),
    10, colors="black", transform=crs.PlateCarree()
)

filled = plt.contourf(
    to_np(lons), to_np(lats), to_np(smooth_slp),
    10, transform=crs.PlateCarree(), cmap="jet"
    # get_map 方法已被弃用
)

plt.colorbar(filled, ax=ax, shrink=0.98, label="Sea Level Pressure (hPa)")

ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))

gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
gl.right_labels = False
gl.top_labels = False
gl.x_inline = False

plt.title("Sea Level Pressure (hPa)", fontsize=14)
plt.savefig("slp_map.png", dpi=300, bbox_inches="tight")
plt.show()

ncfile.close()
reader.close()