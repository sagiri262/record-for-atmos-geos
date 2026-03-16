import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from netCDF4 import Dataset


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

# 打开第一个文件示例
ncfile = Dataset(wrf_files[0])

'''把所有文件都打开成 Dataset 对象
ncfile = [Dataset(f) for f in wrf_files]'''

# 设置物理参数
g = 9.8
# 单位 Pa
p0 = 100000.0
R_cp = 287.0 / 1004.0

timeidx = 0
iy, ix = 50, 80

# 取单点垂直列
PH  = ncfile.variables["PH"][timeidx, :, iy, ix]
PHB = ncfile.variables["PHB"][timeidx, :, iy, ix]
P   = ncfile.variables["P"][timeidx, :, iy, ix]
PB  = ncfile.variables["PB"][timeidx, :, iy, ix] 
T   = ncfile.variables["T"][timeidx, :, iy, ix]

z_stag = (PH + PHB) / g

# 插值到质量层
"""
逻辑：去掉最后一个元素的数组 + 去掉第一个元素的数组

把两者相加再乘 0.5，就是相邻两个层界高度的平均值：

大致类似于：
0.5 * ([z0, z1, z2, z3] + [z1, z2, z3, z4])
=
[(z0+z1)/2, (z1+z2)/2, (z2+z3)/2, (z3+z4)/2]
"""
# / 1000 转换到 km
z = 0.5 * (z_stag[:-1] + z_stag[1:]) / 1000

# 计算总压强
pres = P + PB

# 计算绝对位温
theta = T + 300.0

# 绝对温度
tk = theta * (pres / p0) ** R_cp

# 摄氏度
tc = tk - 273.15


plt.figure(figsize=(8, 6))
plt.plot(tc, z, marker="o")
plt.xlabel("Temprature")
plt.ylabel("Height")
plt.title("WRF vertical temperature profile")
plt.grid(True)
plt.show()

