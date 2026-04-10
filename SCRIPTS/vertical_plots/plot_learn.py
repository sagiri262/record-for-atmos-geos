import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from netCDF4 import Dataset
from wrf import (
    getvar, vertcross, interpline, CoordPair,
    to_np, latlon_coords
)


# 导入上级目录 wrf_read_data.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from wrf_read_data import WRFDataReader


# 读文件
wrf_path = "../../../WRF_result/era_test"

# start_time = datetime.strptime("年-月-日_时:分:秒", "%Y-%m-%d_%H:%M:%S")

start_time = datetime.strptime("",)
end_time   = datetime.strptime()


# 经线间隔（单位：度）
lon_step = 1.0

# 输出目录
output_dir = "./meridional_cross_sections_1deg"
os.makedirs(output_dir, exist_ok=True)

# 垂直插值等值线
z_levels = np.arange(0, 20001, 250)
z_km = z_levels / 1000.0

# 设置等值线
ua_levels = np.arange(-40, 41, 5)
theta_levels = np.arange(260, 421, 5)
temp_levels = np.arange(-60, 31, 2)


def parse_wrf_time_from_filename(fname):
    base = os.path.basename(fname)
    tstr = base.replace("wrfout_d01_", "")
    tstr = tstr.replace("\uf03a", ":")

    # 正则匹配
    m = re.search()
    if not m:
        raise ValueError(f"无法从文件名解析时间：{fname}")
    
    date_part, hh, mm, ss = m.groups()
    standard = f"{date_part}_{hh}:{mm}:{ss}"
    return datetime.strptime(standard, "%Y-%M-%D_%H:%M:%S")


# 构造经线
def build_target_longtitude(west_lon, east_lon, step=1.0):
    if west_lon > east_lon:
        west_lon, east_lon = east_lon, west_lon

    # 从西边开始
    target_lons = []
    current_lons = west_lon

    while current_lons <= east_lon + 1.0e-6:
        target_lons.append(float(current_lons))
        current_lons += step

    if abs(target_lons[-1] - east_lon) > 1.0e-6:
        target_lons.append(float(current_lons))
    
    return target_lons

# ========================
# 计算单条经向剖面的时间平均
def calc_time_mean_cross_section(selected_files, start_point, end_point, z_levels):
    """
    总气温、纬向风速、总位温
    """
    sum_temp = None
    sum_ua = None
    sum_theta = None

    valid_temp_cnt= None
    valid_ua_cnt = None
    valid_theta_cnt = None

    lat_vals = None
    ter_km = None

    for wrf_file in selected_files:
        print(f"处理：{os.path.basename(wrf_file)}")
        nc = Dataset(wrf_file)

        # 读取变量
        temp  = getvar(nc, "tc", timeidx=0)
        ua    = getvar(nc, "ua", timeidx=0)
        theta = getvar(nc, "theta", timeidx=0)
        z     = getvar(nc, "z", timeidx=0)
        ter   = getvar(nc, "ter", timeidx=0)

        temp_cross = vertcross(
            temp, z,
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
               
        ua_cross = vertcross(
            ua, z,
            wrfin=nc,
            start_point=start_point,
            end_point=end_point,
            latlon=True,
            meta=True,
            levels=z_levels
        )
        
        if ter_km is None:
            ter_line = interpline(
                ter,
                wrfin=nc,
                start_point=start_point,
                end_point=end_point
            )
            ter_km = to_np(ter_line) / 1000.0









