import os
import re
import glob
import numpy as np
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime
from wrf_read_data import WRFDataReader


def resolve_input_path(path: str) -> str:
    """
    统一解析输入路径，支持：
    1. 绝对路径
    2. 相对当前工作目录的路径
    3. 相对当前脚本目录的路径

    对于包含通配符的路径，也会按同样逻辑解析。
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path 必须是 str 或 PathLike，当前类型为: {type(path)}")

    raw_path = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)

    # 先按当前工作目录解析
    cwd_candidate = os.path.normpath(os.path.abspath(raw_path))
    if glob.glob(cwd_candidate):
        return cwd_candidate

    # 再按当前脚本目录解析
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_candidate = os.path.normpath(os.path.join(script_dir, raw_path))
    if glob.glob(script_candidate):
        return script_candidate

    # 都未匹配到时，默认返回工作目录解析结果，交由下游抛出更明确报错
    return cwd_candidate


# 时间解析函数
def parse_time_from_filename(filepath: str) -> str:
    """
    时间格式： YYYY-MM-DD HH:MM UTC
    """
    basename = os.path.basename(filepath)
    # 正则匹配文件名
    m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2})-(\d{2}:\d{2}:\d{2})", basename)

    if m:
        date_part = m.group(1)
        time_part = m.group(2)
        dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%M-%D %H:%M:%S")
        return dt.strftime("%Y-%M-%D %H:%M:%S")

    return "Unknown Time"


def open_first_ncfile(wrf_path: str) -> tuple:
    """
    解析 WRF_path

    """
    resolved_path = resolve_input_path(wrf_path)
    reader = WRFDataReader(resolved_path)
    first_file = reader.get_files()[0]
    nc = Dataset(first_file, "r")
    return nc, first_file


def open_all_ncfiles(wrf_path: str) -> tuple:
    """
    使用 WRFDataReader 解析 wrf_path，返回 (Dataset列表, 路径列表)。
    调用方负责逐一 nc.close()。    
    """

    resolved_path = resolve_input_path(wrf_path)
    reader = WRFDataReader(resolved_path)
    files = reader.get_files()
    ncs = [Dataset(f, "r") for f in files]
    return files, ncs

"""
读取基础场数据
"""

# 返回经纬度
# lat, lon, shape (nx, ny)
def get_latlon(nc: Dataset):
    return nc.variables["XLAT"][0], nc.variables["XLONG"][0]


def get_pressure_3d(nc: Dataset) -> np.ndarray:
    # WRF 总气压 = 扰动气压 P + 基态气压 PB（单位：Pa）
    return nc.variables["P"][0] + nc.variables["PB"][0]


# 位势高度
def get_geopotential_height(nc: Dataset) -> np.ndarray:
    # PH/PHB 定义在 w 层（staggered），插值到质量层后除以 g。
    ph = nc.variables["PH"][0]
    phb = nc.variables["PHB"][0]
    geo = ph + phb 
    # (nz, nx, ny)
    geo_mass = 0.5 * (geo[:-1] + geo[1:])    
    return geo_mass / 9.81


def get_temperature_3d(nc: Dataset, press_3d: np.ndarray) -> np.ndarray:
    """
    WRF 的温度 = 扰动位温 (theta - 300k)
    """
    theta = nc.variables["T"][0] + 300.0
    pres_hpa = press_3d / 100.0
    t_k = theta * (pres_hpa / 1000.0) ** 0.2854
    return t_k - 273.15


# (dz, dy, dx)
def destagger_u(u: np.ndarray) -> np.ndarray:
    # U 在 x 方向错格，插值到质量格点
    return 0.5 * (u[:, :, :-1] + u[:, :, 1:])

# V 方向
def destagger_v(v: np.ndarray) -> np.ndarray:
    # V 在 y 方向错格，插值到质量格点
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def destagger_w(w: np.ndarray) -> np.ndarray:
    # W 在 z 方向错格，插值到质量格点
    return 0.5 * (w[:-1, :, :] + w[1:, :, :])


# 气压插值
def interp_to_pressure_fast(field_3d: np.ndarray, pres3d: np.ndarray,
                            target_pa: float) -> np.ndarray:
    """
    将 field3d (nz,ny,nx) 线性插值到 target_pa（Pa）等压面，返回 (ny,nx)。
    pres3d[0] > pres3d[-1]（底层气压最高）。
    """
    nz, ny, nx = field_3d.shape
    above    = pres3d < target_pa
    idx_up   = np.argmax(above, axis=0)
    idx_lo   = idx_up - 1
    valid    = (idx_up > 0) & (idx_up < nz)
 
    iy, ix   = np.mgrid[0:ny, 0:nx]
    idx_lo_s = np.where(valid, idx_lo, 0)
    idx_up_s = np.where(valid, idx_up, 0)
 
    p_lo = pres3d[idx_lo_s, iy, ix]
    p_up = pres3d[idx_up_s, iy, ix]
    f_lo = field_3d[idx_lo_s, iy, ix]
    f_up = field_3d[idx_up_s, iy, ix]
 
    dp      = p_up - p_lo
    safe_dp = np.where(dp == 0, 1, dp)
    w       = (target_pa - p_lo) / safe_dp
    result  = f_lo + w * (f_up - f_lo)
    return np.where(valid, result, np.nan)


# 绘制底图
"""
make_map_axes(图, 定位, 四至范围)
"""
def make_map_axes(fig, pos, extent):
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(pos, projection=proj)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.6, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), 
                   linewidth=0.5, edgecolor="#333333", linestyle="-", zorder=5)
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#f0f0f0", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#d6eaf8", zorder=0)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}
    gl.xlocator = mticker.MaxNLocator(5)
    gl.ylocator = mticker.MaxNLocator(5)

    return ax


def add_common_args(parser):
    """
    向 ArgumentParser 添加所有脚本共用的参数：
      --file   wrfout 路径（支持通配符，需用引号括住）
      --out    输出图片路径（含文件名）
      --dpi    图像分辨率
    """
    parser.add_argument(
        "--file", required=True,
        metavar="PATH",
        help='wrfout 文件路径，支持绝对路径/相对路径和通配符，例如："wrfout_d01_*"（建议加引号）'
    )
    parser.add_argument(
        "--out", default=None,
        metavar="PATH",
        help="输出图片路径（含文件名），不填则使用脚本默认名称"
    )
    parser.add_argument(
        "--dpi", default=150,
        metavar="N",
        help="输出图片 DPI 默认150"
    )

    return parser
