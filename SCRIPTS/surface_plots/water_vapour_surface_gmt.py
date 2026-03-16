import numpy as np
import xarray as xr
import pygmt
from scipy.interpolate import griddata
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
from wrf_read_data import WRFDataReader

# =========================
# 1. 读取 WRF 数据
# =========================
wrf_path = "/Volumes/Lexar/WRF_Data/WRF_second_try/wrfout_d01_*"
print("主脚本传入路径:", wrf_path)

reader = WRFDataReader(wrf_path)
wrf_files = reader.get_files()

print("reader 返回的首文件:", wrf_files[0] if wrf_files else "空")

print("匹配文件数:", len(wrf_files))
if not wrf_files:
    raise FileNotFoundError(f"没有匹配到任何文件：{wrf_path}")

print("首个文件:", wrf_files[0])

ncfile = Dataset(wrf_files[0])

try:
    # 水汽混合比 kg/kg -> g/kg
    qvapor = getvar(ncfile, "QVAPOR") * 1000.0

    # 最低模式层
    qvapor_sfc = qvapor[0, :, :]

    # 经纬度
    lat, lon = latlon_coords(qvapor_sfc)
    lon2d = to_np(lon)
    lat2d = to_np(lat)
    qv2d = to_np(qvapor_sfc)

    # =========================
    # 2. 插值到规则经纬网
    # =========================
    dlon = 0.05
    dlat = 0.05

    lon_min = float(np.nanmin(lon2d))
    lon_max = float(np.nanmax(lon2d))
    lat_min = float(np.nanmin(lat2d))
    lat_max = float(np.nanmax(lat2d))

    west = np.floor(lon_min / dlon) * dlon
    east = np.ceil(lon_max / dlon) * dlon
    south = np.floor(lat_min / dlat) * dlat
    north = np.ceil(lat_max / dlat) * dlat

    region = [west, east, south, north]

    lon_reg = np.arange(west, east + dlon * 0.5, dlon)
    lat_reg = np.arange(south, north + dlat * 0.5, dlat)
    lon_grid, lat_grid = np.meshgrid(lon_reg, lat_reg)

    points = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values = qv2d.ravel()

    qv_reg = griddata(
        points,
        values,
        (lon_grid, lat_grid),
        method="linear"
    )

    # 边界 NaN 用 nearest 补齐
    if np.any(np.isnan(qv_reg)):
        qv_reg_nearest = griddata(
            points,
            values,
            (lon_grid, lat_grid),
            method="nearest"
        )
        qv_reg = np.where(np.isnan(qv_reg), qv_reg_nearest, qv_reg)

    grid = xr.DataArray(
        qv_reg,
        coords={"lat": lat_reg, "lon": lon_reg},
        dims=("lat", "lon"),
        name="qvapor"
    )

    if np.all(np.isnan(grid.values)):
        raise ValueError("插值结果全为 NaN，请检查经纬度和原始数据。")

    # =========================
    # 3. 绘图
    # =========================
    fig = pygmt.Figure()

    vmax = float(np.nanmax(grid.values))
    vmin = float(np.nanmin(grid.values))

    if np.isnan(vmax) or vmax <= 0:
        raise ValueError("水汽场最大值异常，请检查 QVAPOR 数据。")

    cint = max(0.5, round((vmax - vmin) / 20, 2))
    if cint <= 0:
        cint = 0.5

    pygmt.makecpt(cmap="polar", series=[vmin, vmax, cint])

    with pygmt.config(
        MAP_FRAME_TYPE="plain",
        MAP_FRAME_PEN="1p,black",
        FORMAT_GEO_MAP="dddF",
    ):
        fig.grdimage(
            grid=grid,
            region=region,
            projection="M16c",
            cmap=True,
            nan_transparent=True,
        )

        fig.coast(
            region=region,
            projection="M16c",
            water="white",
            shorelines="0.8p,black",
            resolution="i",
        )

        fig.basemap(
            region=region,
            projection="M16c",
            frame=[
                "xaf",
                "yaf",
                "WSen+tWRF Water Vapor Horizontal Distribution",
            ],
        )

        fig.colorbar(
            position="JMR+w8c/0.5c+o0.8c/0c",
            frame=['xaf+l"Water Vapor Mixing Ratio (g/kg)"']
        )

    output_file = "wrf_qvapor_horizontal_landonly.png"
    fig.savefig(output_file, dpi=300)
    print(f"图已保存为 {output_file}")

finally:
    ncfile.close()
    reader.close()