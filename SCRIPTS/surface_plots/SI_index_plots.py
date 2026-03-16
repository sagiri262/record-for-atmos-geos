# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
左图：WRF 原始范围上的气压场
右图：杭州附近 [118E, 122E, 28.5N, 31.5N] 的 Showalter Index (SI)

SI = T500 - T850_lifted
"""

import numpy as np
import xarray as xr
import pygmt
import metpy.calc as mpcalc

from netCDF4 import Dataset
from metpy.units import units
from metpy.interpolate import log_interpolate_1d

from wrf import getvar, interplevel, latlon_coords, to_np
from wrf_read_data import WRFDataReader


# ============================================================
# 1. 读 WRF 文件
# ------------------------------------------------------------
# 你原来这里：
#   ncfile = Dataset(wrf_files[0])
# 但后面代码却一直按 xarray.Dataset 的 d0["变量名"] 方式在用
# 所以这里我保留 netCDF4.Dataset 给 wrf-python 用，
# 同时新增 ds = xr.open_dataset(...) 给你后面 xarray 逻辑用
# ============================================================
wrf_path = "/Volumes/Lexar/WRF_Data/WRF_second_try/wrfout_d01_*"
reader = WRFDataReader(wrf_path)
wrf_files = reader.get_files()

# wrf-python 常用这个对象
ncfile = Dataset(wrf_files[0])

# xarray 数据集，供你后面 d0[...] / dims / isel 使用
ds = xr.open_dataset(wrf_files[0])


# ============================================================
# 2. 时间层选择
# ------------------------------------------------------------
# 你原代码这里直接用了 ds，但 ds 原先没定义
# 现在前面已经补了 ds = xr.open_dataset(...)
# ============================================================
if "Time" in ds.dims:
    d0 = ds.isel(Time=0)
elif "time" in ds.dims:
    d0 = ds.isel(time=0)
else:
    d0 = ds


# ============================================================
# 3. 取经纬度
# ------------------------------------------------------------
# 对 WRF 原始输出，最稳妥的方法不是硬猜 XLONG/XLAT，
# 而是直接用 wrf-python 的 latlon_coords
#
# getvar(ncfile, "slp", timeidx=0) 返回的是 DataArray，
# 可以直接拿它来取经纬度坐标
# ============================================================
slp_tmp = getvar(ncfile, "slp", timeidx=0)  # 先临时取一个 2D 场来拿经纬度
lat2d, lon2d = latlon_coords(slp_tmp)

# 转成 xarray.DataArray，方便后面统一写法
lat2d = xr.DataArray(to_np(lat2d), dims=("south_north", "west_east"))
lon2d = xr.DataArray(to_np(lon2d), dims=("south_north", "west_east"))


# ============================================================
# 4. 左图区域：WRF 原始四至范围
# ------------------------------------------------------------
# region_left = [west, east, south, north]
# 单位：度（经纬度）
# ============================================================
west = float(np.nanmin(lon2d.values))
east = float(np.nanmax(lon2d.values))
south = float(np.nanmin(lat2d.values))
north = float(np.nanmax(lat2d.values))

region_left = [west, east, south, north]


# ============================================================
# 5. 右图区域：杭州附近固定范围
# ------------------------------------------------------------
# 你要求：
#   28.5N ～ 31.5N
#   118E  ～ 122E
# 中心：
#   30N, 120E
# ============================================================
region_right = [118.0, 122.0, 28.5, 31.5]


# ============================================================
# 6. 左图气压场变量
# ------------------------------------------------------------
# 你原来是：
#   if "slp" in d0: ...
# 但 WRF 原始输出通常不会直接把 slp 存成普通变量给你
# 用 wrf-python 的 getvar("slp") 最稳
#
# 返回单位通常已经是 hPa
# ============================================================
p_left_hpa = getvar(ncfile, "slp", timeidx=0)

# 转 numpy，后面画图更直接
p_left_hpa_np = to_np(p_left_hpa)


# ============================================================
# 7. 右图只裁剪小区域后再计算 SI（推荐）
# ------------------------------------------------------------
# 这里保留你的思路：先裁剪杭州小区域，再算 SI
# 这样比全域都做 parcel profile 更省时间
# ============================================================
mask_right = (
    (lon2d >= region_right[0]) & (lon2d <= region_right[1]) &
    (lat2d >= region_right[2]) & (lat2d <= region_right[3])
)

jj, ii = np.where(mask_right.values)
j0, j1 = jj.min(), jj.max()
i0, i1 = ii.min(), ii.max()

lon_r = lon2d.isel(south_north=slice(j0, j1 + 1), west_east=slice(i0, i1 + 1))
lat_r = lat2d.isel(south_north=slice(j0, j1 + 1), west_east=slice(i0, i1 + 1))


# ============================================================
# 8. 取三维气压和温度
# ------------------------------------------------------------
# 你原来假定 pressure/tc/tk 已经在 d0 里
# 对 WRF 原始输出，更稳妥的是：
#   pressure = getvar(ncfile, "pressure", timeidx=0)
#   tc       = getvar(ncfile, "tc", timeidx=0)
#
# pressure 单位通常是 hPa
# tc 单位通常是 degC
# ============================================================
pres3d_hpa = getvar(ncfile, "pressure", timeidx=0)   # 3D, 单位 hPa
temp3d_c = getvar(ncfile, "tc", timeidx=0)           # 3D, 单位 degC

# 裁到右图小区域
pres3d_hpa = pres3d_hpa.isel(south_north=slice(j0, j1 + 1), west_east=slice(i0, i1 + 1))
temp3d_c = temp3d_c.isel(south_north=slice(j0, j1 + 1), west_east=slice(i0, i1 + 1))


# ============================================================
# 9. 计算右图 SI
# ------------------------------------------------------------
# 思路：
# 1) 每个格点取垂直廓线
# 2) 插值出环境温度 T850 / T500
# 3) 850 hPa 气块绝热抬升到 500 hPa
# 4) SI = T500 - T850_lifted
#
# 注意：
# - wrf-python 返回的 3D 变量通常维度是 (bottom_top, south_north, west_east)
# - 所以这里默认 z 在第 0 维
# ============================================================
p = np.ma.filled(to_np(pres3d_hpa), np.nan) * units.hPa
t = np.ma.filled(to_np(temp3d_c), np.nan) * units.degC

nz, ny, nx = p.shape
si = np.full((ny, nx), np.nan, dtype=float)

for j in range(ny):
    for i in range(nx):
        p_prof = p[:, j, i]
        t_prof = t[:, j, i]

        # 取出无单位数值部分做缺测判断
        p_vals = np.asarray(p_prof.magnitude)
        t_vals = np.asarray(t_prof.magnitude)

        good = np.isfinite(p_prof.magnitude) & np.isfinite(t_prof.magnitude)
        if good.sum() < 3:
            continue

        p1 = p_prof[good]
        t1 = t_prof[good]

        # pressure 从大到小排序（近地面 -> 高空）
        order = np.argsort(p1.magnitude)[::-1]
        p1 = p1[order]
        t1 = t1[order]

        # 必须覆盖 850 和 500 hPa
        if not (np.nanmax(p1.magnitude) >= 850 and np.nanmin(p1.magnitude) <= 500):
            continue

        try:
            # 环境场 T850 / T500
            T850 = log_interpolate_1d(850 * units.hPa, p1, t1)
            T500 = log_interpolate_1d(500 * units.hPa, p1, t1)

            # 这里只是占位近似：Td850 = T850 - 2°C
            # 正式业务图建议换成真实 Td850
            Td850 = T850 - 2.0 * units.degC

            # 850 hPa 气块抬升廓线
            prof = mpcalc.parcel_profile(p1, T850.to("degC"), Td850.to("degC"))

            # 抬升到 500 hPa 的 parcel 温度
            T850_lifted = log_interpolate_1d(500 * units.hPa, p1, prof)

            si[j, i] = (T500 - T850_lifted).to("degC").magnitude

        except Exception:
            continue


# ============================================================
# 10. 做成规则经纬网（给 PyGMT 最方便）
# ------------------------------------------------------------
# 你原来用 xarray.interp，但那要求底层接近规则网格
# WRF 常常是曲线网格，这样直接 interp 不够稳
#
# 这里我保留最简单的办法：先做 DataArray，再插值
# 如果你发现格点扭曲明显，下一步就该换 scipy.griddata
# ============================================================
si_da = xr.DataArray(
    si,
    coords={
        "south_north": lat_r[:, 0].values,
        "west_east": lon_r[0, :].values,
    },
    dims=("south_north", "west_east"),
    name="SI",
)

lon_new = np.arange(region_right[0], region_right[1] + 0.001, 0.05)  # 0.05° 网格
lat_new = np.arange(region_right[2], region_right[3] + 0.001, 0.05)

si_reg = si_da.interp(
    west_east=lon_new,
    south_north=lat_new,
    method="linear"
).rename({"west_east": "lon", "south_north": "lat"})


# ============================================================
# 11. 左图气压场转规则网格
# ------------------------------------------------------------
# 你原来是直接用 xarray.interp
# 这里同样保留你的写法，但说明一下：
#   这只适合 lon2d[:,0] / lat2d[0,:] 基本近似规则时
# 真正严格的 WRF 曲线网格，建议后面改成 scipy.griddata
# ============================================================
lon_new_left = np.linspace(west, east, 300)
lat_new_left = np.linspace(south, north, 240)

try:
    p_left_da = xr.DataArray(
        p_left_hpa_np,
        coords={
            "south_north": lat2d[:, 0].values,
            "west_east": lon2d[0, :].values,
        },
        dims=("south_north", "west_east"),
        name="Pressure_hPa",
    )

    p_left_reg = p_left_da.interp(
        west_east=lon_new_left,
        south_north=lat_new_left,
        method="linear"
    ).rename({"west_east": "lon", "south_north": "lat"})

except Exception:
    p_left_reg = None
    print("警告：左图没自动转成规则网格，请后续改成 scipy.griddata 重网格")


# ============================================================
# 12. PyGMT 全局风格
# ------------------------------------------------------------
# 这些参数就是你前面 GMT 那套边框风格的 PyGMT 写法：
#
# MAP_FRAME_TYPE="plain"
#   plain = 普通边框
#
# FORMAT_GEO_MAP="ddd:mm:ssF"
#   经纬度显示格式：度:分:秒 + 方向字母
#
# MAP_GRID_CROSS_SIZE_PRIMARY="0c"
#   0c 表示不画格网交叉小十字
#
# FONT_ANNOT_PRIMARY="8p"
#   坐标注记字号 8 points
#
# MAP_FRAME_AXES="WSEN"
#   四条边框都画
#
# 关于 subplot：
# PyGMT 官方 subplot 接口支持用 subsize 控制子图大小；text 的 offset 也支持用
# dx/dy 直接微调标题位置。:contentReference[oaicite:0]{index=0}
# ============================================================
pygmt.config(
    MAP_FRAME_TYPE="plain",
    FORMAT_GEO_MAP="ddd:mm:ssF",
    MAP_GRID_CROSS_SIZE_PRIMARY="0c",
    FONT_ANNOT_PRIMARY="8p",
    MAP_ANNOT_OBLIQUE="anywhere",
    MAP_FRAME_AXES="WSEN",
)


# ============================================================
# 13. 开始画图
# ------------------------------------------------------------
# 这里我改掉了你原来那个不太稳的：
#   subsize=[("10c","8c"), ("6c","8c")]
#
# 更稳一点的做法：
# - subplot 统一用一个高度，比如 8c
# - 左图 projection="M10c"
# - 右图 projection="M6c"
#
# 这样两个面板上沿样式一致，但地图宽度分别由投影宽度控制
# ============================================================
fig = pygmt.Figure()

with fig.subplot(
    nrows=1,
    ncols=2,
    subsize=("10c", "8c"),        # 每个 panel 的“框架高度”统一按 10c x 8c 处理
    margins=["0.5c", "0.4c"],     # 单位 c=厘米；可自己改
    frame=False,
):

    # ========================================================
    # 13.1 左图：原始 WRF 范围气压场
    # --------------------------------------------------------
    # projection="M10c"
    #   M = Mercator
    #   10c = 地图宽度 10 cm
    #
    # frame=["xa2f1g1", "ya2f1g1", "WSEN"]
    #   a = 主注记间隔（单位：度）
    #   f = 次刻度间隔（单位：度）
    #   g = 格网线间隔（单位：度）
    #
    # 例如 xa2f1g1 表示：
    #   x 方向主注记每 2°
    #   次刻度每 1°
    #   内部经线格网每 1°
    # ========================================================
    with fig.set_panel(panel=0):
        if p_left_reg is not None:
            pygmt.makecpt(cmap="vik", series=[980, 1040, 2], continuous=True)

            fig.grdimage(
                grid=p_left_reg,
                region=region_left,
                projection="M10c",
                cmap=True,
            )

            fig.grdcontour(
                grid=p_left_reg,
                region=region_left,
                projection="M10c",
                levels=2,              # 等值线间隔 2 hPa
                annotation=4,          # 每 4 hPa 标一次
                pen="0.5p,black",
            )

        fig.coast(
            region=region_left,
            projection="M10c",
            shorelines="1/0.5p,black",
            land="gray90",
            water="white",
            frame=["xa2f1g1", "ya2f1g1", "WSEN"],
        )

        # 标题位置：
        # position="TC" = 当前面板顶部中央
        # offset="0c/0.20c" = 向上 0.20 cm
        #
        # 如果你想“往下一点”，改成：
        #   offset="0c/0.05c"
        #   offset="0c/0c"
        #   offset="0c/-0.10c"
        fig.text(
            position="TC",
            text="(a) Pressure field over the original WRF domain",
            font="10p,Helvetica-Bold,black",
            justify="CB",
            offset="0c/0.20c",
            no_clip=True,
        )

    # ========================================================
    # 13.2 右图：杭州附近 SI
    # --------------------------------------------------------
    # projection="M6c"
    #   地图宽度 6 cm
    #
    # frame=["xa1f0.5g0.5", "ya1f0.5g0.5", "WSEN"]
    #   主注记每 1°
    #   次刻度每 0.5°
    #   内部经纬网每 0.5°
    # ========================================================
    with fig.set_panel(panel=1):
        pygmt.makecpt(cmap="roma", series=[-6, 10, 1], continuous=True)

        fig.grdimage(
            grid=si_reg,
            region=region_right,
            projection="M6c",
            cmap=True,
        )

        fig.grdcontour(
            grid=si_reg,
            region=region_right,
            projection="M6c",
            levels=1,              # 等值线间隔 1 ℃
            annotation=2,          # 每 2 ℃ 标一次
            pen="0.35p,black",
        )

        fig.coast(
            region=region_right,
            projection="M6c",
            shorelines="1/0.5p,black",
            land="gray92",
            water="white",
            frame=["xa1f0.5g0.5", "ya1f0.5g0.5", "WSEN"],
        )

        # 杭州中心点
        fig.plot(
            x=[120.0],
            y=[30.0],
            style="c0.12c",         # 圆点直径 0.12 cm
            fill="red",
            pen="0.3p,black",
        )

        fig.text(
            x=120.0,
            y=30.0,
            text="Hangzhou",
            font="8p,Helvetica-Bold,black",
            justify="LM",
            offset="0.12c/0c",      # 向右偏 0.12 cm
            fill="white@40",
        )

        fig.text(
            position="TC",
            text="(b) Showalter Index around Hangzhou",
            font="10p,Helvetica-Bold,black",
            justify="CB",
            offset="0c/0.20c",
            no_clip=True,
        )


# ============================================================
# 14. 色标
# ------------------------------------------------------------
# 你原来是两个公共色标，我保留
# 位置参数：
#   JBC = 相对整张图底部中央锚定
#   +w  = 宽/高
#   +o  = 偏移量
#   +h  = 水平色标
# ============================================================
fig.colorbar(
    position="JBC+w6c/0.25c+o-3.8c/-1.0c+h",
    frame=["xaf+lPressure (hPa)"],
)

fig.colorbar(
    position="JBC+w4c/0.25c+o3.7c/-1.0c+h",
    frame=["xaf+lSI (°C)"],
)


# ============================================================
# 15. 输出
# ============================================================
fig.savefig("wrf_pressure_si_pygmt.png", dpi=300)
fig.savefig("wrf_pressure_si_pygmt.pdf")
fig.show()