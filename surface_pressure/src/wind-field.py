#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中国范围水平风场分布图（10 m）：
- 底图为风速（m/s）着色
- 箭头表示风向与相对大小
- 使用 ERA5 单个 GRIB 文件（shortName: 10u, 10v, msl/sp 用于统一时间）
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ========= 固定配置（按你的环境） =========
GRIB_PATH = "/home/zy/record-for-atmos-geos/surface_pressure/era5-data/data.grib"
TARGET_TIME_STR = "2025-01-24T12:00:00"  # UTC
OUTDIR = Path("/home/zy/record-for-atmos-geos/surface_pressure/src/figs")
QUIVER_STRIDE = 6  # 风矢量抽稀（越小越密）
LON_MIN, LON_MAX = 73, 135
LAT_MIN, LAT_MAX = 18, 54
# ======================================

def open_grib_var_shortname(path, shortname):
    """优先 shortName 过滤；失败则扫描所有 group 找 GRIB_shortName。"""
    try:
        ds = xr.open_dataset(
            path, engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": shortname}},
        )
        da = ds[list(ds.data_vars)[0]]; da.name = shortname
        return da
    except Exception:
        import cfgrib
        dsets = cfgrib.open_datasets(path)
        for ds in dsets:
            for v in ds.data_vars:
                da = ds[v]
                if da.attrs.get("GRIB_shortName") == shortname:
                    da.name = shortname
                    return da
        raise RuntimeError(f"Variable shortName '{shortname}' not found in {path}")

def ensure_lonlat(da):
    """统一经纬度命名，经度转到 [-180,180]。"""
    if "latitude" in da.coords and "lat" not in da.coords:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.coords and "lon" not in da.coords:
        da = da.rename({"longitude": "lon"})
    if "lon" in da.coords and da.lon.max() > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    return da

def subset_china(da):
    """裁剪中国范围，自动适配纬向排序。"""
    da = ensure_lonlat(da)
    if "lat" in da.coords:
        if da.lat[0] > da.lat[-1]:
            da = da.sel(lat=slice(LAT_MAX, LAT_MIN))
        else:
            da = da.sel(lat=slice(LAT_MIN, LAT_MAX))
    if "lon" in da.coords:
        da = da.sel(lon=slice(LON_MIN, LON_MAX))
    return da

def get_time_coord_name(da):
    """返回时间坐标名：优先 'time'，否则 'valid_time'；都没有则 None。"""
    if "time" in da.coords: return "time"
    if "valid_time" in da.coords: return "valid_time"
    return None

def select_nearest_time(da, target_time):
    """按变量自身时间坐标（不改名）选最近时刻。"""
    coord = get_time_coord_name(da)
    if coord is None:
        return da, "NO_TIME"
    out = da.sel({coord: np.datetime64(target_time)}, method="nearest")
    picked = np.datetime_as_string(out[coord].values, unit="m")
    return out, picked

def ensure_2d(da):
    """保证绘图传入 2D：挤掉单例维；若有 step>1，取最后一个。"""
    da2 = da.squeeze(drop=True)
    if "step" in da2.dims and da2.sizes.get("step", 1) > 1:
        da2 = da2.isel(step=-1)
    return da2.squeeze(drop=True)

def add_common(ax):
    ax.coastlines(resolution="50m", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 读取必要变量
    u10 = open_grib_var_shortname(GRIB_PATH, "10u")
    v10 = open_grib_var_shortname(GRIB_PATH, "10v")
    # 用气压对齐时间（msl 优先，否则 sp）
    try:
        p = open_grib_var_shortname(GRIB_PATH, "msl")
    except Exception:
        p = open_grib_var_shortname(GRIB_PATH, "sp")

    # 以气压的最近时刻作为目标时间
    p, picked_p = select_nearest_time(p, TARGET_TIME_STR)
    p_tcoord = get_time_coord_name(p)
    target_time_val = p[p_tcoord].values

    # 其它变量对齐到同一刻
    u10, picked_u = select_nearest_time(u10, target_time_val)
    v10, picked_v = select_nearest_time(v10, target_time_val)

    print("[TIME] p:", picked_p, "| 10u:", picked_u, "| 10v:", picked_v)

    # 裁剪中国 & 保证 2D
    u10 = ensure_2d(subset_china(u10))
    v10 = ensure_2d(subset_china(v10))
    p   = ensure_2d(subset_china(p))  # 仅用于网格对齐/范围

    # 对齐到气压网格（最近邻，不依赖 SciPy）
    u10i = u10.interp(lon=p.lon, lat=p.lat, method="nearest")
    v10i = v10.interp(lon=p.lon, lat=p.lat, method="nearest")

    lon = p.lon; lat = p.lat

    # 计算风速
    wspd = np.hypot(u10i.values, v10i.values)  # m/s

    # 风矢量抽稀
    s = QUIVER_STRIDE
    qlon = lon.values[::s]; qlat = lat.values[::s]
    Uq = u10i.values[::s, ::s]; Vq = v10i.values[::s, ::s]
    LON, LAT = np.meshgrid(qlon, qlat)

    # 绘图
    proj = ccrs.PlateCarree()
    when_str = np.datetime_as_string(target_time_val, unit="m")

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.axes(projection=proj); add_common(ax)
    # 底图：风速（m/s）
    cf = ax.contourf(lon, lat, wspd, levels=20, transform=proj)
    cbar = plt.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("10 m Wind Speed (m/s)")
    # 箭头：风向与大小
    ax.quiver(LON, LAT, Uq, Vq, transform=proj, scale=450, width=0.002, headwidth=3)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax.set_title(f"China | 10 m Horizontal Wind Field\n{when_str}")

    out_png = OUTDIR / "china_wind_only.png"
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)

if __name__ == "__main__":
    main()
