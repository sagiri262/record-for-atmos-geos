#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
固定路径与时间：
- 从单个 ERA5 GRIB 读取：10u,10v,2t,msl/sp,tp
- 用“气压场最近时刻”作为统一目标时间，将所有变量对齐
- 自动处理 tp 的 step 维度，保证绘图传入 2D
- 生成三图：
  1) 中国 10m 风(箭头) + 等压线 + 气压填色（红->蓝 = 高->低）
  2) 同一套等压线 + 2 m 气温（°C）
  3) 同一套等压线 + 总降水（mm）
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================
# 固定输入与输出（按你的要求）
GRIB_PATH = "/home/zy/record-for-atmos-geos/surface_pressure/era5-data/data.grib"
TARGET_TIME_STR = "2025-01-24T12:00:00"  # UTC
OUTDIR = Path("/home/zy/record-for-atmos-geos/surface_pressure/src/figs")
QUIVER_STRIDE = 6  # 风矢量抽稀
# 中国范围
LON_MIN, LON_MAX = 73, 135
LAT_MIN, LAT_MAX = 18, 54
# =========================

def open_grib_var_shortname(path, shortname):
    """优先用 shortName 过滤；失败则扫描所有 group 找 GRIB_shortName。"""
    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": shortname}},
        )
        da = ds[list(ds.data_vars)[0]]
        da.name = shortname
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
    """统一经纬度命名，并把经度转为 -180..180。"""
    if "latitude" in da.coords and "lat" not in da.coords:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.coords and "lon" not in da.coords:
        da = da.rename({"longitude": "lon"})
    if "lon" in da.coords and da.lon.max() > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    return da

def subset_china(da):
    """裁剪中国范围（73–135E，18–54N），自动适配纬向排序。"""
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
    """返回时间坐标名：若有 time 用 time；否则若有 valid_time 用 valid_time；都没有则 None。"""
    if "time" in da.coords:
        return "time"
    if "valid_time" in da.coords:
        return "valid_time"
    return None

def select_nearest_time(da, target_time):
    """
    不改坐标名，按变量自身的时间坐标（time 或 valid_time）取最近时刻。
    返回 (子集后的 DataArray, 实际选到的时间字符串)
    """
    coord = get_time_coord_name(da)
    if coord is None:
        return da, "NO_TIME"
    out = da.sel({coord: np.datetime64(target_time)}, method="nearest")
    picked = np.datetime_as_string(out[coord].values, unit="m")
    return out, picked

def select_exact_else_nearest(da, target_time):
    """
    先尝试精确选到 target_time（同一坐标名下），失败则退回最近邻并给出提示。
    返回 (DataArray, picked_time_str, used_nearest_bool)
    """
    coord = get_time_coord_name(da)
    if coord is None:
        return da, "NO_TIME", False
    t64 = np.datetime64(target_time)
    try:
        out = da.sel({coord: t64})
        picked = np.datetime_as_string(out[coord].values, unit="m")
        return out, picked, False
    except Exception:
        out, picked = select_nearest_time(da, t64)
        return out, picked, True

def ensure_2d(da):
    """
    保证传给绘图的是 2D：
    - 挤掉所有长度为1的维度
    - 若还存在 'step' 且长度>1，则取最后一个 step（累计量终值）
    """
    # squeeze 单例维
    da2 = da.squeeze(drop=True)
    # 如果还有 step 且长度>1，取最后一个
    if "step" in da2.dims and da2.sizes.get("step", 1) > 1:
        da2 = da2.isel(step=-1)
    # 再 squeeze 一次，确保只有 lat/lon
    da2 = da2.squeeze(drop=True)
    return da2

def to_hpa(da):
    out = da / 100.0
    out.attrs["units"] = "hPa"
    return out

def to_celsius(da):
    out = da - 273.15
    out.attrs["units"] = "°C"
    return out

def to_mm(da):
    out = da * 1000.0
    out.attrs["units"] = "mm"
    return out

def add_common(ax):
    ax.coastlines(resolution="50m", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 读取必要变量（按你的文件 shortName）
    u10 = open_grib_var_shortname(GRIB_PATH, "10u")
    v10 = open_grib_var_shortname(GRIB_PATH, "10v")
    t2t = open_grib_var_shortname(GRIB_PATH, "2t")
    tp  = open_grib_var_shortname(GRIB_PATH, "tp")
    try:
        p = open_grib_var_shortname(GRIB_PATH, "msl")
        p_is_msl = True
    except Exception:
        p = open_grib_var_shortname(GRIB_PATH, "sp")
        p_is_msl = False

    # 以“气压场最近时刻”为统一目标时间
    p, picked_p = select_nearest_time(p, TARGET_TIME_STR)
    p_tcoord = get_time_coord_name(p)
    target_time_val = p[p_tcoord].values  # numpy datetime64

    # 其它变量对齐到同一刻（tp 先尝试精确齐，失败再最近邻并告警）
    u10, picked_u10 = select_nearest_time(u10, target_time_val)
    v10, picked_v10 = select_nearest_time(v10, target_time_val)
    t2t, picked_t2t = select_nearest_time(t2t, target_time_val)
    tp, picked_tp, used_nearest_tp = select_exact_else_nearest(tp, target_time_val)

    print("[TIME PICKED] p  :", picked_p, "(MSLP)" if p_is_msl else "(SP)")
    print("[TIME PICKED] 10u:", picked_u10)
    print("[TIME PICKED] 10v:", picked_v10)
    print("[TIME PICKED] 2t :", picked_t2t)
    print("[TIME PICKED] tp :", picked_tp)
    if used_nearest_tp:
        print("[WARN] 'tp' 无精确的", np.datetime_as_string(target_time_val, unit="m"),
              "时刻，已用最近邻：", picked_tp)

    # 裁剪中国
    u10 = subset_china(u10)
    v10 = subset_china(v10)
    t2t = subset_china(t2t)
    tp  = subset_china(tp)
    p   = subset_china(p)

    # 单位
    p_hpa = to_hpa(p)         # Pa -> hPa
    t_c   = to_celsius(t2t)   # K  -> °C
    tp_mm = to_mm(tp)         # m  -> mm（逐小时累计）

    # 确保 2D（特别是 tp 可能有 step 维）
    p_hpa = ensure_2d(p_hpa)
    t_c   = ensure_2d(t_c)
    tp_mm = ensure_2d(tp_mm)
    u10   = ensure_2d(u10)
    v10   = ensure_2d(v10)

    # 以气压网格为基准对齐其他场（最近邻，不依赖 SciPy）
    u10i = u10.interp(lon=p_hpa.lon, lat=p_hpa.lat, method="nearest")
    v10i = v10.interp(lon=p_hpa.lon, lat=p_hpa.lat, method="nearest")
    t_c  = t_c.interp(lon=p_hpa.lon, lat=p_hpa.lat, method="nearest")
    tp_mm = tp_mm.interp(lon=p_hpa.lon, lat=p_hpa.lat, method="nearest")

    lon = p_hpa.lon; lat = p_hpa.lat

    # 风矢量抽稀
    s = QUIVER_STRIDE
    qlon = lon.values[::s]; qlat = lat.values[::s]
    Uq = u10i.values[::s, ::s]; Vq = v10i.values[::s, ::s]
    LON, LAT = np.meshgrid(qlon, qlat)

    proj = ccrs.PlateCarree()
    p_label = "MSLP (hPa)" if p_is_msl else "Surface Pressure (hPa)"
    when_str = np.datetime_as_string(target_time_val, unit="m")

    # 1) 风 + 等压线 + 气压填色（红->蓝 高->低）
    fig1 = plt.figure(figsize=(10, 8), dpi=150)
    ax1 = plt.axes(projection=proj); add_common(ax1)
    cf = ax1.contourf(lon, lat, p_hpa.values, levels=24, cmap="RdBu_r", transform=proj)
    cbar = plt.colorbar(cf, ax=ax1, pad=0.02, aspect=30); cbar.set_label(p_label)
    cs = ax1.contour(lon, lat, p_hpa.values, levels=24, colors="k", linewidths=0.5, transform=proj)
    ax1.clabel(cs, fmt="%.0f", inline=True, fontsize=6)
    ax1.quiver(LON, LAT, Uq, Vq, transform=proj, scale=450, width=0.002, headwidth=3)
    ax1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax1.set_title(f"China | 10 m Wind + {p_label} (filled & isobars)\n{when_str}")
    fig1.tight_layout(); fig1.savefig(OUTDIR / "china_wind_pressure.png", bbox_inches="tight")

    # 2) 等压线 + 2m 温度
    fig2 = plt.figure(figsize=(10, 8), dpi=150)
    ax2 = plt.axes(projection=proj); add_common(ax2)
    cf2 = ax2.contourf(lon, lat, t_c.values, levels=24, cmap="RdBu_r", transform=proj)
    cbar2 = plt.colorbar(cf2, ax=ax2, pad=0.02, aspect=30); cbar2.set_label("2 m Temperature (°C)")
    cs2 = ax2.contour(lon, lat, p_hpa.values, levels=24, colors="k", linewidths=0.5, transform=proj)
    ax2.clabel(cs2, fmt="%.0f", inline=True, fontsize=6)
    ax2.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax2.set_title(f"China | {p_label} Isobars over 2 m Temperature\n{when_str}")
    fig2.tight_layout(); fig2.savefig(OUTDIR / "china_isobars_temperature.png", bbox_inches="tight")

    # 3) 等压线 + 降水（mm）
    fig3 = plt.figure(figsize=(10, 8), dpi=150)
    ax3 = plt.axes(projection=proj); add_common(ax3)
    cf3 = ax3.contourf(lon, lat, tp_mm.values, levels=24, cmap="RdBu_r", transform=proj)
    cbar3 = plt.colorbar(cf3, ax=ax3, pad=0.02, aspect=30); cbar3.set_label("Total Precipitation (mm)")
    cs3 = ax3.contour(lon, lat, p_hpa.values, levels=24, colors="k", linewidths=0.5, transform=proj)
    ax3.clabel(cs3, fmt="%.0f", inline=True, fontsize=6)
    ax3.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax3.set_title(f"China | {p_label} Isobars over Total Precipitation\n{when_str}")
    fig3.tight_layout(); fig3.savefig(OUTDIR / "china_isobars_precip.png", bbox_inches="tight")

    print("Saved:",
          OUTDIR / "china_wind_pressure.png",
          OUTDIR / "china_isobars_temperature.png",
          OUTDIR / "china_isobars_precip.png")

if __name__ == "__main__":
    OUTDIR = OUTDIR  # ensure path var is referenced
    main()
