#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A: 普通 WRF wrfout
   每 3 小时绘制 500 hPa 位势高度等高线图：
   DD-HH-MM-00-500hpa.jpg
   DD-DD-500hpa.gif

B: WRF-Chem wrfout
   每 3 小时绘制近地面总沙尘水平分布：
   DD-HH-MM-00-dust_dist.jpg
   DD-DD-dust_dist.gif
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset, chartostring
from wrf import getvar, interplevel, latlon_coords, get_cartopy, to_np
import imageio.v2 as imageio


# =========================
# 需要你修改的路径
# =========================
A_FILES = [
    "/home/zy/WRF/WRF_result/era5_260512_result/wrfout_d02_2025-04-09_08:00:00",
]

B_FILES = [
    "/home/zy/WRF/WRF_result/era5_260512_result/chem_result/wrfout_d02_2025-04-09_08:00:00",
]

OUT_A = Path("./A_500hpa")
OUT_B = Path("./B_dust_dist")

TIME_INTERVAL_HOURS = 3
LATLON_INTERVAL = 5

# B：默认画最底层 total dust = DUST_1 + ... + DUST_5
DUST_LEVEL_INDEX = 0
DUST_BINS_EXPECTED = ["DUST_1", "DUST_2", "DUST_3", "DUST_4", "DUST_5"]

# dust 色标：浅黄色 -> 深棕色
DUST_CMAP = LinearSegmentedColormap.from_list(
    "dust_yellow_to_brown",
    ["#fff7bc", "#fec44f", "#d95f0e", "#5a1f00"]
)


def open_files(file_list):
    files = []
    for f in file_list:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"找不到文件: {p}")
        files.append(Dataset(str(p)))
    return files


def read_times_from_nc(nc):
    if "Times" not in nc.variables:
        raise KeyError("wrfout 中找不到 Times 变量")
    tstrings = chartostring(nc.variables["Times"][:])
    times = []
    for s in tstrings:
        s = str(s)
        times.append(datetime.strptime(s, "%Y-%m-%d_%H:%M:%S"))
    return times


def collect_time_records(nc_list):
    records = []
    for nc in nc_list:
        times = read_times_from_nc(nc)
        for tidx, dt in enumerate(times):
            records.append((nc, tidx, dt))

    records.sort(key=lambda x: x[2])

    # 去掉多文件拼接时可能重复的时间
    unique = []
    seen = set()
    for rec in records:
        if rec[2] not in seen:
            unique.append(rec)
            seen.add(rec[2])
    return unique


def select_every_3h(records):
    if not records:
        return []
    start = records[0][2]
    selected = []
    for nc, tidx, dt in records:
        hours = round((dt - start).total_seconds() / 3600.0)
        if hours % TIME_INTERVAL_HOURS == 0:
            selected.append((nc, tidx, dt))
    return selected


def tick_values(vmin, vmax, step=5):
    start = np.floor(vmin / step) * step
    end = np.ceil(vmax / step) * step
    return np.arange(start, end + step, step)


def setup_map(ax, lons, lats):
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 只加国界线和海岸线，不对陆地/海洋填色
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="black")
    ax.coastlines(resolution="50m", linewidth=0.5)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.0,
        xlocs=tick_values(lon_min, lon_max, LATLON_INTERVAL),
        ylocs=tick_values(lat_min, lat_max, LATLON_INTERVAL),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}


def make_gif(frame_paths, gif_path, duration=0.45):
    if not frame_paths:
        return
    imgs = [imageio.imread(str(p)) for p in frame_paths]
    imageio.mimsave(str(gif_path), imgs, duration=duration)


def plot_a_500hpa(records, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for nc, tidx, dt in records:
        pressure = getvar(nc, "pressure", timeidx=tidx)      # hPa
        height_m = getvar(nc, "z", timeidx=tidx, units="m")  # geopotential height, m

        z500_m = interplevel(height_m, pressure, 500.0)
        z500_dam = to_np(z500_m) / 10.0

        lats, lons = latlon_coords(z500_m)
        lats = to_np(lats)
        lons = to_np(lons)
        cart_proj = get_cartopy(z500_m)

        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = plt.axes(projection=cart_proj)
        setup_map(ax, lons, lats)

        zmin = np.nanmin(z500_dam)
        zmax = np.nanmax(z500_dam)
        interval = 4.0
        levels = np.arange(
            np.floor(zmin / interval) * interval,
            np.ceil(zmax / interval) * interval + interval,
            interval,
        )

        cs = ax.contour(
            lons,
            lats,
            z500_dam,
            levels=levels,
            colors="black",
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

        ax.set_title(f"500 hPa Geopotential Height / {dt:%Y-%m-%d %H:%M UTC}", fontsize=12)

        fname = f"{dt:%d-%H-%M-00}-500hpa.jpg"
        fpath = outdir / fname
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        plt.close(fig)
        frame_paths.append(fpath)

    if frame_paths:
        gif_name = f"{records[0][2]:%d}-{records[-1][2]:%d}-500hpa.gif"
        make_gif(frame_paths, outdir / gif_name)


def find_case_insensitive_var(nc, varname):
    if varname in nc.variables:
        return varname
    lower_map = {v.lower(): v for v in nc.variables.keys()}
    key = varname.lower()
    if key not in lower_map:
        return None
    return lower_map[key]


def get_dust_bin_names(nc):
    names = []
    for v in DUST_BINS_EXPECTED:
        vv = find_case_insensitive_var(nc, v)
        if vv is not None:
            names.append(vv)
    if not names:
        raise KeyError(
            "B 文件中没有找到 DUST_1...DUST_5。请先 ncdump -h 检查沙尘变量名。"
        )
    return names


def read_surface_total_dust(nc, tidx, dust_names):
    total = None
    for v in dust_names:
        arr = nc.variables[v][tidx, DUST_LEVEL_INDEX, :, :]
        arr = np.ma.filled(arr, np.nan).astype(float)
        if total is None:
            total = arr
        else:
            total = total + arr
    return total


def estimate_dust_vmax(records):
    pctl_values = []

    # 用每一帧的 98 分位估计全局色标上限，避免个别极端值把图压暗
    for nc, tidx, dt in records:
        dust_names = get_dust_bin_names(nc)
        arr = read_surface_total_dust(nc, tidx, dust_names)
        vals = arr[np.isfinite(arr) & (arr > 0)]
        if vals.size > 0:
            pctl_values.append(np.nanpercentile(vals, 98))

    if not pctl_values:
        return 1.0

    vmax = float(np.nanmax(pctl_values))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    return vmax


def plot_b_dust(records, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    dust_vmax = estimate_dust_vmax(records)

    for nc, tidx, dt in records:
        dust_names = get_dust_bin_names(nc)
        dust_total = read_surface_total_dust(nc, tidx, dust_names)

        # 取 pressure 只是为了拿到 WRF 投影和经纬度
        pressure = getvar(nc, "pressure", timeidx=tidx)
        ref2d = pressure[0, :, :]

        lats, lons = latlon_coords(ref2d)
        lats = to_np(lats)
        lons = to_np(lons)
        cart_proj = get_cartopy(ref2d)

        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = plt.axes(projection=cart_proj)
        setup_map(ax, lons, lats)

        mesh = ax.pcolormesh(
            lons,
            lats,
            dust_total,
            cmap=DUST_CMAP,
            vmin=0,
            vmax=dust_vmax,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

        cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.82)
        cbar.set_label("Total dust mixing ratio at lowest model level (ug/kg-dryair)")

        ax.set_title(
            f"WRF-Chem Dust Distribution / {dt:%Y-%m-%d %H:%M UTC}\n"
            f"sum({', '.join(dust_names)}) at bottom_top={DUST_LEVEL_INDEX}",
            fontsize=12,
        )

        fname = f"{dt:%d-%H-%M-00}-dust_dist.jpg"
        fpath = outdir / fname
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        plt.close(fig)
        frame_paths.append(fpath)

    if frame_paths:
        gif_name = f"{records[0][2]:%d}-{records[-1][2]:%d}-dust_dist.gif"
        make_gif(frame_paths, outdir / gif_name)


def main():
    a_nc = open_files(A_FILES)
    b_nc = open_files(B_FILES)

    try:
        a_records_all = collect_time_records(a_nc)
        b_records_all = collect_time_records(b_nc)

        a_records = select_every_3h(a_records_all)
        b_records = select_every_3h(b_records_all)

        print(f"A 总时次数: {len(a_records_all)}, 每 3 小时输出: {len(a_records)}")
        print(f"B 总时次数: {len(b_records_all)}, 每 3 小时输出: {len(b_records)}")

        plot_a_500hpa(a_records, OUT_A)
        plot_b_dust(b_records, OUT_B)

        print(f"A 输出目录: {OUT_A.resolve()}")
        print(f"B 输出目录: {OUT_B.resolve()}")

    finally:
        for nc in a_nc + b_nc:
            nc.close()


if __name__ == "__main__":
    main()