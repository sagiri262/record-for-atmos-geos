#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarsWRF meridional sections at a fixed longitude.

Figure 1:
    latitude-height cross-section at lon0,
    vectors of V and W*50, isobars, equator dashed line, MOLA DEM profile.
Figure 2:
    latitude-height temperature section, ReBu-like filled contours,
    horizontal colorbar at the lower-right outside the axes, isobars,
    equator dashed line, MOLA DEM profile.

Notes
-----
1) This script assumes MarsWRF uses WRF-style NetCDF conventions.
2) Because MarsWRF output variants differ, temperature is searched using
   actual-temperature-like names first. If only the WRF perturbation variable
   "T" exists, this script deliberately stops and asks you to provide an
   actual temperature variable name, rather than silently using a possibly
   wrong conversion.
3) Heights are computed with Mars gravity (3.72 m s-2) from PH+PHB if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# =========================================================
# 0. 导入上级目录中的 wrf_read_data.py
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

try:
    from wrf_read_data import WRFDataReader
    HAS_WRF_READER = True
except Exception:
    HAS_WRF_READER = False

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
import rasterio


G_MARS = 3.72  # m s^-2
P_LEVELS = np.array([610, 400, 200, 100, 50, 10, 5, 1, 0.1], dtype=float)
Y_TICKS = np.arange(-5, 86, 10, dtype=float)
Z_LIM = (-5, 85)


def find_var(ds: xr.Dataset, candidates: list[str], required: bool = True) -> xr.DataArray | None:
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    if required:
        raise KeyError(f"None of these variables were found: {candidates}")
    return None


def squeeze_time(da: xr.DataArray, time_index: int) -> xr.DataArray:
    for dim in da.dims:
        if dim.lower() == "time":
            return da.isel({dim: time_index})
    return da


def unstagger_da(da: xr.DataArray, stag_dim: str, out_dim: str | None = None) -> xr.DataArray:
    if stag_dim not in da.dims:
        return da
    out_dim = out_dim or stag_dim.replace("_stag", "")
    left = da.isel({stag_dim: slice(0, -1)}).rename({stag_dim: out_dim})
    right = da.isel({stag_dim: slice(1, None)}).rename({stag_dim: out_dim})
    n = left.sizes[out_dim]
    coord = np.arange(n)
    left = left.assign_coords({out_dim: coord})
    right = right.assign_coords({out_dim: coord})
    return 0.5 * (left + right)


def reorder_3d(da: xr.DataArray) -> xr.DataArray:
    dims = list(da.dims)
    zdim = next((d for d in dims if d.lower() in ("bottom_top", "lev", "level", "altitude", "z")), None)
    ydim = next((d for d in dims if ("south_north" in d.lower()) or d.lower() in ("lat", "latitude")), None)
    xdim = next((d for d in dims if ("west_east" in d.lower()) or d.lower() in ("lon", "longitude")), None)
    if not all([zdim, ydim, xdim]):
        raise ValueError(f"Cannot infer z/y/x dimensions from dims={dims}")
    return da.transpose(zdim, ydim, xdim)


def reorder_2d(da: xr.DataArray) -> xr.DataArray:
    dims = list(da.dims)
    ydim = next((d for d in dims if ("south_north" in d.lower()) or d.lower() in ("lat", "latitude")), None)
    xdim = next((d for d in dims if ("west_east" in d.lower()) or d.lower() in ("lon", "longitude")), None)
    if not all([ydim, xdim]):
        raise ValueError(f"Cannot infer y/x dimensions from dims={dims}")
    return da.transpose(ydim, xdim)


def wrap_target_lon(lon_target: float, lon_grid: np.ndarray) -> float:
    lon_min = np.nanmin(lon_grid)
    lon_max = np.nanmax(lon_grid)
    if lon_min >= 0 and lon_max > 180:
        lon_target = lon_target % 360.0
    elif lon_max <= 180 and lon_min < 0:
        if lon_target > 180:
            lon_target -= 360.0
    return lon_target


def rowwise_lon_section_indices(lon2d: np.ndarray, lon_target: float) -> np.ndarray:
    lon_target = wrap_target_lon(lon_target, lon2d)
    return np.nanargmin(np.abs(lon2d - lon_target), axis=1)


def extract_section_rowwise(field3d: np.ndarray, ix_by_row: np.ndarray) -> np.ndarray:
    nz, ny, nx = field3d.shape
    out = np.empty((nz, ny), dtype=float)
    rows = np.arange(ny)
    for k in range(nz):
        slc = field3d[k, :, :]
        out[k, :] = slc[rows, ix_by_row]
    return out


def section_sort_by_lat(lat_sec: np.ndarray, *arrs: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(lat_sec)
    out = [lat_sec[order]]
    for arr in arrs:
        if arr.ndim == 1:
            out.append(arr[order])
        elif arr.ndim == 2:
            out.append(arr[:, order])
        else:
            raise ValueError("Only 1D or 2D arrays can be sorted by latitude here.")
    return tuple(out)


def get_temperature(ds: xr.Dataset, time_index: int) -> xr.DataArray:
    # Prefer true temperature / diagnostic temperature names.
    cand_true = ["tk", "TK", "temp", "TEMP", "temperature", "TEMPERATURE", "TABS", "t"]
    da = find_var(ds, cand_true, required=False)
    if da is not None:
        return reorder_3d(squeeze_time(da, time_index))

    if "T" in ds.variables:
        raise KeyError(
            "Only variable 'T' was found. In WRF-style output, 'T' is often perturbation potential temperature, "
            "not true temperature. Please replace the candidate list with your actual temperature variable name "
            "(for example tk/temp/temperature) or convert it explicitly for MarsWRF."
        )

    raise KeyError("No temperature variable found.")


def get_pressure(ds: xr.Dataset, time_index: int) -> xr.DataArray:
    if "pressure" in ds.variables:
        return reorder_3d(squeeze_time(ds["pressure"], time_index))
    if "PRES" in ds.variables:
        return reorder_3d(squeeze_time(ds["PRES"], time_index))
    if "pres" in ds.variables:
        return reorder_3d(squeeze_time(ds["pres"], time_index))
    if "P" in ds.variables and "PB" in ds.variables:
        p = squeeze_time(ds["P"], time_index) + squeeze_time(ds["PB"], time_index)
        return reorder_3d(p)
    if "P" in ds.variables:
        return reorder_3d(squeeze_time(ds["P"], time_index))
    raise KeyError("No pressure variable found. Expected pressure/PRES/pres or P(+PB).")


def get_height_km(ds: xr.Dataset, time_index: int) -> xr.DataArray:
    for name in ["z", "Z", "height", "HEIGHT", "geoht", "GEOHT"]:
        if name in ds.variables:
            da = reorder_3d(squeeze_time(ds[name], time_index))
            arr = da.values.astype(float)
            if np.nanmean(arr) > 1e3:
                arr = arr / 1000.0
            return xr.DataArray(arr, dims=da.dims, coords=da.coords)

    if "PH" in ds.variables and "PHB" in ds.variables:
        phi = squeeze_time(ds["PH"], time_index) + squeeze_time(ds["PHB"], time_index)
        phi = unstagger_da(phi, "bottom_top_stag", "bottom_top")
        z = reorder_3d(phi) / G_MARS / 1000.0
        return z

    raise KeyError("No height variable found. Expected z/height or PH+PHB.")


def get_meridional_v(ds: xr.Dataset, time_index: int) -> xr.DataArray:
    da = find_var(ds, ["V", "v", "va"])
    da = squeeze_time(da, time_index)
    da = unstagger_da(da, "south_north_stag", "south_north")
    return reorder_3d(da)


def get_vertical_w(ds: xr.Dataset, time_index: int) -> xr.DataArray:
    da = find_var(ds, ["W", "w", "wa"])
    da = squeeze_time(da, time_index)
    da = unstagger_da(da, "bottom_top_stag", "bottom_top")
    return reorder_3d(da)


def get_latlon(ds: xr.Dataset, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    lat = reorder_2d(squeeze_time(find_var(ds, ["XLAT", "xlat", "lat", "XLAT_M"]), time_index)).values.astype(float)
    lon = reorder_2d(squeeze_time(find_var(ds, ["XLONG", "xlong", "lon", "XLONG_M"]), time_index)).values.astype(float)
    return lat, lon


def read_dem_profile(dem_tif: str | Path, lat_sec: np.ndarray, lon0: float) -> np.ndarray:
    with rasterio.open(dem_tif) as src:
        dem = src.read(1).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        h, w = dem.shape
        b = src.bounds

        # Cell-center coordinates from GeoTIFF bounds.
        lons = np.linspace(b.left + (b.right - b.left) / (2 * w),
                           b.right - (b.right - b.left) / (2 * w), w)
        lats = np.linspace(b.top - (b.top - b.bottom) / (2 * h),
                           b.bottom + (b.top - b.bottom) / (2 * h), h)

        # Fallback for rasters without a useful planetary CRS transform.
        if (not np.isfinite(lons).all()) or (abs(lons[-1] - lons[0]) < 1):
            lons = np.linspace(0.0, 360.0, w, endpoint=False) + 180.0 / w
            lats = np.linspace(90.0, -90.0, h)

        lon_dem = wrap_target_lon(lon0, lons[np.newaxis, :])
        ix = int(np.nanargmin(np.abs(lons - lon_dem)))
        prof = dem[:, ix]

        # Interpolate to section latitudes. Make lats ascending for interp.
        if lats[0] > lats[-1]:
            lats_i = lats[::-1]
            prof_i = prof[::-1]
        else:
            lats_i = lats
            prof_i = prof

        f = interp1d(lats_i, prof_i, bounds_error=False, fill_value=np.nan)
        topo_m = f(lat_sec)
        return topo_m / 1000.0


def build_pressure_axis(ax: plt.Axes, z_km: np.ndarray, p_pa: np.ndarray, p_levels: np.ndarray) -> plt.Axes:
    z_ref = np.nanmean(z_km, axis=1)
    p_ref = np.nanmean(p_pa, axis=1)
    mask = np.isfinite(z_ref) & np.isfinite(p_ref) & (p_ref > 0)
    logp = np.log10(p_ref[mask])
    z_ok = z_ref[mask]
    order = np.argsort(logp)
    logp_sorted = logp[order]
    z_sorted = z_ok[order]
    zticks = np.interp(np.log10(p_levels), logp_sorted, z_sorted, left=np.nan, right=np.nan)
    valid = np.isfinite(zticks)

    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks(zticks[valid])
    axr.set_yticklabels([f"{p:g}" for p in p_levels[valid]])
    axr.tick_params(direction="out", length=3, width=0.6, labelsize=9)
    axr.set_ylabel("")
    return axr


def rebu_like_cmap() -> LinearSegmentedColormap:
    # A ReBu-like table chosen to visually resemble the provided example.
    colors = [
        "#2A0B8D", "#1D4DB5", "#2C84D8", "#44C7F4",
        "#62E2C7", "#B9EB6C", "#F3EB56", "#F6B03C",
        "#F06A25", "#D92323", "#8A0068"
    ]
    return LinearSegmentedColormap.from_list("ReBu_like", colors, N=256)


def make_figure1(lat_sec, z_sec, v_sec, w_sec, p_sec, topo_km, out_png, lon0, title_right=""):
    fig = plt.figure(figsize=(7.2, 5.8), dpi=180)
    ax = fig.add_axes([0.10, 0.13, 0.78, 0.78])

    ax.set_xlim(-90, 90)
    ax.set_ylim(*Z_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Altitude (km)")
    ax.axvline(0.0, linestyle="--", color="0.5", linewidth=0.5)

    # isobars
    cs_p = ax.contour(lat_sec[np.newaxis, :].repeat(z_sec.shape[0], axis=0),
                      z_sec, p_sec,
                      levels=P_LEVELS, colors="0.55", linewidths=0.3)
    # Optional: keep uncluttered, do not label isobars.
    _ = cs_p

    # terrain
    ax.fill_between(lat_sec, Z_LIM[0], topo_km, color="0.75", zorder=3)
    ax.plot(lat_sec, topo_km, color="k", linewidth=0.7, zorder=4)

    # vectors (subsample for readability)
    skip_y = 4
    skip_z = 3
    X = lat_sec[None, :].repeat(z_sec.shape[0], axis=0)
    q = ax.quiver(X[::skip_z, ::skip_y], z_sec[::skip_z, ::skip_y],
                  v_sec[::skip_z, ::skip_y], (w_sec * 50.0)[::skip_z, ::skip_y],
                  color="0.2", angles="xy", scale_units="xy", scale=6.0,
                  width=0.0018, headwidth=3.5, headlength=4.5)
    ax.quiverkey(q, 0.92, 1.02, 10, "10\n m/s", labelpos="E", coordinates="axes", fontproperties={"size": 8})

    ax.set_title("V, W*50 (m/s)", loc="left", fontsize=11)
    if title_right:
        ax.set_title(title_right, loc="right", fontsize=9)
    else:
        ax.text(0.99, 1.01, f"Lon={lon0:.1f}°", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)

    build_pressure_axis(ax, z_sec, p_sec, P_LEVELS)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure2(lat_sec, z_sec, t_sec, p_sec, topo_km, out_png, lon0, title_right=""):
    fig = plt.figure(figsize=(7.2, 5.8), dpi=180)
    ax = fig.add_axes([0.10, 0.13, 0.78, 0.78])

    ax.set_xlim(-90, 90)
    ax.set_ylim(*Z_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Altitude (km)")
    ax.axvline(0.0, linestyle="--", color="0.5", linewidth=0.5)

    tmin = np.nanmin(t_sec)
    tmax = np.nanmax(t_sec)
    levels = np.arange(np.floor(tmin / 8.0) * 8.0, np.ceil(tmax / 8.0) * 8.0 + 8.0, 8.0)
    cmap = rebu_like_cmap()

    X = lat_sec[np.newaxis, :].repeat(z_sec.shape[0], axis=0)
    cf = ax.contourf(X, z_sec, t_sec, levels=levels, cmap=cmap, extend="both")
    cs = ax.contour(X, z_sec, t_sec, levels=levels, colors="0.25", linewidths=0.35)
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True, inline_spacing=2)

    cs_p = ax.contour(X, z_sec, p_sec, levels=P_LEVELS, colors="0.55", linewidths=0.3)
    _ = cs_p

    ax.fill_between(lat_sec, Z_LIM[0], topo_km, color="0.75", zorder=3)
    ax.plot(lat_sec, topo_km, color="k", linewidth=0.7, zorder=4)

    ax.set_title("Temperature (K)", loc="left", fontsize=11)
    ax.text(0.01, 0.995, f"Tmax:{np.nanmax(t_sec):.2f}, Tmin:{np.nanmin(t_sec):.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    if title_right:
        ax.set_title(title_right, loc="right", fontsize=9)
    else:
        ax.text(0.99, 1.01, f"Lon={lon0:.1f}°", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)

    build_pressure_axis(ax, z_sec, p_sec, P_LEVELS)

    # Lower-right outside colorbar
    cax = fig.add_axes([0.60, 0.055, 0.24, 0.022])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("(K)", fontsize=9)
    cb.ax.tick_params(labelsize=8, length=2)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="MarsWRF meridional sections at fixed longitude")
    ap.add_argument("--nc", required=True, help="MarsWRF NetCDF file")
    ap.add_argument("--dem", required=True, help="MOLA GeoTIFF DEM")
    ap.add_argument("--lon", type=float, default=110.0, help="section longitude in degree east")
    ap.add_argument("--time", type=int, default=0, help="time index")
    ap.add_argument("--tag", default="D01", help="string shown in the upper-right title")
    ap.add_argument("--outdir", default=".", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.nc)

    lat2d, lon2d = get_latlon(ds, args.time)
    ix_row = rowwise_lon_section_indices(lon2d, args.lon)
    rows = np.arange(lat2d.shape[0])
    lat_sec = lat2d[rows, ix_row]

    z = get_height_km(ds, args.time).values.astype(float)
    p = get_pressure(ds, args.time).values.astype(float)
    v = get_meridional_v(ds, args.time).values.astype(float)
    w = get_vertical_w(ds, args.time).values.astype(float)
    t = get_temperature(ds, args.time).values.astype(float)

    z_sec = extract_section_rowwise(z, ix_row)
    p_sec = extract_section_rowwise(p, ix_row)
    v_sec = extract_section_rowwise(v, ix_row)
    w_sec = extract_section_rowwise(w, ix_row)
    t_sec = extract_section_rowwise(t, ix_row)

    lat_sec, z_sec, p_sec, v_sec, w_sec, t_sec = section_sort_by_lat(lat_sec, z_sec, p_sec, v_sec, w_sec, t_sec)
    topo_km = read_dem_profile(args.dem, lat_sec, args.lon)

    title_right = args.tag
    make_figure1(lat_sec, z_sec, v_sec, w_sec, p_sec, topo_km,
                 outdir / f"marswrf_lon{args.lon:.1f}_vw_section.png", args.lon, title_right)
    make_figure2(lat_sec, z_sec, t_sec, p_sec, topo_km,
                 outdir / f"marswrf_lon{args.lon:.1f}_temperature_section.png", args.lon, title_right)

    print("Done.")
    print(outdir / f"marswrf_lon{args.lon:.1f}_vw_section.png")
    print(outdir / f"marswrf_lon{args.lon:.1f}_temperature_section.png")


if __name__ == "__main__":
    main()
