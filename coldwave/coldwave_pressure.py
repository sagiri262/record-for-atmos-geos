# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime


FILE = "/home/zy/record-for-atmos-geos/coldwave/datasets/data.grib"
FIG_DIR = "/home/zy/record-for-atmos-geos/coldwave/figs"   # 输出目录
os.makedirs(FIG_DIR, exist_ok=True)  # 若目录不存在则自动创建
OUTPUT_PNG = os.path.join(FIG_DIR, "coldwave_mslp_wind.png")

t0   = datetime(2025, 10, 17, 0)      # UTC

# =============== helpers ===============
def open_var(file, short_name):
    """Open a single-shortName dataset safely via cfgrib."""
    return xr.open_dataset(
        file, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}}
    )

def pick_var(ds, candidates):
    """Return the first existing variable from candidates."""
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"None of {candidates} found in dataset. Vars={list(ds.variables)}")

# =============== read ===============
# msl (Pa)
ds_msl = open_var(FILE, "msl")
# 10 m wind (m s-1): shortNames are 10u / 10v
ds_u10 = open_var(FILE, "10u")
ds_v10 = open_var(FILE, "10v")

# allow aliases (some workflows rename to u10/v10)
msl  = pick_var(ds_msl, ["msl"])
u10  = pick_var(ds_u10, ["10u", "u10"])
v10  = pick_var(ds_v10, ["10v", "v10"])

# choose time from msl and align others to it
time_vals = ds_msl.time.values
itime = int(np.argmin(np.abs(time_vals - np.datetime64(t0))))
t_sel = ds_msl.time[itime]
print("Using time:", np.datetime_as_string(time_vals[itime], unit="h"))

u10 = u10.sel(time=t_sel, method="nearest")
v10 = v10.sel(time=t_sel, method="nearest")
msl = (msl.isel(time=itime) / 100.0)  # Pa -> hPa

lon = msl.longitude.values
lat = msl.latitude.values
Lon, Lat = np.meshgrid(lon, lat)

# =============== figure ===============
fig = plt.figure(figsize=(15, 15), dpi=600)
proj_plot = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=proj_plot)

extent = [60, 150, 0, 70]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# coastlines & borders (international standard)
ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.5)
ax.coastlines(resolution="50m", linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black")

# MSLP contours
levels_all = np.arange(980, 1061, 5)    # 5 hPa
levels_red = np.arange(1000, 1101, 15)  # emphasize every 15 hPa starting at 1000

cs = ax.contour(Lon, Lat, msl.values, levels=levels_all,
                colors="grey", linewidths=0.9, transform=ccrs.PlateCarree())
ax.clabel(cs, fmt="%d hPa", inline=True, fontsize=10, colors="grey")

csr = ax.contour(Lon, Lat, msl.values, levels=levels_red,
                 colors="red", linewidths=1.4, transform=ccrs.PlateCarree())
ax.clabel(csr, fmt="%d hPa", inline=True, fontsize=12, colors="red")

# white wind vectors
skip = (slice(None, None, 8), slice(None, None, 8))
ax.quiver(Lon[skip], Lat[skip], u10.values[skip], v10.values[skip],
          color="white", transform=ccrs.PlateCarree(), scale=550, width=0.002)

# only left & bottom ticks/labels; 5-degree spacing; rotations
xticks = np.arange(int(extent[0]/5)*5, extent[1]+1, 5)
yticks = np.arange(int(extent[2]/5)*5, extent[3]+1, 5)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True)
ax.tick_params(axis="y", right=False, left=True, labelleft=True)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(45)      # clockwise
    lbl.set_fontsize(8); lbl.set_fontfamily("Times New Roman")
for lbl in ax.get_yticklabels():
    lbl.set_rotation(-45)     # counter-clockwise
    lbl.set_fontsize(8); lbl.set_fontfamily("Times New Roman")

ax.set_title(f"MSLP (hPa) and 10 m Wind — {t0.strftime('%Y-%m-%d %H:%M')} UTC",
             fontsize=18, fontfamily="Times New Roman")

plt.savefig(OUTPUT_PNG, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to {OUTPUT_PNG}")
