"""
MarsWRF 经向剖面图（Python 版）
=====================================
图一：V+W*50 合成风矢量 + 地形剖面（110°E 经线）
图二：温度等温线填色图（RdBu 色表，8K 间隔）

变量对应：
  T       : 扰动位温 (K)，需加 T00=240K → 真实位温 θ = T + T00
            再由 θ 转温度 T_phys = θ * (P/P0)^(R/Cp)  P0=610Pa, R/Cp=0.2857
  P, PB   : 扰动气压 + 基态气压 → 总气压 P_total (Pa)
  PH, PHB : 扰动/基态位势高度 → 几何高度 z = (PH+PHB)/g  (g=3.72 m/s²)
  V       : 经向风 (m/s)，stagger 在 V 格，需插值到质量格
  W       : 垂直速度 (m/s)，stagger 在 W 格（n_vert+1层），需插值到质量格
  HGT     : 地表高度 (m)
  XLAT    : 纬度 (°)
  XLONG   : 经度 (°)

DEM 数据：Mars_MGS_MOLA_DEM_mosaic_global_463m.tif
  → 用 rasterio 读取，沿 110°E 取剖面

支持两种命令写法：

1) 标准 argparse 写法：
python marswrf_wind_vertical.py \
  --file "wrfout_d01_*" \
  --dem Mars_MGS_MOLA_DEM_mosaic_global_463m.tif \
  --lon_ref 110 \
  --sol_label "Sol:85-90" \
  --lst_label "13.74"

2) 兼容你现在的旧写法：
python marswrf_wind_vertical.py \
  file "wrfout_d01_*" \
  dem Mars_MGS_MOLA_DEM_mosaic_global_463m.tif \
  --lon_ref 110 \
  --sol_label "Sol:85-90" \
  --lst_label "13.74"
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Polygon
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

# =========================================================
# 0. 导入上级目录中的 wrf_read_data.py（若存在）
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

try:
    from wrf_read_data import WRFDataReader  # noqa: F401
    HAS_WRF_READER = True
except Exception:
    HAS_WRF_READER = False


def normalize_legacy_args(argv):
    """
    兼容旧式命令：
      python xxx.py file "aaa" dem "bbb" --lon_ref 110 ...
    自动转换为：
      python xxx.py --file "aaa" --dem "bbb" --lon_ref 110 ...

    同时忽略单独传入的 '\' token。
    """
    alias_map = {
        "file": "--file",
        "dem": "--dem",
    }

    normalized = []
    for tok in argv:
        if tok == "\\":
            continue
        normalized.append(alias_map.get(tok, tok))
    return normalized


# =========================================================
# 1. 参数
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--file", "-f",
    default="wrfout_d01_*",
    help="WRF 输出文件（支持通配符）"
)

parser.add_argument(
    "--dem",
    default="Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",
    help="DEM 文件路径"
)

parser.add_argument(
    "--lon_ref",
    type=float,
    default=110.0,
    help="经向剖面经度（°E），默认 110"
)

parser.add_argument(
    "--sol_label",
    default="Sol: 85-90",
    help="标题中的 Sol 标签"
)

parser.add_argument(
    "--lst_label",
    default="13.74",
    help="标题中的 LST 标签"
)

parser.add_argument("--out1", default="fig1_wind_section.png")
parser.add_argument("--out2", default="fig2_temp_section.png")

args = parser.parse_args(normalize_legacy_args(sys.argv[1:]))

# =========================================================
# 2. 常数
# =========================================================
g_mars = 3.72          # 火星重力加速度 m/s²
R_cp = 0.2857          # R/Cp
P_ref = 610.0          # 参考气压 Pa
T00 = 240.0            # WRF T00，若文件中有全局属性则覆盖

PLEVS_PA = np.array([610, 400, 200, 100, 50, 10, 5, 1, 0.1], dtype=float)
ALT_TICKS = [-5, 5, 15, 25, 35, 45, 55, 65, 75, 85]  # km

# =========================================================
# 3. 读取 WRF 文件并做时间平均
# =========================================================
files = sorted(glob.glob(args.file))

if not files:
    sys.exit(f"[ERROR] 未找到文件: {args.file}")

print(f"[INFO] 找到 {len(files)} 个文件")

initialized = False
T_sum = V_sum = W_sum = P_sum = Z_sum = None
HGT_sum = None
XLAT_arr = None
XLONG_arr = None
count = 0

for fpath in files:
    print(f"[INFO] 读取: {os.path.basename(fpath)}")
    with Dataset(fpath, "r") as ds:
        if hasattr(ds, "T00"):
            T00 = float(ds.T00)

        if XLAT_arr is None:
            XLAT_arr = ds.variables["XLAT"][0, :, :]
            XLONG_arr = ds.variables["XLONG"][0, :, :]

        ntimes = ds.variables["T"].shape[0]

        for t in range(ntimes):
            # 总气压
            P_tot = ds.variables["P"][t] + ds.variables["PB"][t]

            # 位势高度 -> 几何高度
            ph = ds.variables["PH"][t]
            phb = ds.variables["PHB"][t]
            z_stag = (ph + phb) / g_mars
            z_mass = 0.5 * (z_stag[:-1, :, :] + z_stag[1:, :, :])

            # 扰动位温 -> 真实温度
            theta = ds.variables["T"][t] + T00
            T_phys = theta * (P_tot / P_ref) ** R_cp

            # V 风：south_north_stag -> south_north
            v_stag = ds.variables["V"][t]
            v_mass = 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])

            # W 风：bottom_top_stag -> bottom_top
            w_stag = ds.variables["W"][t]
            w_mass = 0.5 * (w_stag[:-1, :, :] + w_stag[1:, :, :])

            # 地形
            hgt = ds.variables["HGT"][t]

            if not initialized:
                shp = T_phys.shape
                T_sum = np.zeros(shp)
                V_sum = np.zeros(shp)
                W_sum = np.zeros(shp)
                P_sum = np.zeros(shp)
                Z_sum = np.zeros(shp)
                HGT_sum = np.zeros(hgt.shape)
                initialized = True

            T_sum += T_phys
            V_sum += v_mass
            W_sum += w_mass
            P_sum += P_tot
            Z_sum += z_mass
            HGT_sum += hgt
            count += 1

T_mean = T_sum / count
V_mean = V_sum / count
W_mean = W_sum / count
P_mean = P_sum / count
Z_mean = Z_sum / count
HGT_mean = HGT_sum / count

print(f"[INFO] 共平均 {count} 个时次")
print(f"[INFO] T 范围：{T_mean.min():.1f} ~ {T_mean.max():.1f} K")

# =========================================================
# 4. 沿 lon_ref 取经向剖面
# =========================================================
nz, n_sn, n_we = T_mean.shape
lon_1d = XLONG_arr[0, :]

i_lon = np.argmin(np.abs(lon_1d - args.lon_ref))
print(f"[INFO] 取经度列 idx = {i_lon}，实际经度 = {lon_1d[i_lon]:.2f}°E")

T_sec = T_mean[:, :, i_lon]
V_sec = V_mean[:, :, i_lon]
W_sec = W_mean[:, :, i_lon]
P_sec = P_mean[:, :, i_lon]
Z_sec = Z_mean[:, :, i_lon] / 1000.0      # km
HGT_sec = HGT_mean[:, i_lon] / 1000.0     # km

lat_1d = XLAT_arr[:, i_lon]

# =========================================================
# 5. 插值到统一高度网格
# =========================================================
alt_uniform = np.linspace(-5, 85, 181)  # km
T_uni = np.full((len(alt_uniform), n_sn), np.nan)
V_uni = np.full_like(T_uni, np.nan)
W_uni = np.full_like(T_uni, np.nan)
P_uni = np.full_like(T_uni, np.nan)

for j in range(n_sn):
    z_col = Z_sec[:, j]
    valid = np.isfinite(z_col) & (z_col > -10) & (z_col < 100)

    if valid.sum() < 3:
        continue

    sort_idx = np.argsort(z_col[valid])
    zv = z_col[valid][sort_idx]

    T_uni[:, j] = np.interp(
        alt_uniform, zv, T_sec[valid, j][sort_idx],
        left=np.nan, right=np.nan
    )
    V_uni[:, j] = np.interp(
        alt_uniform, zv, V_sec[valid, j][sort_idx],
        left=np.nan, right=np.nan
    )
    W_uni[:, j] = np.interp(
        alt_uniform, zv, W_sec[valid, j][sort_idx],
        left=np.nan, right=np.nan
    )
    P_uni[:, j] = np.interp(
        alt_uniform, zv, np.log(P_sec[valid, j][sort_idx]),
        left=np.nan, right=np.nan
    )

P_uni = np.exp(P_uni)

# =========================================================
# 6. 读取 DEM 剖面
# =========================================================
dem_lat = None
dem_alt = None

if os.path.exists(args.dem):
    try:
        import rasterio

        with rasterio.open(args.dem) as src:
            dem_lats = np.linspace(-90, 90, 500)
            dem_lons = np.full_like(dem_lats, args.lon_ref)

            # rasterio.sample 需要 (x, y)，通常即 (lon, lat)
            coords = [(lo, la) for lo, la in zip(dem_lons, dem_lats)]
            vals = list(src.sample(coords, indexes=1))

            dem_alt = np.array([v[0] for v in vals], dtype=float) / 1000.0
            dem_lat = dem_lats

            nodata = src.nodata
            if nodata is not None:
                dem_alt[dem_alt == nodata / 1000.0] = np.nan

        print(
            f"[INFO] DEM 读取成功，高度范围："
            f"{np.nanmin(dem_alt):.2f} ~ {np.nanmax(dem_alt):.2f} km"
        )
    except Exception as e:
        print(f"[WARN] DEM 读取失败：{e}，使用 WRF HGT 代替")
else:
    print(f"[WARN] DEM 文件不存在，使用 WRF HGT 代替")

if dem_lat is None or dem_alt is None:
    dem_lat = lat_1d
    dem_alt = HGT_sec


def build_topography_profile(lat_1d, hgt_sec, dem_lat=None, dem_alt=None,
                             ybase=-5.0, use_wrf_hgt=True):
    """
    生成用于绘图/遮罩的地形剖面（单位 km）

    推荐 use_wrf_hgt=True：
    - 剖面遮罩和填充优先使用 WRF 的 HGT
    - 避免外部 DEM 与 WRF 垂直基准不一致
    """
    if use_wrf_hgt or dem_lat is None or dem_alt is None:
        topo_km = np.array(hgt_sec, dtype=float).copy()
    else:
        # 若强行使用 DEM，则先插值到 WRF 纬度
        topo_km = np.interp(lat_1d, dem_lat, dem_alt)

        # 用 HGT 做整体平移对齐，减小 DEM/WRF 垂直基准差异
        valid = np.isfinite(topo_km) & np.isfinite(hgt_sec)
        if valid.sum() >= 2:
            offset = np.nanmedian(hgt_sec[valid] - topo_km[valid])
            topo_km = topo_km + offset
        else:
            topo_km = np.array(hgt_sec, dtype=float).copy()

    # 补 NaN，避免 fill_between/多边形断裂
    nan_mask = ~np.isfinite(topo_km)
    if nan_mask.any():
        good = np.where(~nan_mask)[0]
        if good.size >= 2:
            topo_km[nan_mask] = np.interp(np.where(nan_mask)[0], good, topo_km[good])
        elif good.size == 1:
            topo_km[nan_mask] = topo_km[good[0]]
        else:
            topo_km[:] = ybase

    # 不允许低于图框下边界
    topo_km = np.clip(topo_km, ybase, None)
    return topo_km


def add_terrain_patch(ax, lat_1d, topo_km, ybase=-5.0,
                      facecolor="#AAAAAA", edgecolor="black",
                      linewidth=0.8, zorder=10):
    """
    手动构造封闭多边形：
    地形线（左->右）+ 底边（右->左）
    """
    poly_x = np.concatenate([lat_1d, lat_1d[::-1]])
    poly_y = np.concatenate([topo_km, np.full_like(topo_km, ybase)[::-1]])

    patch = Polygon(
        np.column_stack([poly_x, poly_y]),
        closed=True,
        facecolor=facecolor,
        edgecolor="none",
        zorder=zorder
    )
    ax.add_patch(patch)

    # 单独画地形轮廓线
    ax.plot(
        lat_1d, topo_km,
        color=edgecolor,
        linewidth=linewidth,
        zorder=zorder + 0.1
    )


# =========================================================
# 7. 计算右轴等压线对应高度
# =========================================================
mid_j = n_sn // 2
p_col_mid = P_uni[:, mid_j]
z_col_mid = alt_uniform

plev_alts = []
for plev in PLEVS_PA:
    valid = np.isfinite(p_col_mid)
    if valid.sum() < 3:
        plev_alts.append(np.nan)
        continue

    logP_v = np.log(p_col_mid[valid])
    z_v = z_col_mid[valid]
    sort_i = np.argsort(logP_v)[::-1]

    alt_v = np.interp(
        np.log(plev),
        logP_v[sort_i],
        z_v[sort_i],
        left=np.nan,
        right=np.nan
    )
    plev_alts.append(alt_v)

plev_alts = np.array(plev_alts)
print(f"[INFO] 等压线高度对应：{list(zip(PLEVS_PA, np.round(plev_alts, 1)))}")

# =========================================================
# 8. 图一：风场剖面
# =========================================================
print("\n[INFO] 绘制图一（风场剖面）...")

fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

V_max = max(abs(np.nanmax(V_uni)), abs(np.nanmin(V_uni)))
V_max = max(V_max, 10.0)
norm_v = mcolors.TwoSlopeNorm(vmin=-V_max, vcenter=0, vmax=V_max)

cf = ax1.contourf(
    lat_1d, alt_uniform, V_uni,
    levels=30, cmap="RdBu_r", norm=norm_v,
    extend="both", alpha=0.85
)

# 等压线
for plev, alt_lev in zip(PLEVS_PA, plev_alts):
    if np.isfinite(alt_lev) and -5 <= alt_lev <= 85:
        ax1.axhline(alt_lev, color="gray", lw=0.3, ls="-", zorder=2)

# 稀疏抽样画风矢量
skip_lat = max(1, n_sn // 30)
skip_alt = max(1, len(alt_uniform) // 25)

Lq = lat_1d[::skip_lat]
Aq = alt_uniform[::skip_alt]
Vq = V_uni[::skip_alt, ::skip_lat]
Wq = W_uni[::skip_alt, ::skip_lat] * 50.0

# 地形基准高度
terrain_base = -5.0

# 推荐：剖面绘图和遮罩优先使用 HGT_sec（WRF 自己坐标系）
# 如果你以后想试“保留 DEM 形状，但做整体高程对齐”，只要把最后那个参数改成 use_wrf_hgt=False
topo_plot = build_topography_profile(
    lat_1d=lat_1d,
    hgt_sec=HGT_sec,
    dem_lat=dem_lat,
    dem_alt=dem_alt,
    ybase=terrain_base,
    use_wrf_hgt=True
)


for jj, la in enumerate(Lq):
    j_orig = np.argmin(np.abs(lat_1d - la))
    """    
        hgt_here = (
        dem_alt[np.argmin(np.abs(dem_lat - la))]
        if dem_lat is not None else HGT_sec[j_orig]
    )
    """
    hgt_here = topo_plot[j_orig]
    for ii, al in enumerate(Aq):
        if al < hgt_here:
            Vq[ii, jj] = np.nan
            Wq[ii, jj] = np.nan

Lgrid, Agrid = np.meshgrid(Lq, Aq)

q = ax1.quiver(
    Lgrid, Agrid, Vq, Wq,
    color="black", scale=200, width=0.002,
    headwidth=4, headlength=5,
    minshaft=1.5, alpha=0.8, zorder=4
)

ref_speed = 10.0
ax1.quiverkey(
    q, X=0.97, Y=1.02, U=ref_speed,
    label=f"{ref_speed:.0f} m/s",
    labelpos="N", fontproperties={"size": 7},
    coordinates="axes"
)

ax1.axvline(x=0, color="gray", lw=0.5, ls="--", zorder=5)


# 地形填充
topo_alt = np.interp(lat_1d, dem_lat, dem_alt)

# 只对有效的 DEM 采样点绘图
valid_topo = np.isfinite(topo_alt)




"""
# 填充地形色
ax1.fill_between(
    lat_1d, terrain_base, topo_alt, where=valid_topo,
    color="#AAAAAA", alpha=1, zorder=10
)

# 绘图
ax1.plot(lat_1d, topo_alt,
         color="black", linewidth=0.8, zorder=10)
"""
# 地形封闭填充 + 地形轮廓线
add_terrain_patch(
    ax1,
    lat_1d,
    topo_plot,
    ybase=terrain_base,
    facecolor="#AAAAAA",
    edgecolor="black",
    linewidth=0.8,
    zorder=10
)


ax1.set_xlim(-90, 90)
ax1.set_ylim(-5, 85)
ax1.set_xticks(np.arange(-75, 76, 15))
ax1.set_yticks(ALT_TICKS)
ax1.set_xlabel("Latitude (deg)", fontsize=10)
ax1.set_ylabel("Altitude (km)", fontsize=10)
ax1.tick_params(direction="in", which="both", top=True, right=False)
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))

ax1_r = ax1.twinx()
ax1_r.set_ylim(-5, 85)
ax1_r.set_yticks([a for a in plev_alts if np.isfinite(a) and -5 <= a <= 85])
ax1_r.set_yticklabels(
    [
        f"{int(p) if p >= 1 else p}"
        for p, a in zip(PLEVS_PA, plev_alts)
        if np.isfinite(a) and -5 <= a <= 85
    ],
    fontsize=7
)
ax1_r.tick_params(direction="in", which="both")
ax1_r.set_ylabel("Pressure (Pa)", fontsize=9, labelpad=8)

lon_label = f"D01 ({args.sol_label}, LST:{args.lst_label})"
ax1.set_title("V, W*50 (m/s)", loc="left", fontsize=9, fontweight="bold")
ax1.set_title(lon_label, loc="right", fontsize=8)

plt.tight_layout()
plt.savefig(args.out1, bbox_inches="tight", dpi=150)
print(f"[INFO] 图一已保存：{args.out1}")
plt.close()

# =========================================================
# 9. 图二：温度剖面
# =========================================================
print("\n[INFO] 绘制图二（温度剖面）...")

T_plot_min = np.floor(np.nanmin(T_uni) / 8) * 8
T_plot_max = np.ceil(np.nanmax(T_uni) / 8) * 8
T_plot_min = max(T_plot_min, 108)
T_plot_max = min(T_plot_max, 268)
levels_T = np.arange(T_plot_min, T_plot_max + 1, 8)
n_levs = len(levels_T) - 1

cmap_T = plt.get_cmap("RdBu_r", n_levs)
norm_T = BoundaryNorm(levels_T, ncolors=n_levs, clip=False)

fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)

cf2 = ax2.contourf(
    lat_1d, alt_uniform, T_uni,
    levels=levels_T, cmap=cmap_T, norm=norm_T,
    extend="both"
)

cs2 = ax2.contour(
    lat_1d, alt_uniform, T_uni,
    levels=levels_T, colors="black", linewidths=0.6,
    linestyles="-"
)
ax2.clabel(
    cs2, fmt="%d", fontsize=6.5, inline=True,
    inline_spacing=2, use_clabeltext=True
)

for plev, alt_lev in zip(PLEVS_PA, plev_alts):
    if np.isfinite(alt_lev) and -5 <= alt_lev <= 85:
        ax2.axhline(alt_lev, color="gray", lw=0.3, ls="-", zorder=2)

ax2.axvline(x=0, color="gray", lw=0.5, ls="--", zorder=5)


# 地形封闭填充 + 地形轮廓线
add_terrain_patch(
    ax2,
    lat_1d,
    topo_plot,
    ybase=terrain_base,
    facecolor="#AAAAAA",
    edgecolor="black",
    linewidth=0.8,
    zorder=10
)

"""
# 地形填充
topo_alt = np.interp(lat_1d, dem_lat, dem_alt)

# 只对有效的 DEM 采样点绘图
valid_topo = np.isfinite(topo_alt)

terrain_base = -5.0

# 填充地形色
ax2.fill_between(
    lat_1d, terrain_base, topo_alt, where=valid_topo,
    color="#AAAAAA", alpha=1, zorder=10
)

# 绘图
ax2.plot(lat_1d, topo_alt,
         color="black", linewidth=0.8, zorder=10)
"""

ax2.set_xlim(-90, 90)
ax2.set_ylim(-5, 85)
ax2.set_xticks(np.arange(-75, 76, 15))
ax2.set_yticks(ALT_TICKS)
ax2.set_xlabel("Latitude (deg)", fontsize=10)
ax2.set_ylabel("Altitude (km)", fontsize=10)
ax2.tick_params(direction="in", which="both", top=True, right=False)
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(5))

ax2_r = ax2.twinx()
ax2_r.set_ylim(-5, 85)
ax2_r.set_yticks([a for a in plev_alts if np.isfinite(a) and -5 <= a <= 85])
ax2_r.set_yticklabels(
    [
        f"{int(p) if p >= 1 else p}"
        for p, a in zip(PLEVS_PA, plev_alts)
        if np.isfinite(a) and -5 <= a <= 85
    ],
    fontsize=7
)
ax2_r.tick_params(direction="in", which="both")
ax2_r.set_ylabel("Pressure (Pa)", fontsize=9, labelpad=8)

T_max_val = np.nanmax(T_uni)
T_min_val = np.nanmin(T_uni)

cbar_ax = fig2.add_axes([0.15, -0.06, 0.70, 0.030])
cb2 = fig2.colorbar(
    cf2, cax=cbar_ax, orientation="horizontal",
    ticks=levels_T[::2]
)
cb2.set_label("(K)", fontsize=9, labelpad=2)
cb2.ax.tick_params(labelsize=7, direction="in")

ax2.set_title("Temperature (K)", loc="left", fontsize=9, fontweight="bold")
ax2.set_title(f"D01 ({args.sol_label}, LST:{args.lst_label})", loc="right", fontsize=8)
ax2.text(
    0.01, 0.97,
    f"Tmax:{T_max_val:.2f}, Tmin:{T_min_val:.2f}",
    transform=ax2.transAxes, fontsize=7.5,
    va="top", ha="left"
)

plt.savefig(args.out2, bbox_inches="tight", dpi=150)
print(f"[INFO] 图二已保存：{args.out2}")
plt.close()

print("[DONE]")