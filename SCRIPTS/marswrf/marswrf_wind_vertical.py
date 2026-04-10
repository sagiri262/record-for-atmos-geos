"""
MarsWRF 经向剖面图（Python 版）
=====================================
图一：V+W*50 合成风矢量 + 地形剖面（110°E 经线）
图二：温度等温线填色图（RdBu 色表，8K 间隔）
 
变量对应：
  T       : 扰动位温 (K)，需加 T00=240K → 真实温度 θ = T + T00
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
 
使用方法：
python marswrf_section_python.py \
  --file "wrfout_d01_*" \
  --dem Mars_MGS_MOLA_DEM_mosaic_global_463m.tif \
  --lon_ref 110 \
  --sol_label "Sol:85-90" --lst_label "13.74"
"""


import argparse
import glob
import os
import sys
 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")

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


parser = argparse.ArgumentParser()
# 输入 wrfout 文件
parser.add_argument("--file", "-f", default="wrfout_d01_*",
                    help="WRF 输出文件（通配符）")
# 
parser.add_argument("--dem", default="Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",
                    help="WRF 输出文件（通配符）")
# 经向剖面
parser.add_argument("--lon_ref", type=float, default=110.0,
                    help="经向剖面经度（°E），默认 110")
# 标题中时间标签
parser.add_argument("--sol_label", default="Sol: 85-90",
                    help="标题中的时间标签")
# LST标签
parser.add_argument("--lst_label", default="13.74",
                    help="标题中 LST 标签")

parser.add_argument("--out1", default="fig1_wind_section.png")
parser.add_argument("--out2", default="fig2_temp_section.png")

args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════
#  常数
# ══════════════════════════════════════════════════════════════════════
g_mars  = 3.72          # 火星重力加速度 m/s²
R_cp    = 0.2857        # R/Cp (CO2 大气 ≈ 0.2381，若用地球值 0.2857 看实际配置)
P_ref   = 610.0         # 参考气压 Pa（火星 MOLA 零点气压）
T00     = 240.0         # WRF T00（基态温度，Pa），通常从文件全局属性读取
 
# 等压线（Pa）→ 右轴标注，从下到上
PLEVS_PA = np.array([610, 400, 200, 100, 50, 10, 5, 1, 0.1]) * 1.0
# 对应近似高度（km）——仅用于右轴刻度映射，实际高度从数据计算
ALT_TICKS = [-5, 5, 15, 25, 35, 45, 55, 65, 75, 85]   # km，左轴


"""
读取WRF文件，按时次平均
"""
files = sorted(glob.glob(args.files))

if not files:
    sys.exit(f"[ERROR]未找到文件: {args.file}")

print(f"[INFO] 找到 {len(files)} 个文件")

# 从第一个文件处获得 netcdf 维度，然后累加
initialized = False
T_sum = V_sum = W_sum = P_sum = Z_sum = None
HGT_sum = XLAT_arr = XLONG_arr = None
count = 0

for fpath in files:
    print(f"读取: {os.path.basename(fpath)}")
    with Dataset(fpath, "r") as ds:
        if hasattr(ds, "T00"):
            T00 = float(ds.T00)
        
        """
        ds.variables是一个“变量字典”，里面保存文件中的所有变量。
        等价于 print(ds.variables.keys())
        输出： dict_keys(['Times', 'XLAT', 'XLONG', 'T2', ...])
        所以   ds.variables["XLAT"]取到的是名为 XLAT 的变量对象
        不是普通 Python list，而是一个 netCDF Variable 对象。
        """

        # 读取坐标
        if XLAT_arr is None:
            """
            (SN, WE)表示(南北向格点数，东西向格点数)
            数据结构 float XLAT(Time, south_north, west_east) ;
            0 —— 取第 0 个时间层
            : —— 取所有 south_north
            : —— 取所有 west_east
            所以变成 (south_north, west_east)
            """
            XLAT_arr  = ds.variables["XLAT"][0, :, :]
            XLONG_arr = ds.variables["XLONG"][0, :, :]
        
        # shape[0] 就是取这个变量第 0 维的长度，也就是时间层数。
        ntimes = ds.variables["T"].shape[0]

        for t in range(ntimes):
            # 气压
            P_tot = (ds.variables["P"][t] + 
                    ds.variables["PB"][t])
            
            # 位势高度
            # ds.variables["P"][t, :, :, :] 表示依次取第t个时次，下面的写法和我给出的写法是等价的
            # 取时次，其他全部保留
            ph = ds.variables["PH"][t]
            phb = ds.variables["PHB"][t]
            z_stag = (ph + phb) / g_mars
            z_mass = 0.5 * (z_stag[:-1] + z_stag[1:])

            # 温度（单位：开尔文）
            # wrfout是扰动位温， 真实位温 theta = T + T00
            # 物理温度 T_phys = theta * (P/P_ref)^(R/Cp)
            theta = ds.variables["T"][t] + T00
            T_phys = theta * (P_tot / P_ref) ** R_cp

            # V 风
            # 变量数据结构 float V(Time, bottom_top, south_north_stag, west_east)
            v_stag = ds.variables["V"][t]
            v_mass = 0.5 * (v_stag[:, :-1, :] + 
                            v_stag[:, 1:, :])
            
            # W 风，变量结构如V风差不多
            w_stag = ds.variables["W"][t]
            w_mass = 0.5 * (w_stag[:, :-1, :] + 
                            w_stag[:, 1:, :])
            

            # 地形
            hgt = ds.variables["HGT"][t]

            if not initialized:
                shp         = T_phys.shape
                T_sum       = np.zeros(shp)
                V_sum       = np.zeros(shp)
                W_sum       = np.zeros(shp)
                P_sum       = np.zeros(shp)
                Z_sum       = np.zeros(shp)
                HGT_sum     = np.zeros(hgt.shape)
                initialized =True

            T_sum   += T_phys
            V_sum   += v_mass
            W_sum   += w_mass
            P_sum   += P_tot
            Z_sum   += z_mass
            HGT_sum += hgt
            count   += 1

T_mean   = T_sum   / count
V_mean   = V_sum   / count
W_mean   = W_sum   / count
P_mean   = P_sum   / count
Z_mean   = Z_sum   / count
HGT_mean = HGT_sum / count

print(f"[INFO] 共平均 {count} 时次")
print(f"[INFO] T 范围：{T_mean.min():.1f} ~ {T_mean.max():.1f} K")


"""
沿 lon_ref 取经向剖面东经110°
"""
nz, n_sn, n_we = T_mean.shape
lon_1d         = XLONG_arr[0, :]

i_lon = np.argmin(np.abs(lon_1d - args.lon_ref))
print(f"[INFO] 取经度列 idx={i_lon}，实际经度={lon_1d[i_lon]:.2f}°E")

# 列出各剖面变量
T_sec   = T_mean[:, :, i_lon]
V_sec   = V_mean[:, :, i_lon]
W_sec   = W_mean[:, :, i_lon]
P_sec   = P_mean[:, :, i_lon]
# 高度单位为 km
Z_sec   = Z_mean[:, :, i_lon]   / 1000.0
HGT_sec = HGT_mean[:, :, i_lon] / 1000.0

lat_1d = XLAT_arr[:, i_lon]

# 建立统一高度网格
# 使用等距插值
alt_uniform = np.linspace(-5, 85, 181)
T_uni = np.full((len(alt_uniform), n_sn), np.nan)
V_uni = np.full_like(T_uni, np.nan)
W_uni = np.full_like(T_uni, np.nan)
P_uni = np.full_like(T_uni, np.nan)

"""
把每一个纬向列 j 上，原本“不等距”的垂直高度数据
插值到统一的等距高度网格 alt_uniform 上。
作用：这样后面画剖面图、等高线图时，各列都对应同一套高度坐标，二维数组就规整了。
"""
for j in range(n_sn):
    # 取这一列的原始高度
    z_col = Z_sec[:, j]

    # 对每个变量在垂直方向上插值到均匀高度网格
    # 筛选出可用的高度层
    # np.isfinite(z_col)：不是 NaN，不是 inf
    valid = np.isfinite(z_col) & (z_col > -10) & (z_col < 100)
    # 最后valid输出[True, False, True, ...] 这样的布尔数组
    if valid.sum() < 3:
        continue

    sort_idx = np.argsort(z_col[valid])
    # 得到排序后的高度数组
    zv = z_col[valid][sort_idx]

    """
    已知第 j 列上，温度在若干原始高度 zv 上的值
    现在要算出它在统一高度 alt_uniform 上的值

    np.interp(x_new, x_old, y_old)

    含义是：
    x_new：新坐标，这里是统一高度 —— alt_uniform
    x_old：旧坐标，这里是原始高度 —— zv
    y_old：旧坐标上的变量值
    返回结果就是变量在 x_new 上的插值结果。
    """
    T_uni[:, j] = np.interp(alt_uniform, zv, T_sec[valid, j][sort_idx],
                             left=np.nan, right=np.nan)
    V_uni[:, j] = np.interp(alt_uniform, zv, V_sec[valid, j][sort_idx],
                             left=np.nan, right=np.nan)
    W_uni[:, j] = np.interp(alt_uniform, zv, W_sec[valid, j][sort_idx],
                             left=np.nan, right=np.nan)
    # 气压随高度通常近似指数衰减，不是线性变化。
    P_uni[:, j] = np.interp(alt_uniform, zv, np.log(P_sec[valid, j][sort_idx]),
                             left=np.nan, right=np.nan)
    """
    上面的代码做的事：
    固定一列 j，用这列的原始高度 z_col，把温度、风、气压都插值到统一高度 alt_uniform
    """

# 还原气压值
# 因为前面只是借助 log(P) 来做插值，并不是想把气压永久变成对数气压。
P_uni = np.exp(P_uni)


# 读取DEM剖面
dem_lat = None
dem_lon = None

if os.path.exists(args.dem):
    try:
        import rasterio
        with rasterio.open(args.dem) as src:
            # 构建经线上的采样点
            dem_lats = np.linspace(-90, 90, 500)
            dem_lons = np.full_like(dem_lats, args.lon_ref)
            # 数组格式：(lon, lat)
            coords = [(lo, la) for lo, la in zip(dem_lons, dem_lats)]
            vals = list(src.sample(coords, indexes=1))
            # 高度单位 km
            dem_alt = np.array([v[0] for v in vals], dtype=float) / 1000.0
            dem_lat = dem_lats
            nodata = src.nodata
            if nodata is not None:
                dem_alt[dem_alt == nodata / 1000.0] = np.nan
        print(f"[INFO] DEM 读取成功，高度范围：{np.nanmin(dem_alt):.2f} ~ {np.nanmax(dem_alt):.2f} km")
    except Exception as e:
        print(f"[WARN] DEM 读取失败：{e}，使用 WRF HGT 代替")
else:
    print(f"[WARN] DEM 文件不存在，使用 WRF HGT 代替")

if dem_lat is None:
    dem_lat = lat_1d
    dem_alt = HGT_sec


# 计算右轴等压线对应高度
def pressure_to_alt(p_target_pa, P_col, Z_col):
    """
    找到接近 p_target 的高度
    """
    logP = np.log(P_col)
    logT = np.log(p_target_pa)

    # 插值，注意气压是随高度上升减少，注意方向
    # 注意方向，从低层到高层，logP 值是从大到小
    idx = np.argsort(logP)[::-1]

    # 注意有效值范围，不在有效范围直接范围无穷大
    if logT < logP[idx].min() or logT > logP[idx].max():
        return np.nan
    return np.interp(logT, logP[idx], Z_col[idx])

# 取纬度中间， 列出压力高度的对应关系
mid_j = n_sn // 2
p_col_mid = P_uni[:, mid_j]
z_col_mid = alt_uniform

plev_alts = []
for plev in PLEVS_PA:
    valid = np.isfinite(p_col_mid)
    # 不在有效范围内
    if valid.sum() < 3:
        plev_alts.append(np.nan)
        continue
    # 将对数化的压力值进行单调插值
    logP_v = np.log(p_col_mid[valid])
    z_v = z_col_mid[valid]
    sort_i = np.argsort(logP_v)[::-1]
    alt_v = np.interp(np.log(plev), logP_v[sort_i], z_v[sort_i],
                      left=np.nan, right=np.nan)
    plev_alts.append(alt_v)

plev_alts = np.array(plev_alts)
print(f"[INFO] 等压线高度对应：{list(zip(PLEVS_PA, np.round(plev_alts,1)))}")

# 构建网格
LAT2D, ALT2D = np.meshgrid(lat_1d, alt_uniform)   # (n_alt, n_sn)


"""
绘制图一

"""

print("\n 【INFO】绘制图一")
fig1, ax1 = plt.subplot(figsize=(8,6), dpi=300)

# 背景色
# 正=北风/蓝，负=南风/红
V_max = max(abs(np.nanmax(V_uni)), abs(np.nanmin(V_uni)))
V_max = max(V_max, 10.0)
norm_v = mcolors.TwoSlopeNorm(vmin=-V_max, vcenter=0, vmax=V_max)

cf = ax1.contoutf(lat_1d, alt_uniform, V_uni,
                  levels=30, cmap="RdBu_r", norm=norm_v,
                  extend="both", alpha=0.85)

# 绘制等压线
# （灰色，0.3 线宽）
# 在等高线坐标系中，等压线对应等高度线（近似）
for plev, alt_lev in zip(PLEVS_PA, plev_alts):
    if np.isfinite(alt_lev) and -5 <= alt_lev <=85:
        ax1.axhline(alt_lev, color="gray", lw=0.3, ls="-", zorder=2)

# 
skip_lat = max(1, n_sn // 30)
skip_alt = max(1, len(alt_uniform) // 25)
Lq = lat_1d[::skip_lat]
Aq = alt_uniform[::skip_alt]
# V, W *50
Vq = V_uni[::skip_alt, ::skip_lat] 
Wq = W_uni[::skip_alt, ::skip_lat] * 50

for jj, la in enumerate(Lq):
    j_orig = np.argmin(np.abs(lat_1d - la))
    hgt_here = dem_alt[np.argmin(np.abs(dem_lat - la))] if dem_lat is not None else HGT_sec[j_orig]
    for ii, al in enumerate(Aq):
        if al < hgt_here:
            Vq[ii, jj] = np.nan
            Wq[ii, jj] = np.nan

Lgrid, Agrid = np.meshgrid(Lq, Aq)
# 合成风速
speed = np.sqrt(Vq ** 2 + wq ** 2)


q = ax1.quiver(Lgrid, Agrid, Vq, Wq,
               color="black", scale=200, width=0.02,
               headwidth=4, headlength=5,
               minshaft=1.5, alpha=0.8, zorder=4)

# 画箭头比例尺，单位 m/s
ref_speed = 10.0
ax1.quiverkey(q, X=0.97, Y=1.02, U=ref_speed,
              label=f"{ref_speed:.0f}\m/s",
              labelpos="N", fontproperties={"size": 7},
              coordinates="axes")

# 赤道垂直虚线
ax1.axvline(x=0, color="gray", lw=0.5, ls="--", zorder=5)
 
# 地形填充
topo_alt = np.interp(lat_1d, dem_lat, dem_alt)

# 只对有效的 DEM 采样点绘图
valid_topo = np.isfinite(topo_alt)

terrain_base = -5.0

# 填充地形色
ax1.fill_between(
    lat_1d, terrain_base, topo_alt, where=valid_topo,
    color="#AAAAAA", alpha=1, zorder=3
)

# 绘图
ax1.plot(lat_1d, topo_alt,
         color="black", linewidth=0.8, zorder=4)


# 坐标轴
ax1.set_xlim(-90, 90)
ax1.set_ylim(-5, 85)
ax1.set_xticks(np.arange(-72.5, 78, 15))
ax1.set_yticks(ALT_TICKS)
ax1.set_xlabel("Latitude (deg)", fontsize=10)
ax1.set_ylabel("Altitude (km)", fontsize=10)
ax1.tick_params(directions="in", which="both",
                top=True, right=False)
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))

# 等压标注
ax1_r = ax1.twinx()
ax1_r.set_ylim(-5, 85)
ax1_r.set_yticks([a for a in plev_alts if np.isfinite(a) and -5 <= a <= 85])
ax1_r.set_yticklabels(
    [f"{int(p) if p >= 1 else p}" for p, a in zip(PLEVS_PA, plev_alts)
     if np.isfinite(a) and -5 <= a <= 85],
    fontsize=7
)
ax1_r.tick_params(direction="in", which="both")
ax1_r.set_ylabel("Pressure (Pa)", fontsize=9, labelpad=8)
 
# 标题与标注
lon_label = f"D01 ({args.sol_label}, LST:{args.lst_label})"
ax1.set_title(f"V,W*50 (m/s)", loc="left", fontsize=9, fontweight="bold")
ax1.set_title(lon_label, loc="right", fontsize=8)
 
plt.tight_layout()
plt.savefig(args.out1, bbox_inches="tight", dpi=150)
print(f"[INFO] 图一已保存：{args.out1}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
#  ██████  图二：温度等温线填色图  ██████
# ══════════════════════════════════════════════════════════════════════
print("\n[INFO] 绘制图二（温度剖面）...")
 
# 色标设置：RdBu 反转（冷蓝-暖红），8K 间隔
T_plot_min = np.floor(np.nanmin(T_uni) / 8) * 8
T_plot_max = np.ceil (np.nanmax(T_uni) / 8) * 8
T_plot_min = max(T_plot_min, 108)   # 与例图对应
T_plot_max = min(T_plot_max, 268)
levels_T   = np.arange(T_plot_min, T_plot_max + 1, 8)
n_levs     = len(levels_T) - 1
 
# 色表选择：RdBu 反转（低温蓝，高温红）
cmap_T = plt.get_cmap("RdBu_r", n_levs)
norm_T = BoundaryNorm(levels_T, ncolors=n_levs, clip=False)
 
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)
 
# ── 6.1 等温线填色（contourf）
cf2 = ax2.contourf(lat_1d, alt_uniform, T_uni,
                   levels=levels_T, cmap=cmap_T, norm=norm_T,
                   extend="both")
 
# ── 6.2 等温线标注（间隔 8K，黑色线）
cs2 = ax2.contour(lat_1d, alt_uniform, T_uni,
                  levels=levels_T, colors="black", linewidths=0.6,
                  linestyles="-")
ax2.clabel(cs2, fmt="%d", fontsize=6.5, inline=True,
           inline_spacing=2, use_clabeltext=True)
 
# ── 6.3 等压线（灰色，0.3 线宽）
for plev, alt_lev in zip(PLEVS_PA, plev_alts):
    if np.isfinite(alt_lev) and -5 <= alt_lev <= 85:
        ax2.axhline(alt_lev, color="gray", lw=0.3, ls="-", zorder=2)
 
# ── 6.4 赤道垂直虚线
ax2.axvline(x=0, color="gray", lw=0.5, ls="--", zorder=5)
 
# 地形填充
topo_alt = np.interp(lat_1d, dem_lat, dem_alt)

# 只对有效的 DEM 采样点绘图
valid_topo = np.isfinite(topo_alt)

terrain_base = -5.0

# 填充地形色
ax2.fill_between(
    lat_1d, terrain_base, topo_alt, where=valid_topo,
    color="#AAAAAA", alpha=1, zorder=3
)

# 绘图
ax2.plot(lat_1d, topo_alt,
         color="black", linewidth=0.8, zorder=4)
 
# ── 6.6 坐标轴
ax2.set_xlim(-90, 90)
ax2.set_ylim(-5, 85)
ax2.set_xticks(np.arange(-72.5, 78, 15))
ax2.set_yticks(ALT_TICKS)
ax2.set_xlabel("Latitude (deg)", fontsize=10)
ax2.set_ylabel("Altitude (Km)", fontsize=10)
ax2.tick_params(direction="in", which="both", top=True, right=False)
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(5))
 
# ── 6.7 右轴：等压标注
ax2_r = ax2.twinx()
ax2_r.set_ylim(-5, 85)
ax2_r.set_yticks([a for a in plev_alts if np.isfinite(a) and -5 <= a <= 85])
ax2_r.set_yticklabels(
    [f"{int(p) if p >= 1 else p}" for p, a in zip(PLEVS_PA, plev_alts)
     if np.isfinite(a) and -5 <= a <= 85],
    fontsize=7
)
ax2_r.tick_params(direction="in", which="both")
ax2_r.set_ylabel("Pressure (Pa)", fontsize=9, labelpad=8)
 
# ── 6.8 Colorbar（图框右下角外侧，水平）
T_max_val = np.nanmax(T_uni)
T_min_val = np.nanmin(T_uni)
label_str  = f"Tmax:{T_max_val:.2f}, Tmin:{T_min_val:.2f}"
 
# 在图框外部右下角放置水平 colorbar
cbar_ax = fig2.add_axes([0.15, -0.06, 0.70, 0.030])   # [left, bottom, width, height]
cb2 = fig2.colorbar(cf2, cax=cbar_ax, orientation="horizontal",
                    ticks=levels_T[::2])
cb2.set_label("(K)", fontsize=9, labelpad=2)
cb2.ax.tick_params(labelsize=7, direction="in")
 
# ── 6.9 标题
ax2.set_title("Temperature (K),", loc="left", fontsize=9, fontweight="bold")
ax2.set_title(f"D01 ({args.sol_label}, LST:{args.lst_label})", loc="right", fontsize=8)
ax2.text(0.01, 0.97,
         f"Tmax:{T_max_val:.2f}, Tmin:{T_min_val:.2f}",
         transform=ax2.transAxes, fontsize=7.5,
         va="top", ha="left")
 
plt.savefig(args.out2, bbox_inches="tight", dpi=150)
print(f"[INFO] 图二已保存：{args.out2}")
plt.close()
print("[DONE]")






