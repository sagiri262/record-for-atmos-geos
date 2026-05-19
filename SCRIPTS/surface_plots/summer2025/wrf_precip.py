"""
wrf_03_precip.py — WRF 近地面累计降水图
=========================================
变量：RAINC（对流）+ RAINNC（网格）= 总累计降水（mm）
仅绘制 ≥ 0.1 mm 的区域。

用法示例：
  python wrf_precip.py --file "wrfout_d01_*"
  python wrf_precip.py --file "wrfout_d01_2026-04-12_00:00:00" \\
      --out rain.png --thresh 1.0
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import (
    add_common_args,
    open_first_ncfile,
    parse_time_from_filename,
    get_latlon,
)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="绘制 WRF 近地面累计降水图（RAINC + RAINNC）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    parser.add_argument(
        "--thresh", type=float, default=0.1,
        metavar="mm",
        help="降水显示阈值（mm），低于此值不着色"
    )
    parser.add_argument(
        "--levels",
        default="0.1,1,5,10,25,50,100,150,200,300",
        metavar="mm,...",
        help="降水分级色阶（mm），用逗号分隔"
    )
    return parser


# ---------------------------------------------------------------------------
# 核心绘图
# ---------------------------------------------------------------------------

# 默认色阶颜色（9 段，对应 10 个分级边界）
DEFAULT_COLORS = [
    "#d4f0ff",   # 0.1~1
    "#a3d9f5",   # 1~5
    "#56b4e9",   # 5~10
    "#009e73",   # 10~25
    "#f0e442",   # 25~50
    "#e69f00",   # 50~100
    "#d55e00",   # 100~150
    "#cc79a7",   # 150~200
    "#7b2d8b",   # 200~300+
]


def main():
    args   = build_parser().parse_args()
    nc, first_file = open_first_ncfile(args.file)
    time_str = parse_time_from_filename(first_file)

    # 解析色阶
    p_levels = [float(x.strip()) for x in args.levels.split(",")]
    n_seg    = len(p_levels) - 1
    # 颜色数量自动匹配分级数
    colors   = DEFAULT_COLORS[:n_seg] if n_seg <= len(DEFAULT_COLORS) \
               else DEFAULT_COLORS + ["#400040"] * (n_seg - len(DEFAULT_COLORS))

    lat, lon = get_latlon(nc)
    rainc    = nc.variables["RAINC"][0]
    rainnc   = nc.variables["RAINNC"][0]
    rain     = rainc + rainnc
    rain_masked = np.ma.masked_less(rain, args.thresh)

    cmap_p = plt.matplotlib.colors.ListedColormap(colors)
    norm_p = BoundaryNorm(p_levels, ncolors=len(colors), clip=False)

    extent = [float(lon.min()), float(lon.max()),
              float(lat.min()), float(lat.max())]

    fig = plt.figure(figsize=(10, 7))
    proj = ccrs.PlateCarree()
    ax   = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.8, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.6, edgecolor="#333333", zorder=5)
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#d6eaf8", zorder=0)

    cf = ax.contourf(lon, lat, rain_masked,
                     levels=p_levels, cmap=cmap_p, norm=norm_p,
                     transform=ccrs.PlateCarree(), zorder=2, extend="max")

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.85)
    cbar.set_label("Accumulated Precipitation (mm)", fontsize=9)
    cbar.set_ticks(p_levels)
    cbar.set_ticklabels([str(v) for v in p_levels], fontsize=7)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}

    ax.set_title(f"Accumulated Precipitation (RAINC + RAINNC)  |  {time_str}",
                 fontsize=11, fontweight="bold", pad=6)

    out = args.out or "wrf_03_precip.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    nc.close()
    print(f"[✓] 降水图已保存：{os.path.abspath(out)}")


if __name__ == "__main__":
    main()