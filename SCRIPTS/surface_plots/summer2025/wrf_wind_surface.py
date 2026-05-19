"""
wrf_01_wind.py — WRF 水平风场四合图
=====================================
子图布局：
  +-------------+---------------+
  |  近地面风场  |   800hPa风场  |
  +-------------+---------------+
  |  500hPa风场 |   100hPa风场  |
  +-------------+---------------+

用法示例：
  python wrf_wind.py --file "wrfout_d01_*"
  python wrf_wind.py --file "wrfout_d01_2026-04-12_00:00:00" --out my_wind.png --dpi 200
  python wrf_wind.py --file "wrfout_d01_*" --thin 8 --levels "850,500,200"

参数说明见 --help。
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import (
    add_common_args,
    open_first_ncfile,
    parse_time_from_filename,
    get_latlon, get_pressure_3d,
    destagger_u, destagger_v,
    interp_to_pressure_fast,
    make_map_axes,
)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="绘制 WRF 水平风场四合图（近地面 / 三个气压层）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    parser.add_argument(
        "--thin", type=int, default=6,
        metavar="N",
        help="风羽抽稀间隔（格点数），越大箭头越稀"
    )
    parser.add_argument(
        "--levels", default="800,500,100",
        metavar="hPa,hPa,hPa",
        help="三个气压层（hPa），用逗号分隔，从低到高，例如 800,500,100"
    )
    parser.add_argument(
        "--speed_max", type=float, default=34.0,
        metavar="m/s",
        help="风速填色上限（m/s），超过此值归入最深色"
    )
    parser.add_argument(
        "--speed_step", type=float, default=2.0,
        metavar="m/s",
        help="风速填色步长（m/s）"
    )
    return parser


# ---------------------------------------------------------------------------
# 核心绘图
# ---------------------------------------------------------------------------

def plot_wind_panel(ax, lon, lat, u, v, title: str,
                    speed_levels: np.ndarray, thin: int):
    """在单个子图中绘制风速填色 + 风羽"""
    spd = np.sqrt(u**2 + v**2)

    cf = ax.contourf(lon, lat, spd,
                     levels=speed_levels, cmap="YlOrRd", alpha=0.70,
                     transform=ccrs.PlateCarree(), zorder=2, extend="max")
    ax.contour(lon, lat, spd,
               levels=[10, 20, 30], colors="gray",
               linewidths=0.5, alpha=0.6,
               transform=ccrs.PlateCarree(), zorder=3)
    ax.barbs(lon[::thin, ::thin], lat[::thin, ::thin],
             u[::thin, ::thin],   v[::thin, ::thin],
             length=4, linewidth=0.6, color="navy",
             transform=ccrs.PlateCarree(), zorder=4)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    return cf


def main():
    args   = build_parser().parse_args()
    nc, first_file = open_first_ncfile(args.file)
    time_str = parse_time_from_filename(first_file)

    # 解析气压层参数
    ua_levels_hpa = [float(x.strip()) for x in args.levels.split(",")]
    if len(ua_levels_hpa) != 3:
        raise ValueError("--levels 需要恰好三个数值，例如 800,500,100")

    lat, lon = get_latlon(nc)
    pres3d   = get_pressure_3d(nc)
    extent   = [float(lon.min()), float(lon.max()),
                float(lat.min()), float(lat.max())]

    u3d = destagger_u(nc.variables["U"][0])
    v3d = destagger_v(nc.variables["V"][0])

    speed_levels = np.arange(0, args.speed_max + args.speed_step, args.speed_step)

    # 四个面板：近地面 + 三个气压层
    panel_titles = ["近地面 (Lowest Level)"] + [f"{int(h)} hPa" for h in ua_levels_hpa]
    panel_data   = [(u3d[0], v3d[0])] + [
        (interp_to_pressure_fast(u3d, pres3d, h * 100),
         interp_to_pressure_fast(v3d, pres3d, h * 100))
        for h in ua_levels_hpa
    ]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"WRF Horizontal Wind  |  {time_str}",
                 fontsize=13, fontweight="bold", y=0.99)

    cf_ref = None
    for pos, (title, (u, v)) in zip([221, 222, 223, 224],
                                     zip(panel_titles, panel_data)):
        ax     = make_map_axes(fig, pos, extent)
        cf_ref = plot_wind_panel(ax, lon, lat, u, v, title,
                                  speed_levels, args.thin)

    # 公共色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cb = fig.colorbar(cf_ref, cax=cbar_ax)
    cb.set_label("Wind Speed (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.05, right=0.91, top=0.96, bottom=0.04,
                        hspace=0.25, wspace=0.18)

    out = args.out or "wrf_01_wind.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    nc.close()
    print(f"[✓] 风场图已保存：{os.path.abspath(out)}")


if __name__ == "__main__":
    main()