"""
创建 cdsapi 的 API 密钥文件
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: 你的API密钥
EOF
chmod 600 ~/.cdsapirc

如何使用命令行下载 ERA5 的文件

python download_era5_for_wrf.py \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --area North West South East \
  --outdir PATH \
  --prefix 设置别名（可选）

"""


import argparse
from pathlib import Path
from datetime import datetime
import cdsapi


PRESSURE_LEVELS = [
    "1000", "975", "950", "925", "900", "875", "850",
    "825", "800", "775", "750", "700", "650", "600",
    "550", "500", "450", "400", "350", "300", "250",
    "225", "200", "175", "150", "125", "100", "70",
    "50", "30", "20", "10", "7", "5", "3", "2", "1"
]

TIMES = [f"{h:02d}:00" for h in range(24)]

PL_VARS = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

SL_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "land_sea_mask",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
    "skin_temperature",
    "snow_depth",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ERA5 pressure-level and single-level data for WRF/WPS"
    )
    parser.add_argument("--start", required=True, help="Start date, format: YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, format: YYYY-MM-DD")
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        required=True,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Area in CDS order: North West South East",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory, e.g. /home/zy/ERA5",
    )
    parser.add_argument(
        "--prefix",
        default="era5",
        help="Output filename prefix, default: era5",
    )
    return parser.parse_args()


def month_start(dt):
    return dt.replace(day=1)


def next_month(dt):
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)


def iter_month_ranges(start_dt, end_dt):
    current = month_start(start_dt)
    while current <= end_dt:
        y = current.year
        m = current.month

        chunk_start = max(start_dt, current)
        chunk_end = min(end_dt, next_month(current) - (next_month(current) - next_month(current)).replace(day=1))
        # 上面这句不好读，下面直接重算一次更清楚
        nm = next_month(current)
        chunk_end = min(end_dt, nm.replace(day=1) - (nm.replace(day=1) - current.replace(day=1)).replace(days=0) if False else end_dt)

        # 重新用更稳定的方法求该月最后一天
        if current.month == 12:
            month_end = current.replace(year=current.year + 1, month=1, day=1) - (current.replace(year=current.year + 1, month=1, day=1) - current.replace(year=current.year + 1, month=1, day=1))
        # 简化成直接逻辑
        if current.month == 12:
            month_last_day = datetime(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_last_day = datetime(current.year, current.month + 1, 1) - timedelta(days=1)

        chunk_start = max(start_dt, current)
        chunk_end = min(end_dt, month_last_day)

        yield chunk_start, chunk_end
        current = next_month(current)


# 为了避免上面需要 timedelta，把它单独导入并重写一次更清晰的版本
from datetime import timedelta

def iter_month_ranges(start_dt, end_dt):
    current = start_dt.replace(day=1)
    while current <= end_dt:
        if current.month == 12:
            month_last_day = datetime(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_last_day = datetime(current.year, current.month + 1, 1) - timedelta(days=1)

        chunk_start = max(start_dt, current)
        chunk_end = min(end_dt, month_last_day)

        yield chunk_start, chunk_end

        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


def day_list(start_dt, end_dt):
    days = []
    cur = start_dt
    while cur <= end_dt:
        days.append(cur.strftime("%d"))
        cur += timedelta(days=1)
    return days


def download_one_month(client, chunk_start, chunk_end, area, outdir, prefix):
    year = chunk_start.strftime("%Y")
    month = chunk_start.strftime("%m")
    days = day_list(chunk_start, chunk_end)

    pressure_file = outdir / f"{prefix}_pressure_{year}{month}_{days[0]}-{days[-1]}.grib"
    single_file = outdir / f"{prefix}_single_{year}{month}_{days[0]}-{days[-1]}.grib"

    print(f"\n下载月份片段: {chunk_start.date()} ~ {chunk_end.date()}")
    print(f"Pressure file: {pressure_file}")
    print(f"Single file  : {single_file}")

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": PL_VARS,
            "pressure_level": PRESSURE_LEVELS,
            "year": [year],
            "month": [month],
            "day": days,
            "time": TIMES,
            "area": area,
            "data_format": "grib",
            "download_format": "unarchived",
        },
        str(pressure_file),
    )

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": SL_VARS,
            "year": [year],
            "month": [month],
            "day": days,
            "time": TIMES,
            "area": area,
            "data_format": "grib",
            "download_format": "unarchived",
        },
        str(single_file),
    )


def main():
    args = parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    if end_dt < start_dt:
        raise ValueError("结束日期不能早于开始日期")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    print("开始下载 ERA5 数据")
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"区域范围: N/W/S/E = {args.area}")
    print(f"输出目录: {outdir}")
    print(f"文件前缀: {args.prefix}")

    for chunk_start, chunk_end in iter_month_ranges(start_dt, end_dt):
        download_one_month(
            client=client,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            area=args.area,
            outdir=outdir,
            prefix=args.prefix,
        )

    print("\n全部下载完成。")


if __name__ == "__main__":
    main()