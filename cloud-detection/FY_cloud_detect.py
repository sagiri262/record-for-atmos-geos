"""
FY_cloud_detect.py


This script handles FY-4 L1 data that may arrive in HDF4 or HDF5 format.
It will:
- detect whether the input file is HDF5 (via h5py) or fallback assume HDF4
- list subdatasets (via GDAL) and convert the requested subdataset to NetCDF
using GDAL Translate (format='NETCDF')
- open the converted NetCDF with xarray and extract the optical band array
- run a simple rule-based cloud detector on the provided optical band (MULTI)
- output mask + visualization (optionally export polygons GeoJSON)


Usage:
python FY_cloud_detect.py --input FY4_L1.hdf --var-name MULTI --time-index 0 \
--out-mask fy4_cloud_mask.nc --out-plot fy4_clouds.png


Dependencies:
numpy, xarray, h5py, osgeo (gdal), skimage, shapely, geopandas (optional), matplotlib


Note: You told me the FY-4B imager L1 MULTI band spectral range is 0.55-0.75 um.
Use --var-name to pass the right variable name inside the HDF/NetCDF.

此文件处理 FY-4 L1 数据（可能为 HDF4 或 HDF5），流程包含：
 1. 检查 HDF 文件类型（HDF5 或 HDF4/UNKNOWN）
 2. 使用 GDAL 列出子数据集（subdatasets），并将指定子数据集转换为 NetCDF
 3. 使用 xarray 打开已转换的 NetCDF，提取用户指定的光学波段（例如 MULTI）
 4. 对该波段做简单的可见光规则云检测（基于反射率阈值）
 5. 输出云掩膜（NetCDF）并生成带红色轮廓的可视化图

说明：你之前提供的 MULTI 波段范围为 0.55-0.75 μm，本脚本默认使用变量名 `MULTI`，若 HDF 内变量名不同请用 --var-name 指定。

"""

import os
import argparse
import numpy as np
import xarray as xr
import h5py
from osgeo import gdal
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage import measure
from scipy import ndimage as ndi

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except Exception:
    gpd = None
    from shapely.geometry import Polygon


def detect_hdf_type(path):
    """Return 'HDF5' or 'HDF4/UNKNOWN'"""
    try:
        if h5py.is_hdf5(path):
            return 'HDF5'
    except Exception:
        pass
    return 'HDF4/UNKNOWN'


def list_subdatasets(path):
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError('GDAL can not open file: ' + path)
    sds = ds.GetSubDatasets()
    return sds


# 已经确定可以转为NetCDF，开始转换
def convert_subdataset_to_netcdf(hdf_path, subdataset_index, out_nc_path):
    ds = gdal.Open(hdf_path)
    sds = ds.GetSubDatasets()

    if len(sds) == 0:
        # try translating the whole file
        print('No subdatasets found; Attempting to translate whole datasets')
        res = gdal.Translate(out_nc_path, hdf_path ,format='NETCDF')
        del res
        return out_nc_path

    if subdataset_index < 0 or subdataset_index >= len(sds):
        raise RuntimeError('Subdataset index out of range')

    sub_name = sds[subdataset_index][0]
    print('Converting subdataset:', sub_name)

    res = gdal.Translate(out_nc_path, sub_name ,format='NETCDF')
    del res
    return out_nc_path


def detect_clouds_fy4_vis(vis_arr, th_vis=0.25, min_area=100):
    # 云检测模块的函数
    """基于单波段（可见光）进行云检测的简单规则

    - 假定 vis_arr 是 TOA 反射率或已缩放到 0..1 的反射率
    - 如果输入是 DN 或 0..10000 的尺度，尝试自动缩放到 0..1
    - 利用阈值 th_vis 判断反射率高的像元为云候选
    - 使用形态学闭运算填补小洞，去除小面积孤立像元
    """
    # vis_arr expected 0..1 reflectance
    vis = vis_arr.astype('float32')
    m = np.nanmax(vis)
    if 1.0 < m <= 10000:
        vis = vis / 1000.0
    if 1.0 < m <= 65535:
        vis = vis / 65535.0

    mask_vis = vis > th_vis
    #
    mask = binary_closing(mask_vis, footprint=disk(5))
    mask = remove_small_objects(mask, min_size=min_area)
    return mask.astype(bool)


def polygonize_mask(mask, min_area_pixel=10):
    """将二值掩膜转换为多边形（像素坐标形式）

    - 使用 skimage.measure.find_contours 提取轮廓
    - 构建 shapely Polygon 并过滤面积较小的多边形
    - 如果 geopandas 可用，返回 GeoDataFrame，否则返回 Polygon 列表
    """
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    polys = []

    for contour in contours:
        coords = [(float(c[1]), float(c[0])) for c in contour]
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if poly.area >= min_area_pixel:
            polys.append(poly)

    if gpd is not None:
        gdf = gpd.GeoDataFrame(geometry=polys)
        return gdf

    return polys


def save_mask_to_netcdf(mask, out_path, template_ds=None, varname='cloud_mask'):
    """将 2D boolean 掩膜保存为 NetCDF 文件（uint8 类型）

    - mask: 2D boolean 数组
    - template_ds: 若提供 xarray Dataset，可用其坐标信息
    - varname: 掩膜变量名
    """
    da = xr.DataArray(mask.astype('uint8'), dims=('y', 'x'))
    ds = da.to_dataset(name=varname)

    if template_ds is not None:
        try:
            ds = ds.assign_coords({
                'y': template_ds.coords.get('y', np.arange(mask.shape[0])),
                'x': template_ds.coords.get('x', np.arange(mask.shape[1]))
            })
        except Exception:
            pass
        ds.to_netcdf(out_path)


def plot_truecolor_with_mask(rgb, mask, title='FY4 真彩色 + 云轮廓', out_plot=None):
    """绘制真彩色图像并叠加云掩膜轮廓

    - rgb: HxWx3 数组（0..1）
    - mask: HxW 布尔数组
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(rgb, vmax=1.0)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap='gray', alpha=0.4)
    cont = ndi.binary_dilation(mask.astype(np.uint8), iterations=1) ^ mask.astype(np.uint8)
    ax.contour(cont, levels=[0.5], colors='r', linewidths=1)
    ax.set_title(title)
    ax.axis('off')

    if out_plot:
        fig.savefig(out_plot, bbox_inches='tight', dpi=1200)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 FY-4 HDF 文件')
    parser.add_argument('--var-name', default='MULTI', help='HDF/NetCDF 中可见光波段变量名（如 MULTI）')
    parser.add_argument('--subdataset-index', type=int, default=0, help='若有多个子数据集，选择索引')
    parser.add_argument('--out-nc', default='fy4_converted.nc', help='转换后的 NetCDF 路径')
    parser.add_argument('--out-mask', default='fy4_cloud_mask.nc', help='输出云掩膜 NetCDF 路径')
    parser.add_argument('--out-plot', default='fy4_clouds.png', help='输出可视化图像路径')
    parser.add_argument('--th-vis', type=float, default=0.25, help='可见光反射率阈值')
    args = parser.parse_args()

    # 步骤 1：检测 HDF 类型
    hdf_type = detect_hdf_type(args.input)
    print('Detected HDF type:', hdf_type)

    # 步骤 2：列出子数据集
    sds = list_subdatasets(args.input)
    print('Found', len(sds), 'subdatasets (GDAL)')
    for i, s in enumerate(sds):
        print(i, s[0])
    # 步骤 3：将选定子数据集转换为 NetCDF
    converted = convert_subdataset_to_netcdf(args.input, args.subdataset_index, args.out_nc)
    print('Converted to', converted)

    # 步骤 4：使用 xarray 打开转换后的 NetCDF
    ds = xr.open_dataset(converted)
    print('Converted dataset variables:', list(ds.variables.keys()))

    # 步骤 5：确定要使用的波段变量
    if args.var_name not in ds and args.var_name not in ds.variables:
        cand = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if len(cand) == 0:
            raise RuntimeError('找不到合适的 2D 变量；请指定 --var-name')
        var = cand[0]
        print('Using variable', var)
    else:
        var = args.var_name

    # 步骤 6：若变量有额外维度（例如时间、通道），取第一层
    arr = ds[var].isel({k:0 for k in ds[var].dims[2:]}) if ds[var].ndim > 2 else ds[var]
    vis = arr.values

    # 步骤 7：进行云检测
    mask = detect_clouds_fy4_vis(vis, th_vis=args.th_vis)

    # 步骤 8：保存云掩膜为 NetCDF
    save_mask_to_netcdf(mask, args.out_mask, template_ds=ds, varname='cloud_mask')
    print('Saved cloud mask to', args.out-mask)

    # 步骤 9：尝试构建简单 RGB（若 NetCDF 包含 MULTI 波段）
    rgb = None
    if 'MULTI' in ds:
        r = ds['MULTI'].values
        r = r.astype('float32')
        if np.nanmax(r) > 1.0:
            r = r / 10000.0
        rgb = np.stack([r, r, r], axis=-1)

    # 步骤 10：若 rgb 构建成功则绘图
    if rgb is not None:
        plot_truecolor_with_mask(rgb, mask, out_plot=args.out_plot)
        print('Saved plot to', args.out_plot)

if __name__ == '__main__':
    main()






















