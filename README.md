# 代码随想记录

##  -- surface_pressure 目录
主要记录怎么使用再分析数据，使用再分析数据计算大气相关参数，如，大气垂直层上的等压面、水汽湿度分层等

## ERA5 数据介绍
ERA5是ECMWF（欧洲中期天气预报中心）对1950年1月至今全球气候的第五代大气再分析数据集。ERA5由ECMWF的哥白尼气候变化服务（C3S）生产。ERA5提供了大量大气、陆地和海洋气候变量的每小时估计值。这些数据覆盖了30公里网格上的地球，并使用137个从地表到80公里高度的高度来解析大气，包括在降低空间和时间分辨率时所有变量的不确定性信息。ERA5将模型数据与来自世界各地的观测数据结合起来，形成一个全球完整的、一致的数据集，取代了其前身ERA-Interim再分析。

ERA5提供逐小时 0.25分辨率，垂直37层的再分析资料。也提供月平均资料。但是没有日平均资料，因此如果需要日平均资料，则需要自己将24小时数据下载后再处理，或者可以在 GEE 上获取，但是需要注意，GEE 的数据时间会有大约半年左右的延迟，也可以下载 hourly 数据在云端合成 daily 数据后下载。关于分辨率，除了0.25°的，在下载时，好像也可以通过更改 python下载脚本中的 'grid': [1.0, 1.0], 参数进行下载。

常用的 CDS Climate Data Store 有：

ERA5（大气/陆面/海浪的核心再分析，CDS 上的常用子集）
 - ERA5 hourly data on single levels from 1940 to present
 - ERA5 monthly averaged data on single levels from 1940 to present
 - ERA5 hourly data on pressure levels from 1940 to present
 - ERA5 monthly averaged data on pressure levels from 1940 to present

ERA5 “post-processed daily statistics”（日统计派生入口）
 - ERA5 post-processed daily statistics on single levels from 1940 to present
 - ERA5 post-processed daily statistics on pressure levels from 1940 to present

ERA5 “hourly time-series”（单点长序列优化入口）
 - ERA5 hourly time-series data on single levels from 1940 to present

ERA5-Land（陆面高分辨率强项）
 - ERA5-Land hourly data from 1950 to present
 - ERA5-Land monthly averaged data from 1950 to present
 - ERA5-Land post-processed daily statistics from 1950 to present
 - ERA5-Land hourly time-series data from 1950 to present

### 数据参数
这是一个很大工作量。首先，参数的单位并不适合直接分析或作图，需要一定的换算才能够使用，比如降水数据。
Table 1: surface and single level parameters: invariants (in time)
| count | name                                             | units   | Variable name in CDS                             | shortName | paramId |  an |  fc |
| ----: | ------------------------------------------------ | ------- | ------------------------------------------------ | --------- | ------: | :-: | :-: |
|     1 | Lake cover                                       | (0 - 1) | lake_cover                                       | cl        |      26 |  x  |  x  |
|     2 | Lake depth                                       | m       | lake_depth                                       | dl        |  228007 |  x  |  x  |
|     3 | Low vegetation cover                             | (0 - 1) | low_vegetation_cover                             | cvl       |      27 |  x  |     |
|     4 | High vegetation cover                            | (0 - 1) | high_vegetation_cover                            | cvh       |      28 |  x  |     |
|     5 | Type of low vegetation                           | ~       | type_of_low_vegetation                           | tvl       |      29 |  x  |     |
|     6 | Type of high vegetation                          | ~       | type_of_high_vegetation                          | tvh       |      30 |  x  |     |
|     7 | Soil type                                        | ~       | soil_type                                        | slt       |      43 |  x  |     |
|     8 | Standard deviation of filtered subgrid orography | m       | standard_deviation_of_filtered_subgrid_orography | sdfor     |      74 |  x  |     |
|     9 | Geopotential                                     | m² s⁻²  | geopotential                                     | z         |     129 |  x  |  x  |
|    10 | Standard deviation of sub-gridscale orography    | ~       | standard_deviation_of_orography                  | sdor      |     160 |  x  |     |
|    11 | Anisotropy of sub-gridscale orography            | ~       | anisotropy_of_sub_gridscale_orography            | isor      |     161 |  x  |     |
|    12 | Angle of sub-gridscale orography                 | radians | angle_of_sub_gridscale_orography                 | anor      |     162 |  x  |     |
|    13 | Slope of sub-gridscale orography                 | ~       | slope_of_sub_gridscale_orography                 | slor      |     163 |  x  |     |
|    14 | Land-sea mask                                    | (0 - 1) | land_sea_mask                                    | lsm       |     172 |  x  |  x  |

''Table 2 研究近地表面气压 & 垂直气压''
| 研究对象            | 参数（CDS 变量名 / shortName）                    |      单位 | 用途/说明                                                            |
| --------------- | ------------------------------------------ | ------: | ---------------------------------------------------------------- |
| 近地表面气压（地表）      | surface_pressure / **sp**                  |      Pa | 地表气压本体；做区域气压场、气压梯度、天气系统分析常用。([ECMWF Confluence][1])              |
| 海平面气压（常用于天气图）   | mean_sea_level_pressure / **msl**          |      Pa | 把气压折算到海平面，便于比较不同地形高度处的气压系统（高低压、等压线）。([ECMWF Confluence][1])      |
| 垂直结构：高度/位势高度相关  | geopotential / **z**（pressure levels）      |  m² s⁻² | 等压面上的位势（常用来换算位势高度 Z = z/g），画 500hPa 高度场等。([ECMWF Confluence][1]) |
| 垂直结构：温度         | temperature / **t**（pressure levels）       |       K | 等压面温度场、厚度、稳定度等分析。([ECMWF Confluence][1])                         |
| 垂直结构：湿度（两种常用表达） | specific_humidity / **q**（pressure levels） | kg kg⁻¹ | 等压面比湿；水汽输送、水汽含量诊断常用。([ECMWF Confluence][1])                      |
|                 | relative_humidity / **r**（pressure levels） |       % | 等压面相对湿度；判断云雨潜势、干湿层结。([ECMWF Confluence][1])                      |
| 垂直运动            | vertical_velocity / **w**（pressure levels） |  Pa s⁻¹ | 等压面垂直速度（ω）；上升/下沉运动诊断。([ECMWF Confluence][1])                     |

[1]: https://confluence.ecmwf.int/display/CKB/ERA5%3A%2Bdata%2Bdocumentation "ERA5: data documentation - Copernicus Knowledge Base - ECMWF Confluence Wiki"




## NOAA 再分析数据介绍


### 主要研究大气物理现象
#### 1、近地表气象数据
主要包括 **近地面大气压**、**海平面气压**，这二者存在不同，近地表面大气压表示近地面的气压场；而海平面气压则通过一定的折算关系，将因地形影响的不同高度气压折算到一个高度然后用于分析。
近地面**风速**、**气温**、**降水量**，是最常见、最贴近日常生活的一些气象参数。其中，用于分析与直接使用之间存在一些关系。
**风速** ： 风速在水平面上被分解为 U 分量和 V 分量，变量名为：10m_u_component_of_wind / 10u 和 10m_v_component_of_wind / 10v。单位上不需要换算，但是在计算实际风向和风速时，需要合成矢量，计算结果需要换算。
**气温** ：气温数据的单位时开尔文，需要减去绝对零度得到摄氏度
**降水** ：降水单位是米，实际常用毫米为单位，乘以 1000

