# NetCDF文件的基本处理流程

netcdf的文件处理流程包括：怎么获得数据，数据到手之后怎么检查文件的格式是什么，维度信息、变量定义、全局属性，再做数据的读取、缺值解析、时间解析、变量单位的确认，预处理nc文件然后计算文件中的变量和数据。因为nc数据的处理是有一定的流程的，有很多信息如头文件、部分变量等都是相同的，比如时间、经纬度等。这些流程是基本固定的。

在Linux环境下处理nc文件是最好的。这里可以使用 NCL、python、matlab 等都可以处理。windows环境下，只建议使用 matlab 来处理 nc 文件，原因是windows底层的一些库和一些东西是不支持或者不兼容 nc 文件的一些细节的，在windows上用python库或者NCL处理nc数据、绘图的时候会碰到很奇怪的问题。但是同样的代码迁移到linux系统就没了。为了在一开始就杜绝这种奇怪的问题，我建议使用WSL，即 Windows Linux 子系统来处理。

在 Linux 下具体的操作和命令就不讲了，这些东西自己上网就可以搜到

本文档中，会使用python的 wrf（主要用于wrfout_d01*数据处理），xarray、netcdf4 等库。在命令行中，可以使用 ncdump 在命令行操作nc文件。

## 目录
- [一、文件类型](##一、文件类型)
- [二、维度信息](##二、维度信息)
- [三、变量定义](##三、变量定义)
- [四、ncdump文件格式探查：维度、变量、属性速览](##四、ncdump文件格式探查：维度、变量、属性速览)

本文大致归纳为：
STEP 1环境准备：系统依赖安装、Python 虚拟环境配置
STEP 2ncdump 文件格式探查：维度、变量、属性速览
STEP 3netCDF4 底层读取：精确切片、缺测值处理、时间解码
STEP 4xarray 高层次操作：标签切片、气候态、距平、加权均值
STEP 5wrf-python 专项：诊断量获取、垂直插值到等压面、常用变量速查表
STEP 6数据计算：统计、频谱、相关、趋势、空间插值、重采样
STEP 7可视化：matplotlib + cartopy 填色图与地图投影
STEP 8写出与转换：NetCDF 压缩输出、CSV、GeoTIFF


## 一、文件类型
我们首先要确定文件的类型是什么，这里要讲讲netcdf文件的发展。请参考:https://www.unidata.ucar.edu/software/netcdf
网络通用数据表 (netCDF) 是一组软件库及与机器无关的数据格式，支持创建、访问和共享面向数组的科学数据，从数学关系上看，NetCDF数据格式中存储的数据具有多对一的函数关系，"多"是指维，"一"是指变量值，这种数据结构的最大特点是能够方便地使用多维矩阵。例如，某个气象站点记录的随时间变化的温度数据以一维数组的形式存储，某个区域内在指定时间的温度以二维数组的形式存储，某个区域内随时间变化的温度用三维数组存储，某个区域内随时间和高度变化的温度用四维数组存储。

Python中有一系列的工具可以操作和使用 NetCDF数据，其中常用的由netCDF4和xarray等。Matlab中也提供一系列的函数和接口用于处理nc文件。可见附录1中的详细表格。

文件格式主流的有：netCDF4、netCDF5、HDF5等，不同的文件格式得用不同的命令，我们只考虑netcdf4.

可以用以下命令打开：
`file ncfile.nc`
`ncdump -k ncfile.nc`

第一个命令 `file ncfile.nc` 用来判断时文件是 NETCDF 还是 HDF 文件。
第二个命令 `ncdump -k ncfile.nc` 用来显示 netcdf 类型，常见结果有：`classic、64-bit offset、netCDF-4、netCDF-4 classic model`。
我们检查出来文件的类型，才能确定后面要用什么库或者依赖什么的，方便后续处理。很多时候，我们不检查类型就直接处理的话就往往会出现奇怪的报错，又要回到检查文件类型，那我们在工作开始前就做这个事情。

如果系统里还没有 ncdump，可以安装：

`sudo apt update`
`sudo apt install -y netcdf-bin libnetcdf-dev`

## 维度信息
维度信息决定了变量的组织方式。看维度时，重点不是“有几个名字”，而是要弄清楚每个维度代表什么、长度是多少、顺序怎样、是否存在无限维。

使用以下命令来查看文件的维度信息
`ncdump -h ncfile.nc`

我们要读的是里面的 "dimensions" 字段，这里保存着维度信息，例如：
dimensions:
    time = UNLIMITED ;
    lat = 181 ;
    lon = 360 ;
例子里有三个维度，即时间维度、经度维度、纬度维度三个维度。在实际的netcdf文件中，还有如：time、lat、lon、level、lev、x、y。因为在实际的大气研究中，我们还需要水平上的其他维度，如 x, y 距离；在垂直方向上，我们也需要 level、z 等表示垂直层数、位势高度等维度。

这一阶段最容易出错的地方，是想当然地把所有变量都当成 (time, lat, lon) 来处理。实际上，不同数据源的维度顺序可能是 (lat, lon)、(time, level, lat, lon)、(Time, south_north, west_east)，甚至还有交错网格维。因此，看到维度后，应该先把“维度名称、长度、顺序”单独记下来。

我们写一个简单的python代码来实现读取维度数据：
```python
import xarray as xr

ds = xr.open_dataset("example.nc")
print(ds.dims)
```

## 三、变量定义

变量定义是理解文件内容的核心。看变量时，主要要回答这些问题：文件里有哪些变量、哪些是坐标变量、哪些是业务变量、每个变量依赖哪些维度、单位是什么、有没有缺测值定义、是否带缩放和偏移属性。

仍然使用：

`ncdump -h example.nc`

在 variables: 段中，通常会看到类似内容：

variables:
    double time(time) ;
        time:units = "hours since 2000-01-01 00:00:00" ;
    float lat(lat) ;
        lat:units = "degrees_north" ;
    float lon(lon) ;
        lon:units = "degrees_east" ;
    float t2m(time, lat, lon) ;
        t2m:units = "K" ;
        t2m:_FillValue = -9999.f ;

这里通常可以把变量分成三类，即坐标变量、数据变量、辅助变量。坐标变量包括 time、lat、lon，它们用于描述维度本身；数据变量包括温度、降水、风速、气压等，它们是后续分析的主要对象；辅助变量则可能包括 crs、mask、边界变量、质量控制变量等。

看变量定义时，建议重点检查这些内容：变量名、变量维度、变量单位、缺测值属性、长名称和标准名、是否包含 scale_factor、add_offset。尤其是单位、缺测值、维度顺序，必须先看清楚再计算，否则后面算出来的结果很可能全错。

Python 中可以直接查看变量：

```python
import xarray as xr

ds = xr.open_dataset("example.nc")
da = ds["t2m"]

print(da.dims)
print(da.shape)
print(da.attrs)
```

也可以用 netCDF4：

```python
from netCDF4 import Dataset

nc = Dataset("example.nc")
var = nc.variables["t2m"]

print(var.dimensions)
print(var.shape)
for attr in var.ncattrs():
    print(attr, getattr(var, attr))
```



接下来是硬核的部分
## 四、ncdump文件格式探查：维度、变量、属性速览 

ncdump 是 NetCDF 官方工具链中的常用命令行工具，主要用于把 NetCDF 文件的结构信息和数据内容以文本形式输出出来。实际使用中，它最常见的用途包括，判断文件格式、查看文件头、浏览维度定义、查看变量声明、检查全局属性、查看坐标变量、抽样读取变量内容。

ncdump 很适合做首轮探查，但不适合承担复杂计算任务。实际工作中，比较合理的分工通常是：用 ncdump 看结构、看变量、看属性、看坐标，用 xarray 或 netCDF4 做程序化读取和统计计算，用 wrf-python 处理 WRF 输出中的诊断量和插值问题。

第一，优先使用 ncdump -h，因为它最稳妥、最节省终端输出；第二，查看变量时尽量用 -v 配合 head，避免大数组刷屏；第三，坐标检查优先用 -c，尤其是时间、经纬度、层次坐标；第四，遇到复杂文件时用 grep 和 less 配合阅读，比全量输出更高效。ncdump 最重要的价值不在于“把数据打印出来”，而在于帮助我们快速建立对 NetCDF 文件结构的认识。对于新的数据文件而言，这一步往往比直接写代码更关键。

如果系统中尚未安装 ncdump，在 Ubuntu 22.04 下可以先安装：

```shell
sudo apt update
sudo apt install -y netcdf-bin
```

### 检查文件格式
上文已经提到文件格式检查的重要性了，下面给出常用的检查文件格式的命令。这个动作很简单，但很有必要，因为它决定了后续兼容性判断和读取方式。

最常用的命令是：

`ncdump -k example.nc`

这个命令用于查看文件底层格式，常见输出包括 classic、64-bit offset、netCDF-4、netCDF-4 classic model。看到这些结果后，就能大致判断这个文件是传统 NetCDF3 结构，还是基于 HDF5 的 NetCDF4 文件。

如果只是想做首轮探查，一般先执行：

```shell
ncdump -k example.nc
ncdump -h example.nc
```

前者看格式，后者看结构，两步通常就能完成最基础的文件识别。

### 检查维度、变量、属性
对 NetCDF 文件做“速览”时，最核心的信息主要有三类，即维度、变量、属性。

维度用于描述数据组织方式，例如 time、lat、lon、level，变量用于描述实际的数据对象和坐标对象，例如温度、降水、风速、时间、经纬度，属性则用于补充说明，例如单位、缺测值、来源、处理历史、约定规范。

最常用的命令是：

`ncdump -h example.nc`

这个命令会输出文件头信息，其中通常包含三个关键部分，即 dimensions:、variables:、global attributes:。

一个典型的输出片段大致如下：

netcdf example {
dimensions:
    time = UNLIMITED ;
    lat = 181 ;
    lon = 360 ;

variables:
    double time(time) ;
        time:units = "hours since 2000-01-01 00:00:00" ;
    float lat(lat) ;
        lat:units = "degrees_north" ;
    float lon(lon) ;
        lon:units = "degrees_east" ;
    float t2m(time, lat, lon) ;
        t2m:units = "K" ;
        t2m:_FillValue = -9999.f ;

// global attributes:
        :title = "Example dataset" ;
        :source = "model output" ;
}


从这样的输出中，可以很快看出，文件里有哪些维度、哪些变量、每个变量依赖哪些维度、变量带了哪些属性、文件整体又有哪些全局属性。对于新文件来说，这一步通常已经足够帮助我们建立第一印象。

### 常见操作总结
1. 查看文件格式
`ncdump -k example.nc`

这个命令用于查看 NetCDF 文件格式。常用于第一步确认文件类型，判断它是 classic、64-bit offset、netCDF-4 还是 netCDF-4 classic model。如果后续要做格式转换、兼容性排查，这个命令通常是首选。

2. 查看文件头信息
`ncdump -h example.nc`

这是 ncdump 最常用的命令。它只输出文件头，不输出完整数据，适合快速查看维度、变量、属性。日常所说的“先看看这个 nc 文件里有什么”，大多数时候就是在用这个命令。

使用 -h 时，重点通常看三部分，即 dimensions、variables、global attributes。如果只是做结构探查，基本不需要把全部数据打印出来。

3. 查看坐标变量和值的概要
`ncdump -c example.nc`

-c 的作用可以理解为“在输出头信息的同时，把坐标变量的值也一并显示出来”。这个命令特别适合检查时间坐标、经纬度坐标、层次坐标。

例如，使用这个命令后，可以快速判断，经度是 0~360 还是 -180~180，纬度是递增还是递减，时间坐标是否连续、是否符合预期。对于新文件来说，这一步非常有用，因为很多后续错误都来自对坐标的误判。

4. 查看指定变量的数据
`ncdump -v t2m example.nc`

这个命令用于查看指定变量的定义和数据内容。如果变量比较小，可以直接完整输出；如果变量很大，通常建议结合 head 或重定向来使用，例如：

`ncdump -v t2m example.nc | head -n 40`

这样做的目的，是先抽样看看变量值是否合理，而不是把整块大数组全部刷到终端里。实际使用中，-v 最适合查看小变量、坐标变量、标量变量，或者只想抽样看看某个变量的前几行数据。

5. 排除某些变量不显示
`ncdump -x example.nc`

严格来说，是否使用这一类参数要看具体版本和场景，但在常规使用里，更常见的思路其实不是“输出全部再排除”，而是优先选择 -h 或 -v 只看自己关心的部分。对于大文件来说，这比无差别打印更实用。

如果只是想避开大变量，通常更建议直接使用：

```shell
ncdump -h example.nc
ncdump -v time,lat,lon example.nc
```

也就是说，日常探查时更推荐“按需看变量”，而不是“全量打印后再过滤”。

6. 结合 grep 检查关键信息

虽然这不是 ncdump 自身的参数，但它是实际工作中非常常用的方式。比如，只想看时间相关信息，可以写：

`ncdump -h example.nc | grep time -n`

只想看单位信息，可以写：

`ncdump -h example.nc | grep units -n`

只想看缺测值定义，可以写：

`ncdump -h example.nc | grep FillValue -n`

这种方式特别适合大文件快速定位信息，也适合在终端里做第一轮排查。

7. 结合 less 分页查看

对于结构比较复杂的文件，直接输出到屏幕上往往不方便阅读，这时可以配合 less 使用：

`ncdump -h example.nc | less`

这样可以逐页查看维度、变量和属性，尤其适合变量很多、属性很多、包含 group 结构的文件。

8. 将输出保存到文本文件

如果需要留档、比对或分享，可以把结果保存到文本文件中：

`ncdump -h example.nc > header.txt`

如果要保存某个变量的输出：

`ncdump -v t2m example.nc > t2m_dump.txt`

这在做数据交接、问题排查、生成说明文档时都很有用。

9. 日常工作可以直接执行的流程

在日常工作里，ncdump 的使用通常不需要很复杂。一个比较实用的流程往往就是下面这样几步。

先看文件格式是什么，确认格式没问题后检查文件整体结构、坐标变量等，最后抽取我们需要的关键变量。

```shell
ncdump -k example.nc
ncdump -h example.nc
ncdump -c example.nc | less
ncdump -v time example.nc | head -n 30
ncdump -v lat,lon example.nc | head -n 40
ncdump -v t2m example.nc | head -n 40
```

这样处理之后，通常就已经能回答这些关键问题，即文件是什么格式、有哪些维度、主要变量是什么、变量单位是什么、有没有缺测值、时间坐标和经纬度是否合理。


## 五、xarray数据读取：精确切片、缺测值处理、时间解码
在 NetCDF 文件处理流程中，xarray 通常承担的是高层读取与分析的工作。相比 netCDF4 更偏底层的访问方式，xarray 更适合做结构查看、变量选择、按坐标切片、缺测值屏蔽、时间坐标解析、统计计算与结果输出。

这一阶段的重点，不是把所有数据一次性读进来，而是先通过 xarray 快速建立对数据集结构的认识，再针对目标变量做精确切片、缺测值处理、时间解码。这样做，既能减少误读变量和坐标的风险，也便于后续把代码沉淀为统一脚本。

xarray 是面向带标签多维数组的数据分析工具，特别适合处理 NetCDF、GRIB、Zarr 这类具有维度、坐标、属性的数据文件。它的核心优势在于，变量和坐标都带名字，切片时既可以按位置取，也可以按坐标取，时间序列处理和统计计算也比较自然。

在实际工作里，xarray 最常见的用途包括，打开 NetCDF 文件、查看数据集结构、读取变量、按时间和空间切片、处理缺测值、解析时间坐标、计算均值和累计量、重采样、输出 NetCDF 文件。如果只是最基本地读取 NetCDF 文件，通常需要至少安装 xarray 和 netCDF4。

### 打开文件与查看文件结构
使用 xarray 处理 NetCDF 文件时，通常先打开数据集，再查看整体结构。最常用的命令是：

```python
import xarray as xr

ds = xr.open_dataset("ncfile.nc")
print(ds)
```

*open_dataset()* 用来打开单个 NetCDF 文件，返回的是一个 Dataset 对象。这个对象通常包含维度、坐标、变量、属性四部分信息。打印 ds 后，通常就能看到文件中有哪些维度、有哪些坐标、有哪些数据变量，以及每个变量的形状和类型。

进一步查看结构时，最常用的命令包括：

```python
print(ds.dims)
print(ds.coords)
print(ds.data_vars)
print(ds.attrs)
```

这里，*dims* 用来看维度名称和长度，*coords* 用来看坐标变量，*data_vars* 用来看主要数据变量，*attrs* 用来看全局属性。

如果只关心某一个变量，可以先取出变量对象：

```python
da = ds["t2m"]
print(da)
print(da.dims)
print(da.shape)
print(da.attrs)
```
这一步的意义在于，先把变量的维度顺序、形状、单位、缺测值属性看清楚，再决定如何切片和计算。

如果要一次打开多个同构文件，可以用：

`ds = xr.open_mfdataset("data/*.nc", combine="by_coords")`

这个命令适合按时间拼接多个 NetCDF 文件，但前提是变量结构和坐标基本一致。

### 数据切片
netcdf的一个精髓就是对数据进行切片。netcdf有一个常见的现象，我们可能需要太平洋海域、印度洋海域，但是生成的nc文件经常是全球的。因为模拟或者实际观测数据往往需要比用户实际使用的范围要大，不然经常出现数据不够用，或者精度不达标的现象出现。所以，数据到手后往往第一步就是检查数据的经纬度维度，这是切片的关键。切片的逻辑就是：“准确地拿到我们需要范围内的数据”。

常见的切片方法主要有 isel()、sel()、loc、条件筛选、变量组合筛选。接下来我们一条一条的讲解。

#### 1、isel()
如果已经知道变量的维度顺序，可以用 isel() 按位置取值：

`da0 = ds["t2m"].isel(time=0)`

这表示取第一个时间片。如果想同时截取一个局部网格区域，可以写成：

`sub = ds["t2m"].isel(time=0, lat=slice(0, 10), lon=slice(0, 20))`

isel() 的优点是直接、稳定，适合明确知道索引位置的场景；但它不直观，因为必须先清楚维度顺序和位置编号。

#### 2、sel()
1. 根据真实值切片
如果想按真实坐标值切片，通常用 sel()：

`sub = ds["t2m"].sel(time="2024-01-01")`

2. 按时间范围和经纬度范围切片

```python
sub = ds["t2m"].sel(
    time=slice("2024-01-01", "2024-01-31"),
    lat=slice(40, 20),
    lon=slice(100, 130)
)
```

这里的关键点在于，sel() 是按坐标值而不是按位置取值，因此更适合业务分析场景。需要注意的是，纬度有时是递减排列，如果 lat 是从北到南排布，slice(40, 20) 才能正确选中；若纬度递增，则要改成 slice(20, 40)。

3. 取某个具体经纬度点附近的值
可以用最近邻方法：

`point = ds["t2m"].sel(lat=31.2, lon=121.5, method="nearest")`

这个用法很适合站点对比、单点时间序列抽取。

4. 要先按范围切出子集，再针对多个变量一起分析，通常直接对整个 Dataset 切片：

`sub_ds = ds.sel(time=slice("2024-01-01", "2024-01-10"), lon=slice(100, 120))`

这样可以同时保留多个变量和对应坐标。













































## 附录一
### 高级接口函数
| 函数              | 说明                          |
| --------------- | --------------------------- |
| `nccreate`      | 在 netCDF 文件中创建变量            |
| `ncdisp`        | 在命令行窗口中显示 netCDF 数据源内容      |
| `ncinfo`        | 返回有关 netCDF 数据源的信息          |
| `ncread`        | 读取 netCDF 数据源中的变量数据         |
| `ncreadatt`     | 读取 netCDF 数据源中的属性           |
| `ncwrite`       | 将数据写入 netCDF 文件             |
| `ncwriteatt`    | 向 netCDF 文件写入属性             |
| `ncwriteschema` | 将 netCDF 架构定义添加到 netCDF 文件中 |

### 普通接口函数
| 分类      | 函数                        | 说明                                   |
| ------- | ------------------------- | ------------------------------------ |
| 库函数     | `netcdf.getChunkCache`    | 返回 netCDF 库的默认块缓存设置                  |
| 库函数     | `netcdf.inqLibVers`       | 返回 netCDF 库版本信息                      |
| 库函数     | `netcdf.setChunkCache`    | 设置 netCDF 库的默认块缓存设置                  |
| 库函数     | `netcdf.setDefaultFormat` | 更改默认 netCDF 文件的格式                    |
| 文件操作    | `netcdf.abort`            | 还原最近的 netCDF 文件定义                    |
| 文件操作    | `netcdf.close`            | 关闭 netCDF 文件                         |
| 文件操作    | `netcdf.create`           | 创建新的 netCDF 数据集                      |
| 文件操作    | `netcdf.endDef`           | 结束 netCDF 文件定义模式                     |
| 文件操作    | `netcdf.inq`              | 返回有关 netCDF 文件的信息                    |
| 文件操作    | `netcdf.inqFormat`        | 确定 netCDF 文件的格式                      |
| 文件操作    | `netcdf.inqGrps`          | 返回子组 ID 数组                           |
| 文件操作    | `netcdf.inqUnlimDims`     | 返回组中所有可见的无限维度的 ID                    |
| 文件操作    | `netcdf.open`             | 打开 netCDF 数据源                        |
| 文件操作    | `netcdf.reDef`            | 让打开的 netCDF 文件进入定义模式                 |
| 文件操作    | `netcdf.setFill`          | 设置 netCDF 填充模式                       |
| 文件操作    | `netcdf.sync`             | 将 netCDF 文件同步到磁盘                     |
| 维度      | `netcdf.defDim`           | 创建 netCDF 维度                         |
| 维度      | `netcdf.inqDim`           | 返回 netCDF 维度名称和长度                    |
| 维度      | `netcdf.inqDimID`         | 返回维度 ID                              |
| 维度      | `netcdf.renameDim`        | 更改 netCDF 维度名                        |
| 组       | `netcdf.defGrp`           | 在 netCDF 文件中创建组                      |
| 组       | `netcdf.inqDimIDs`        | 返回组中维度标识符列表                          |
| 组       | `netcdf.inqGrpName`       | 返回组名                                 |
| 组       | `netcdf.inqGrpNameFull`   | 返回组的完整路径名                            |
| 组       | `netcdf.inqGrpParent`     | 返回父组的 ID                             |
| 组       | `netcdf.inqNcid`          | 返回组的 ID                              |
| 组       | `netcdf.inqVarIDs`        | 返回组中所有变量的 ID                         |
| 变量      | `netcdf.defVar`           | 创建 netCDF 变量                         |
| 变量      | `netcdf.defVarChunking`   | 定义 netCDF 变量的分块参数                    |
| 变量      | `netcdf.defVarDeflate`    | 定义 netCDF 变量的压缩参数                    |
| 变量      | `netcdf.defVarFill`       | 定义 netCDF 变量的填充参数                    |
| 变量      | `netcdf.defVarFletcher32` | 定义 netCDF 变量的校验参数                    |
| 变量      | `netcdf.getVar`           | 读取 netCDF 变量中的数据                     |
| 变量      | `netcdf.inqVar`           | 返回关于 netCDF 变量的信息                    |
| 变量      | `netcdf.inqVarChunking`   | 返回 netCDF 变量的分块参数                    |
| 变量      | `netcdf.inqVarDeflate`    | 返回 netCDF 变量的压缩参数                    |
| 变量      | `netcdf.inqVarFill`       | 返回 netCDF 变量的填充参数                    |
| 变量      | `netcdf.inqVarFletcher32` | 返回 netCDF 变量的校验参数                    |
| 变量      | `netcdf.inqVarID`         | 返回与变量名相关联的 ID                        |
| 变量      | `netcdf.putVar`           | 将数据写入 netCDF 变量                      |
| 变量      | `netcdf.renameVar`        | 更改 netCDF 变量名                        |
| 属性      | `netcdf.copyAtt`          | 将属性复制到新位置                            |
| 属性      | `netcdf.delAtt`           | 删除 netCDF 属性                         |
| 属性      | `netcdf.getAtt`           | 返回 netCDF 属性                         |
| 属性      | `netcdf.inqAtt`           | 返回有关 netCDF 属性的信息                    |
| 属性      | `netcdf.inqAttID`         | 返回 netCDF 属性的 ID                     |
| 属性      | `netcdf.inqAttName`       | 返回 netCDF 属性名称                       |
| 属性      | `netcdf.putAtt`           | 将数据写入 netCDF 属性                      |
| 属性      | `netcdf.renameAtt`        | 更改 netCDF 属性名称                       |
| 用户定义的类型 | `netcdf.defVlen`          | 定义用户定义的可变长度数组类型 `NC_VLEN`，自 R2022a 起 |
| 用户定义的类型 | `netcdf.inqUserType`      | 返回用户定义类型的信息，自 R2022a 起               |
| 用户定义的类型 | `netcdf.inqVlen`          | 返回用户定义的 `NC_VLEN` 类型信息，自 R2022a 起    |
| 实用工具    | `netcdf.getConstant`      | 返回命名常量的数值                            |
| 实用工具    | `netcdf.getConstantNames` | 返回 netCDF 库已知的常量列表                   |
