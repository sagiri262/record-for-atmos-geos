## WRF Pre-process System

包括  ./geogrid.exe   ./ungrib.exe   ./metgrid.exe 

*./geogrid.exe*的作用：
选择模型域将地面数据水平插值到每个域的模型网格上
1、将经纬度坐标系转为网格坐标系，网格坐标系的单位为长度（namelist.wps中是米为单位），且因为经纬度网格分辨率较大，还需要进行插值
2.将3D球面弧形坐标系投影为3D直角坐标系。对于地面数据，也称为静态数据，其包括土壤类别、土地利用率、地形高度、土壤温度、植被数、反照率等，数据分辨率率不大相同，也插值到模式网格点上。


*./ungrib.exe*
从原始数据解码grib数据
气象场数据原始格式常为 GRIB1 和 GRIB2，如GFS、ERA5、MERRA2等等。NetCDF数据无法单独读取，需要将netCDF文件里的变量一一对应到WRF可以识取的变量名，这里就是把变量写到 Vtable 里，如 ua 对应到 WRF 中是 UU。

*./metgrid.exe*
水平插值满足数据到模型网格的每个domain里
将上面 ungrib 的得到的气象数据与 geogrid 的得到的水平插值静态地理数据进行水平场插值。这个插值是在现有的气压层上进行2D水平插值，
对应的METGRID.TBL文件定义了要用网格插值的每个气象场的参数，只有参数在文件中定义了，才能被metgrid读取并插值。



