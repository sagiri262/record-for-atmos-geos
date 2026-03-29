import os, sys
from netCDF4 import Dataset 
from wrf import getvar, ll_to_xy, to_np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from wrf_read_data import WRFDataReader


wrf_path = "../../../WRFV4.6.0/test/em_real/wrfout*"
reader = WRFDataReader(wrf_path)

wrf_file = reader.get_files()

"""
=================================================================
wrf_file 的任务是读取文件名到一个列表
下面的 target_files 就可以使用[:5]这样读取的方法来读取列表中的文件名
=================================================================
"""
#target_files = wrf_file[:5]

target_files = wrf_file[:]

if len(target_files) == 0:
    raise FileNotFoundError("没有找到任何数据！")


for f in target_files:
    print(f"读取文件: {f}")
    ncfile = Dataset(f)

    # 在这里处理当前文件
    # 比如：
    # slp = getvar(ncfile, "slp")
    # ...

    ncfile.close()


'''
======================
读取任意数量的文件
======================
'''

num_files = 10   # 想读多少个就写多少个
target_files = wrf_file [:num_files]


'''
=====================
想要读特定某一天的数据
设定好日期就行
=====================
'''

target_day = "2022-11-26"
target_files = [f for f in wrf_file if target_day in os.path.basename(f)]

# 依旧判空
if len(target_files) == 0:
    raise FileNotFoundError(f"没有找到 {target_day} 的 wrfout 文件。")

print(f"{target_day} 共找到 {len(target_files)} 个文件：")
for f in target_files:
    print(f)