from pds4_tools.reader import pds4_read
import numpy as np

'''
有三个 *.xml 文件，这三个文件所在目录一个比一个深
ACS/urn_esa_psa_em16_tgo_acs_data_raw_acs_raw_sc_nir_20250810t234426-20250810t234503-34418-1-1__4.0/em16_tgo_acs/bundle_em16_tgo_acs.xml
ACS/urn_esa_psa_em16_tgo_acs_data_raw_acs_raw_sc_nir_20250810t234426-20250810t234503-34418-1-1__4.0/em16_tgo_acs/data_raw/collection_data_raw.xml
ACS/urn_esa_psa_em16_tgo_acs_data_raw_acs_raw_sc_nir_20250810t234426-20250810t234503-34418-1-1__4.0/em16_tgo_acs/data_raw/Science_Phase/Orbit_Range_34400_34499/Orbit_34418/acs_raw_sc_nir_20250810T234426-20250810T234503-34418-1-1__4_0.xml
'''

import os 
import glob


# base_dir = os.path.dirname()

#label_path = "./urn_esa_psa_em16_tgo_acs_data_raw_acs_raw_sc_nir_20250810t234426-20250810t234503-34418-1-1__4.0/em16_tgo_acs/data_raw/Science_Phase/Orbit_Range_34400_34499/Orbit_34418/acs_raw_sc_nir_20250810T234426-20250810T234503-34418-1-1__4_0.xml"
label_path  = "./urn_esa_psa_em16_tgo_acs_data_raw_acs_raw_sc_nir_20250810t234426-20250810t234503-34418-1-1__4.0"
prod = pds4_read(label_path)

base_dir = os.path.dirname(label_path)

candidates = glob.glob(os.path.join(base_dir, "**", "*.xml"), recursive=True)

bad = ("bundle", "collection", "catalog", "context")

# 1) 列出所有 data structures，先“看见”它们叫什么
for i, s in enumerate(prod.structures):
    print(i, type(s), getattr(s, 'id', None), getattr(s, 'meta_data', None))

# 2) 常见情况：一个 Table_Binary / Table_Delimited 里包含多字段（tangent_height, wavenumber, transmittance...）
#    也可能是 Array + 另一个 Table。下面用“尝试式”取字段：
table = None
for s in prod.structures:
    if s.type.startswith("Table"):
        table = s
        break

# 打印字段名
print(table.dtype.names)

# 3) 按字段名“模糊匹配”取数据（你也可以直接改成 label 里的精确字段名）
def pick(names, candidates):
    for c in candidates:
        for n in names:
            if c.lower() in n.lower():
                return n
    return None

names = table.dtype.names
col_h = pick(names, ["tangent", "tan_height", "tangent_height"])
col_x = pick(names, ["wavenumber", "wn", "wavelength", "lambda"])
col_t = pick(names, ["transmittance", "trans", "i_over_i0", "i_i0"])

h = np.array(table[col_h])                 # [n_obs] 或 [n_tangent]
x = np.array(table[col_x])                 # 可能是 [n_pixel] 或 [n_obs, n_pixel]
T = np.array(table[col_t])                 # 透过率：与 x 维度配套

print(h.shape, x.shape, T.shape)