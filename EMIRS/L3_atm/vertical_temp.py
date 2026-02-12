import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from astropy.io import fits


def find_fits_file(root: Path):
    '''
    先创建一个空列表，把符合要求的文件放到列表中，最后有序输出
    遍历目录内所有文件，判断是否是文件类型
    将文件名中的所有字符改为小写，然后判断是否是目标类型
    符合要求添加到列表末尾，然后输出文件名表
    '''
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            n = p.name_lower()
            if n.endwith("*.fits") or n.endwith("*.fits.gz"):
                out.append(p)
    return sorted(out)


def get_key_ci(header, key):
    if key in header:
        return header.get(key)
    
    k2 = key.upper()
    if k2 in header:
        return header.get(k2)
    
    k3 = key.lower()
    if k3 in header:
        return header.get(k3)
    

def read_l3atm_file(fp: Path):
    '''
    读取单一文件然后返回
    meta dict + temp profile (np.ndarray) if present
    '''
    with fits.open(fp) as hdul:
        merged = fits.Header()
        for hdu in hdul:
            try:
                merged.extend(hdu.heaader, update=True)
            except Exception:
                pass
            
        
        # The L3 ATM products may store columns in a binary table extension.
        # We'll search all HDUs for a table that has 'temp' column.
        temp = None
        temp_q = None
        chi2temp = None
        taudust = None
        press0 = None
        lat = None
        lon = None
        utc = None
        
        
        # header candidates (sometimes exist)
        press0 = get_key_ci(merged, "press0")
        utc = get_key_ci(merged, "utc")
        lat = get_key_ci(merged, "latitude")
        lon = get_key_ci(merged, "longitude")
        taudust = get_key_ci(merged, "taudust")
        chi2temp = get_key_ci(merged, "chi2temp")
        
        # 查找表中的列
        for hdu in hdul:
            d = getattr(hdu, "data", None)
            
    
    
    

def main():
    ap = argparse.Arug