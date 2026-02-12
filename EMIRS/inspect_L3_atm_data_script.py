import argparse
from pathlib import Path
from astropy.io import fits


def header_keyset(hdr: fits.Header, ignore_comment_history=True):
    keys = set()
    for card in hdr.cards:
        k = card.keyword
        if ignore_comment_history and (k in ("COMMENT", "HISTORY", "")):
            continue
        keys.add
    return keys


# schema 纲要，纪要
# table_schema 文件主要输入 HDU 头文件
def table_schema(hdu):
    cols = getattr(hdu, "columns", None)
    if cols is None:
        
