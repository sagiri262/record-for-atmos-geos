#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./extract_registry_io_i.sh Registry/Registry.EM output_dir
# 如果不写 output_dir，默认输出到当前目录

infile="${1:?用法: $0 <Registry文件> [输出目录]}"
outdir="${2:-.}"

mkdir -p "$outdir"

txt_out="$outdir/registry_io_contains_i.txt"
csv_out="$outdir/registry_io_contains_i.csv"

printf 'Table,Type,Sym,Dims,Use,NumTLev,Stagger,IO,DNAME,DESCRIP,UNITS\n' > "$csv_out"
: > "$txt_out"

