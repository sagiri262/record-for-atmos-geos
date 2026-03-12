import glob
import xarray as xr
from pathlib import Path
from typing import List, Union, Optional


class WRFDataReader:
    def __init__(self, paths: Union[str, List[str]],
                engine: str="netcdf4", combine: str="nested",
                concat_dim: str = "Time", parallel: bool=False,
                chunks: Optional[dict]=None,
                decode_times: bool=False,
                decode_cf: bool=False):
        self.paths = paths
        self.engine = engine
        self.combine = combine
        self.concat_dim = concat_dim
        self.parallel = parallel
        self.chunks = chunks
        self.decode_times = decode_times
        self.decode_cf = decode_cf

        self.files = self._resolve_files(paths)
        self.ds = None


    def _resolve_files(self, paths: Union[str, List[str]]) -> List[str]:
        '''
        解析输入路径
        - 如果是字符串，展开为文件列表
        - 如果是路径中只有一个文件，直接使用
        - 如果是列表，排序后使用
        '''
        if isinstance(paths, str):
            if any(ch in paths for ch in ["*", "?", "["]):
                files = sorted(glob.glob(paths))
            else:
                p = Path(paths)
                if not p.exists():
                    raise FileNotFoundError(f"文件不存在")
                files = [str[p]]
        elif isinstance():
            files = sorted([str(Path(p)) for p in paths])
        else:
            raise TypeError("Paths 必须是字符串或字符串列表")
        
        if not files:
            raise FileNotFoundError(f"没有匹配到任何文件：{paths}")
        
        return files


    def open(self) -> xr.Dataset:
        """
        打开单文件 - open_dataset
        打开多文件 - open_mfdataset
        """
        if self.ds is not None:
            return self.ds
        
        if len(self.files) == 1:
                self.ds = xr.open_dataset(
                    self.files[0],
                    engine=self.engine,
                    chunks=self.chunks,
                    decode_times=self.decode_times,
                    decode_cf=self.decode_cf,
                )
        else:
            self.ds = xr.open_mfdataset(
            self.files,
                engine=self.engine,
                combine=self.combine,
                concat_dim=self.concat_dim,
                parallel=self.parallel,
            chunks=self.chunks,
                decode_times=self.decode_times,
                decode_cf=self.decode_cf,
            )

        return self.ds


    def close(self):
        if self.ds is not None:
            self.ds.close()
            self.ds = None

    def get_files(self) -> List[str]:
        return self.files

    def get_dataset(self) -> xr.Dataset:
        return self.open()

    def list_vars(self) -> List[str]:
        ds = self.open()
        return list(ds.variables)

    def list_data_vars(self) -> List[str]:
        ds = self.open()
        return list(ds.data_vars)

    def list_coords(self) -> List[str]:
        ds = self.open()
        return list(ds.coords)

    def list_dims(self) -> dict:
        ds = self.open()
        return dict(ds.dims)

    def has_var(self, var_name: str) -> bool:
        ds = self.open()
        return var_name in ds.variables

    def get_var(self, var_name: str):
        ds = self.open()
        if var_name not in ds.variables:
            raise KeyError(f"变量不存在: {var_name}")
        return ds[var_name]

    def get_time_dim(self) -> Optional[int]:
        ds = self.open()
        if "Time" in ds.dims:
            return ds.dims["Time"]
        return None

    def summary(self) -> str:
        ds = self.open()
        lines = []
        lines.append("=== WRFDataReader Summary ===")
        lines.append(f"文件数量: {len(self.files)}")
        lines.append(f"首文件: {self.files[0]}")
        lines.append(f"尾文件: {self.files[-1]}")
        lines.append(f"维度: {dict(ds.dims)}")
        lines.append(f"坐标变量数: {len(ds.coords)}")
        lines.append(f"数据变量数: {len(ds.data_vars)}")
        lines.append(f"前10个数据变量: {list(ds.data_vars)[:10]}")
        return "\n".join(lines)

    def save_var_list(self, outfile: str, data_vars_only: bool = False):
        """
        保存变量列表到文本文件
        """
        vars_out = self.list_data_vars() if data_vars_only else self.list_vars()
        with open(outfile, "w", encoding="utf-8") as f:
            for name in vars_out:
                f.write(name + "\n")