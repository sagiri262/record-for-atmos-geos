;=======================================================================
; MarsWRF TAU_OD2D  Ls–Lat 图（等高线平滑版）—— NCL
; -----------------------------------------------------------------------
; 核心改进：
;   旧版：分箱 → gsn_csm_contour（RasterFill 有格点块状感）
;   新版：散点 → 精细分箱均值 → smth9_n 平滑 → cnFillMode="AreaFill"
;         (连续等高线填色，天然平滑，无格点锯齿)
;
; NCL 平滑方案：
;   smth9_n   — 九点加权平滑，可多次迭代，适合规则格点
;   (备选) wgt_runave_n — 可变长度滑动平均
;
; 使用：
;   ncl marswrf_tau_od2d_smooth_ncl.ncl \
;       'file_pattern="wrfout_d01_*"' \
;       'ls_bins=180' 'lat_bins=90' \
;       'smooth_iter=10' \
;       'co2_thresh=1.0' \
;       'out_file="tau_od2d_smooth.png"'
;=======================================================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin

  ;---------------------------------------------------------------------
  ; 0. 用户参数
  ;---------------------------------------------------------------------
  if (.not. isvar("file_pattern")) then
    file_pattern = "wrfout_d01_*"
  end if
  if (.not. isvar("ls_bins")) then
    ls_bins  = 180        ; 每 2° 一箱，分辨率更高
  end if
  if (.not. isvar("lat_bins")) then
    lat_bins = 90
  end if
  if (.not. isvar("smooth_iter")) then
    smooth_iter = 10      ; smth9_n 迭代次数；越多越平滑
  end if
  if (.not. isvar("smooth_p")) then
    smooth_p = 0.5        ; smth9_n 参数 p（0~1，越大平滑越强）
  end if
  if (.not. isvar("smooth_q")) then
    smooth_q = -0.25      ; smth9_n 参数 q（负值抑制过平滑）
  end if
  if (.not. isvar("co2_thresh")) then
    co2_thresh = 1.0
  end if
  if (.not. isvar("tau_vmin")) then
    tau_vmin = 0.0
  end if
  if (.not. isvar("tau_vmax")) then
    tau_vmax = 3.0
  end if
  if (.not. isvar("nlevs")) then
    nlevs = 50
  end if
  if (.not. isvar("out_file")) then
    out_file = "tau_od2d_smooth.png"
  end if

  ;---------------------------------------------------------------------
  ; 1. 文件列表
  ;---------------------------------------------------------------------
  flist  = systemfunc("ls " + file_pattern + " 2>/dev/null")
  nfiles = dimsizes(flist)
  if (nfiles .eq. 0) then
    print("ERROR: 未找到文件 " + file_pattern)
    exit
  end if
  print("找到文件数 = " + nfiles)

  ;---------------------------------------------------------------------
  ; 2. 读取网格信息（从第一个文件）
  ;---------------------------------------------------------------------
  f0     = addfile(flist(0), "r")
  xlat0  = f0->XLAT(0,:,:)              ; (SN, WE)
  n_sn   = dimsizes(xlat0(:,0))
  ; 代表纬度 = 经向均值
  lat_src = dim_avg_n(xlat0, 1)         ; (SN,)
  delete([/f0, xlat0/])

  ;---------------------------------------------------------------------
  ; 3. 分箱网格定义
  ;---------------------------------------------------------------------
  ls_step  = 360.0 / ls_bins
  lat_step = 180.0 / lat_bins

  ls_ctr  = fspan(ls_step  * 0.5, 360.0 - ls_step  * 0.5, ls_bins)
  lat_ctr = fspan(-90.0 + lat_step * 0.5, 90.0 - lat_step * 0.5, lat_bins)

  ; 累加数组（lat_bins × ls_bins）
  tau_sum   = new((/lat_bins, ls_bins/), double, 0.0d)
  co2_sum   = new((/lat_bins, ls_bins/), double, 0.0d)
  count_arr = new((/lat_bins, ls_bins/), integer, 0)

  ;---------------------------------------------------------------------
  ; 4. 遍历文件/时次，散点累加到分箱
  ;---------------------------------------------------------------------
  do ifile = 0, nfiles - 1
    print("  处理：" + flist(ifile))
    fi     = addfile(flist(ifile), "r")
    ntimes = dimsizes(fi->L_S)

    do it = 0, ntimes - 1
      ; ── Ls 箱索引
      ls_val = doubletofloat(fi->L_S(it)) mod 360.0
      i_ls   = floattointeger(ls_val / ls_step)
      if (i_ls .ge. ls_bins) then
        i_ls = ls_bins - 1
      end if

      ; ── TAU_OD2D 纬向均值 → (SN,)
      tau_z = dim_avg_n(fi->TAU_OD2D(it,:,:), 1)   ; (SN,)
      co2_z = dim_avg_n(fi->CO2ICE(it,:,:),   1)

      ; ── 将每个 SN 格点插值到均匀纬度格，再累加
      tau_interp = linint1(lat_src, tau_z, False, lat_ctr, 0)
      co2_interp = linint1(lat_src, co2_z, False, lat_ctr, 0)

      do i_lat = 0, lat_bins - 1
        if (.not. ismissing(tau_interp(i_lat))) then
          tau_sum(i_lat, i_ls)   = tau_sum(i_lat, i_ls)   + tau_interp(i_lat)
          co2_sum(i_lat, i_ls)   = co2_sum(i_lat, i_ls)   + co2_interp(i_lat)
          count_arr(i_lat, i_ls) = count_arr(i_lat, i_ls) + 1
        end if
      end do

      delete([/tau_z, co2_z, tau_interp, co2_interp/])
    end do
    delete([/fi, ntimes/])
  end do

  ;---------------------------------------------------------------------
  ; 5. 计算分箱均值
  ;---------------------------------------------------------------------
  _FillVal = 1e20
  tau_bin  = new((/lat_bins, ls_bins/), float, _FillVal)
  co2_bin  = new((/lat_bins, ls_bins/), float, _FillVal)

  tau_bin  = where(count_arr .gt. 0,  \
                   floattodouble(tau_sum) / count_arr, _FillVal)
  co2_bin  = where(count_arr .gt. 0,  \
                   floattodouble(co2_sum) / count_arr, _FillVal)

  ; ── 对稀疏缺失值做简单邻居填补
  ; NCL 内置 linmsg 可对一维缺失值线性插值
  ; 对二维，逐行、逐列做 linmsg 迭代
  do iter = 0, 2
    do ilat = 0, lat_bins - 1
      tau_bin(ilat, :) = linmsg(tau_bin(ilat, :), 0)
      co2_bin(ilat, :) = linmsg(co2_bin(ilat, :), 0)
    end do
    do ils = 0, ls_bins - 1
      tau_bin(:, ils) = linmsg(tau_bin(:, ils), 0)
      co2_bin(:, ils) = linmsg(co2_bin(:, ils), 0)
    end do
  end do

  ; 仍为缺失值的位置设为0（边界情况）
  tau_bin = where(ismissing(tau_bin), 0.0, tau_bin)
  co2_bin = where(ismissing(co2_bin), 0.0, co2_bin)

  ;---------------------------------------------------------------------
  ; 6. smth9_n 二维平滑
  ;    smth9_n(x, p, q, cyclic_lat, cyclic_lon, niter, dims)
  ;    dims=(/0,1/) → 同时对两个维度平滑
  ;    cyclic=False/True → Ls 方向设 True（首尾相连）
  ;
  ;    注意：smth9_n 期望 (...,lat,lon) 维序
  ;    此处我们把 ls 视为 "lon"（周期维），lat 为 "lat"
  ;    tau_bin(lat_bins, ls_bins) → lat 对应 dim0，ls 对应 dim1
  ;---------------------------------------------------------------------
  tau_smooth = smth9_n(tau_bin, smooth_p, smooth_q, \
                       False, True, smooth_iter, (/0,1/))
  co2_smooth = smth9_n(co2_bin, smooth_p, smooth_q, \
                       False, True, smooth_iter, (/0,1/))

  ;---------------------------------------------------------------------
  ; 7. 附加坐标元数据（NCL 绘图需要坐标变量）
  ;---------------------------------------------------------------------
  tau_smooth!0  = "lat"
  tau_smooth!1  = "ls"
  tau_smooth&lat = lat_ctr
  tau_smooth&ls  = ls_ctr
  tau_smooth&lat@units = "degrees_north"
  tau_smooth&ls@units  = "degrees"

  co2_smooth!0  = "lat"
  co2_smooth!1  = "ls"
  co2_smooth&lat = lat_ctr
  co2_smooth&ls  = ls_ctr

  ;---------------------------------------------------------------------
  ; 8. 绘图
  ;---------------------------------------------------------------------
  ; 判断输出格式
  sfx = str_get_cols(out_file, -3, -1)
  wks_type = "png"
  if (sfx .eq. "pdf") then
    wks_type = "pdf"
  end if
  if (sfx .eq. "eps") then
    wks_type = "eps"
  end if
  wks = gsn_open_wks(wks_type, str_sub_str(out_file, "." + sfx, ""))
  gsn_define_colormap(wks, "MPL_jet")

  ; ── 主资源
  res = True

  ; 图框
  res@gsnMaximize         = True
  res@vpWidthF            = 0.80
  res@vpHeightF           = 0.38

  ; 坐标范围
  res@trXMinF             = 0.0
  res@trXMaxF             = 360.0
  res@trYMinF             = -90.0
  res@trYMaxF             = 90.0

  ; ── 关键：使用 AreaFill 而非 RasterFill
  ;    AreaFill = 等高线面填充，天然平滑无格点
  res@cnFillOn            = True
  res@cnFillMode          = "AreaFill"    ; <── 平滑关键设置
  res@cnLinesOn           = False         ; 不显示等高线边框
  res@cnLineLabelsOn      = False
  res@cnInfoLabelOn       = False

  ; 等值线级别
  res@cnLevelSelectionMode = "EqualSpacingF"
  res@cnMinLevelValF       = tau_vmin
  res@cnMaxLevelValF       = tau_vmax
  res@cnLevelSpacingF      = (tau_vmax - tau_vmin) / nlevs

  ; X 轴
  res@tiXAxisString        = "Solar Longitude  L~B~s~N~ (~S~o~N~)"
  res@tiXAxisFontHeightF   = 0.018
  res@tmXBMode             = "Explicit"
  res@tmXBValues           = (/0., 45., 90., 135., 180., 225., 270., 315., 360./)
  res@tmXBLabels           = (/"0","45","90","135","180","225","270","315","360"/)
  res@tmXBMinorValues      = ispan(0,360,15)*1.0
  res@tmXBMinorOn          = True
  res@tmXTOn               = True
  res@tmXTLabelsOn         = False

  ; Y 轴
  res@tiYAxisString        = "Latitude (~S~o~N~)"
  res@tiYAxisFontHeightF   = 0.018
  res@tmYLMode             = "Explicit"
  res@tmYLValues           = (/-90., -60., -30., 0., 30., 60., 90./)
  res@tmYLLabels           = (/"90~S~o~N~S","60~S~o~N~S","30~S~o~N~S", \
                                "0~S~o~N~",  "30~S~o~N~N","60~S~o~N~N","90~S~o~N~N"/)
  res@tmYLMinorValues      = ispan(-90,90,10)*1.0
  res@tmYLMinorOn          = True
  res@tmYROn               = True
  res@tmYRLabelsOn         = False

  ; 刻度向内
  res@tmXBTickSpacingF     = 45.0

  ; 色标
  res@lbOrientation              = "Horizontal"
  res@lbBoxLinesOn               = False
  res@lbTitleString              = "Dust Column Optical Depth  (TAU_OD2D)"
  res@lbTitleFontHeightF         = 0.012
  res@lbLabelFontHeightF         = 0.011
  res@pmLabelBarOrthogonalPosF   = 0.14

  ; 标题
  res@tiMainString         = "MarsWRF  Daily Mean Dust Optical Depth  (smooth AreaFill)"
  res@tiMainFontHeightF    = 0.018

  ; ── 绘制主图（TAU_OD2D 平滑填色）
  plot = gsn_csm_contour(wks, tau_smooth, res)

  ; ── 叠加 CO2 极冠白线
  res_co2                        = True
  res_co2@cnFillOn               = False
  res_co2@cnLinesOn              = True
  res_co2@cnLineColor            = "white"
  res_co2@cnLineThicknessF       = 2.5
  res_co2@cnLevelSelectionMode   = "ExplicitLevels"
  res_co2@cnLevels               = (/co2_thresh/)
  res_co2@cnInfoLabelOn          = False
  res_co2@cnLineLabelsOn         = False
  res_co2@gsnDraw                = False
  res_co2@gsnFrame               = False

  plot_co2 = gsn_csm_contour(wks, co2_smooth, res_co2)
  overlay(plot, plot_co2)

  draw(plot)
  frame(wks)

  print("图片已保存：" + out_file)

end
