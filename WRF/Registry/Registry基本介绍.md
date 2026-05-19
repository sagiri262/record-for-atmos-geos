目的：
控制WRF输出 （wrfout*, 也称为history文件）， 特别是生成辅助输出文件（auxhist*）以及控制输出频率。有时候，WRF输出许多不需要的变量，或者输出太过频繁，占用空间太大； 有时候，WRF输出又不够频繁，有些量没有输出，这些都可以利用以下的步骤解决。

# 1. Registry简介
WRF内部变量主要通过Registry来控制（WRF/Registry）。与real run相对应的是Registry.EM。

该文件每一行分为9列：
Type   Sym   Dims   Use   Tlev   Stag   IO   Dname   Descrip
每个变量对应一列，比如：
state   real   u   ikjb   dyn_em   2   X   i01rhusdf   "U"   "X WIND COMPONENT"
这一行就对应WRF中所用的U变量

控制输出的在于IO一栏（i01rhusdf），
- i: 输入。
- r: 重启(restart)输出。如果在IO栏中有"r", 则该变量就会输出到wrfrst*文件中，以便于未来restart run(hot start)时作为初始条件使用。
- h：输出，即输出到history文件中。
- usdf：控制nesting（互相套合的，就是嵌套吧）选项
我们可以看到，对输出的控制集中在"h"字符上。

举例说明，
- 如果删除h, 则U变量将不会输出到history文件中（wrfout*），也不会有任何的辅助输出。
- 如果改为h1, 则U变量将不会输出到history主文件中（wrfout*）, 但会输出到辅助输出文件1中。
- 如果改为h01，则U变量既会输出到history主文件中（wrfout*），同时也会输出到辅助输出文件1中。
- h之后可以跟0(主输出)，也可以跟"1,2,3,4,5,6,7,8,9",对应1到9号辅助输出文件。以下写法，都是合法的：
*h, h01, h012, h347*

注意：每次对Registry进行修改后，必须对WRF进行重新编译(compile)，即：
$./clean -a
$./compile em_real

# 2. 修改namelist.input
举例说明，在Registry中修改rainc和rainnc输出为h02后，要在run/namelist.input中修改添加以下变量：
auxhist2_outname = "rainfall_d<domain>_<date>"      #辅助文件2的命名
auxhist2_interval    = 60                                               #变量输出的间隔，以分钟为单位
io_form_auxhist2    = 2                                                 #辅助文件2的输出格式，2对应netcdf
frame_per_auxhist2 = 1                                                #每个文件中包含的输出数

如果想要修改输出频率，可以在这里更改auxhist2_interval
注意： 如果没有添加这些变量在namelist.input中，辅助输出文件（rainfall*）还是不能生成。
=====================================================================

Registry实际上是一个利用PERL来自动生成WRF源代码的输入文件。这也是每次修改Registry之后要重新编译WRF的原因。在对Registry进行修改之前，有必要对原文件进行备份。