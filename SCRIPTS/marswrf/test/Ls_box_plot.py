import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 假数据
x = np.linspace(120, 360, 200)
y = np.linspace(0, 1, 80)
X, Y = np.meshgrid(x, y)
Z = 16 * np.exp(-((X - 280)/25)**2 - ((Y - 0.2)/0.15)**2) + 8*np.exp(-((X - 230)/10)**2 - ((Y - 0.8)/0.08)**2)

fig, ax = plt.subplots(figsize=(8, 3))

# 分级边界
levels = np.arange(2, 18, 2)   # 2,4,...,16
cmap = plt.get_cmap("YlGnBu_r")   # 你可换成更接近图里的配色
norm = mcolors.BoundaryNorm(levels, cmap.N)

# 主图
pcm = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')

ax.set_xlabel("Solar longitude")
ax.set_title("averaged dust mixing ratio")

# 自定义 colorbar 位置：[left, bottom, width, height]
cax = fig.add_axes([0.32, 0.12, 0.28, 0.04])

cb = fig.colorbar(
    pcm,
    cax=cax,
    orientation='horizontal',
    ticks=levels,
    boundaries=levels
)

cb.ax.tick_params(labelsize=8, pad=2)