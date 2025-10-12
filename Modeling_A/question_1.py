import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import openpyxl

# —— 1. 已知参数 —— #
# Heave (垂荡)
m_heave = 1335.535          # 附加质量 (kg)
d_heave = 656.3616          # 兴波阻尼系数 (N·s/m)
f_heave_amp = 6250.0        # 垂荡激励力振幅 (N)

# Pitch (纵摇)
I_pitch = 6779.315          # 附加转动惯量 (kg·m^2)
d_pitch = 151.4388          # 兴波阻尼系数 (N·m·s/rad)
f_pitch_amp = 1230.0        # 纵摇激励力矩振幅 (N·m)

# 波频
omega = 1.4005              # rad/s

# 时间设定
T = 2 * np.pi / omega
t_final = 40 * T            # 前 40 个周期
dt = 0.2                    # 时间步长
t_eval = np.arange(0, t_final + 1e-8, dt)

# —— 2. 微分方程组定义 —— #
# 状态变量 y = [z, zdot, phi, phidot]
def motion_equations(t, y):
    z, z_dot, phi, phi_dot = y
    
    # 外部波浪激励
    F_heave = f_heave_amp * np.cos(omega * t)
    M_pitch = f_pitch_amp * np.cos(omega * t)
    
    # 阻尼力计算
    D_heave = d_heave * z_dot
    D_pitch = d_pitch * phi_dot
    
    # 方程（不含刚度项假设）
    dz_ddot = (F_heave - D_heave) / m_heave
    dphi_ddot = (M_pitch - D_pitch) / I_pitch
    
    return [z_dot, dz_ddot, phi_dot, dphi_ddot]

# —— 3. 初始条件 ---- #
y0 = [0.0, 0.0, 0.0, 0.0]  # 初始位置与速度均为0

# —— 4. 求解 —— #
sol = solve_ivp(motion_equations, [0, t_final], y0, t_eval=t_eval, method='RK45')

t = sol.t
z = sol.y[0]
z_dot = sol.y[1]
phi = sol.y[2]
phi_dot = sol.y[3]

# —— 5. 保存 Excel 结果 —— #
df_heave = pd.DataFrame({'time': t, 'heave_disp': z, 'heave_vel': z_dot})
df_pitch = pd.DataFrame({'time': t, 'pitch_ang': phi, 'pitch_vel': phi_dot})

df_heave.to_excel('result1_1.xlsx', index=False)
df_pitch.to_excel('result1_2.xlsx', index=False)

# —— 6. 在论文中输出指定时刻结果 —— #
specific_times = [10, 20, 40, 60, 100]
print("Time(s) | Heave disp (m) | Heave vel (m/s) | Pitch (rad) | Pitch vel (rad/s)")
for ts in specific_times:
    idx = np.argmin(np.abs(t - ts))
    print(f"{t[idx]:6.1f} | {z[idx]:14.6f} | {z_dot[idx]:14.6f} | {phi[idx]:12.6f} | {phi_dot[idx]:14.6f}")
