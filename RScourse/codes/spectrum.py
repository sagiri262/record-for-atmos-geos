import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from scipy.signal import savgol_filter


# ======================
# 1. 读取数据
# ======================
asteroid_spectra = pd.read_csv('../asteroid_spectra.csv')
meteorite_spectra = pd.read_csv('../meteorite_spectra.csv')

print(f"Asteriod spectra shape: {asteroid_spectra.shape}")
print(f"Meteorite spectra shape: {meteorite_spectra.shape}")

# 第一列是波长
wavelength = asteroid_spectra['Wavelength(um)'].values
# 保险起见，检查两者波长是否一致
if not np.allclose(wavelength, meteorite_spectra['Wavelength(um)'].values):
    print("Warning: asteroid 和 meteorite 的波长轴不完全一致，请注意！")

# 各自的样本列名（每一列 = 一个光谱样本）
asteroid_band_columns = asteroid_spectra.columns[1:]
meteorite_band_columns = meteorite_spectra.columns[1:]

print(f"Asteroid bands: {len(asteroid_band_columns)}")
print(f"Meteorite bands: {len(meteorite_band_columns)}")

# 为了让维度一致，取两者样本数的较小值
num_common_samples = min(len(asteroid_band_columns), len(meteorite_band_columns))
asteroid_used_cols = asteroid_band_columns[:num_common_samples]
meteorite_used_cols = meteorite_band_columns[:num_common_samples]

# 取出对应的子表（包含波长 + 对应的样本列）
asteroid_spectra_used = asteroid_spectra[['Wavelength(um)'] + list(asteroid_used_cols)]
meteorite_spectra_used = meteorite_spectra[['Wavelength(um)'] + list(meteorite_used_cols)]

# ======================
# 2. 构建样本矩阵
# ======================
# DataFrame 形状： (n_wavelengths, n_samples)
# 我们希望做分类时：X 形状 = (n_samples, n_wavelengths)
asteroid_samples = asteroid_spectra_used.iloc[:, 1:].T.values  # (n_asteroid_samples, n_wavelengths)
meteorite_samples = meteorite_spectra_used.iloc[:, 1:].T.values  # (n_meteorite_samples, n_wavelengths)

print(f"Asteroid samples matrix shape: {asteroid_samples.shape}")
print(f"Meteorite samples matrix shape: {meteorite_samples.shape}")

# 合并为一个大的样本矩阵
X_orig = np.vstack((asteroid_samples, meteorite_samples))  # (n_total_samples, n_wavelengths)
y = np.concatenate((
    np.zeros(asteroid_samples.shape[0]),   # 小行星 = 0
    np.ones(meteorite_samples.shape[0])    # 陨石 = 1
))

# ======================
# 3. Savitzky–Golay 平滑（作用在每条光谱上）
# ======================
window_length = 11   # 必须是奇数
polyorder = 3

if window_length % 2 == 0:
    window_length += 1

X_smooth = savgol_filter(
    X_orig,
    window_length=window_length,
    polyorder=polyorder,
    axis=1   # 沿着波长方向平滑
)

# 分类时可以用平滑后的光谱
X_for_clf = X_smooth

# ======================
# 4. 标准化 + KNN 分类
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_for_clf)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("====== 分类结果 ======")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=['Asteroid', 'Meteorite']))

# ======================
# 5. 画四幅图：原始光谱 / 平滑光谱 / 一阶导 / 二阶导
# ======================
# 选一条小行星光谱做例子（你也可以改成某条陨石光谱）
example_spec = asteroid_samples[0]              # 原始光谱
example_smooth = savgol_filter(
    example_spec,
    window_length=window_length,
    polyorder=polyorder
)

# 一般建议对“平滑后”的光谱求导，以减弱噪声影响
first_deriv = np.gradient(example_smooth, wavelength)
second_deriv = np.gradient(first_deriv, wavelength)

# 1. 原始光谱
plt.figure(figsize=(8, 5))
plt.plot(wavelength, example_spec)
plt.title("Original Spectrum (Example Asteroid)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflectance")
plt.grid(True)

# 2. 平滑光谱
plt.figure(figsize=(8, 5))
plt.plot(wavelength, example_smooth)
plt.title("Smoothed Spectrum (Savitzky–Golay)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflectance")
plt.grid(True)

# 3. 一阶导数光谱
plt.figure(figsize=(8, 5))
plt.plot(wavelength, first_deriv)
plt.title("1st Derivative of Smoothed Spectrum")
plt.xlabel("Wavelength (µm)")
plt.ylabel("dR/dλ")
plt.grid(True)

# 4. 二阶导数光谱
plt.figure(figsize=(8, 5))
plt.plot(wavelength, second_deriv)
plt.title("2nd Derivative of Smoothed Spectrum")
plt.xlabel("Wavelength (µm)")
plt.ylabel("d²R/dλ²")
plt.grid(True)

plt.tight_layout()
plt.show()
