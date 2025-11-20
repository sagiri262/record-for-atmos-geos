import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import savgol_filter



# 1. 数据加载
asteroid_spectra = pd.read_csv('../asteroid_spectra.csv')
meteorite_spectra = pd.read_csv('../meteorite_spectra.csv')

# 2. 检查数据集形状
print(f"Asteriod spectra shape: {asteroid_spectra.shape}")
print(f"Meteorite spectra shape: {meteorite_spectra.shape}")

# 3. 获取波段列（忽略波长列）
asteroid_band_columns = asteroid_spectra.columns[1:] 
meteorite_band_columns = meteorite_spectra.columns[1:]

# 4. 打印波段列检查
print(f"Asteroid bands: {asteroid_band_columns}")
print(f"Meteorite bands: {meteorite_band_columns}")

# 5. 手动对齐波段：由于编号差异，直接选择共同数量的波段列
num_common_bands = min(len(asteroid_band_columns), len(meteorite_band_columns))

# 6. 对齐波段列
asteroid_spectra_aligned = asteroid_spectra[['Wavelength(um)'] + list(asteroid_band_columns[:num_common_bands])]
meteorite_spectra_aligned = meteorite_spectra[['Wavelength(um)'] + list(meteorite_band_columns[:num_common_bands])]

# 7. 提取特征数据（去掉波长列）
asteroid_features = asteroid_spectra_aligned.iloc[:, 1:].values  # 去掉波长列
meteorite_features = meteorite_spectra_aligned.iloc[:, 1:].values  # 去掉波长列

# 8. 检查对齐后的特征形状
print(f"Aligned Asteroid features shape: {asteroid_features.shape}")
print(f"Aligned Meteorite features shape: {meteorite_features.shape}")

# 9. 合并数据集
X = np.vstack((asteroid_features, meteorite_features))

# 10. 创建标签
y_asteroid = np.zeros(asteroid_features.shape[0])  # 小行星标记为0
y_meteorite = np.ones(meteorite_features.shape[0])  # 陨石标记为1
y = np.concatenate((y_asteroid, y_meteorite))  # 合并标签

# 11. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 12. 数据分割为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 13. K-近邻模型训练
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 14. 预测与评估
y_pred = knn.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# 15. 绘制分类结果的光谱图
plt.figure(figsize=(10, 6))

# 分别绘制小行星与陨石的光谱图
plt.subplot(1, 2, 1)
plt.title("Asteroid Spectra")
plt.plot(asteroid_spectra_aligned.iloc[:, 0], asteroid_spectra_aligned.iloc[:, 1:].mean(axis=1), label='Average Spectrum')
plt.xlabel("Wavelength (um)")
plt.ylabel("Reflectance")
plt.grid()

plt.subplot(1, 2, 2)
plt.title("Meteorite Spectra")
plt.plot(meteorite_spectra_aligned.iloc[:, 0], meteorite_spectra_aligned.iloc[:, 1:].mean(axis=1), label='Average Spectrum', color='orange')
plt.xlabel("Wavelength (um)")
plt.ylabel("Reflectance")
plt.grid()

plt.tight_layout()

# 16. 新需求：单独画一个新图，把两个光谱放在同一幅图

plt.figure(figsize=(8, 6))
plt.plot(asteroid_spectra_aligned.iloc[:, 0],
         asteroid_spectra_aligned.iloc[:, 1:].mean(axis=1),
         label='Asteroid Avg',
         linewidth=2)

plt.plot(meteorite_spectra_aligned.iloc[:, 0],
         meteorite_spectra_aligned.iloc[:, 1:].mean(axis=1),
         label='Meteorite Avg',
         linewidth=2,
         color='orange')

plt.title("Asteroid vs Meteorite Average Spectra")
plt.xlabel("Wavelength (um)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid(True)

plt.show()