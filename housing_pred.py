import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def create_density_scatter_plot(y_true, y_pred, dataset_name, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：散点密度图
    cmap = LinearSegmentedColormap.from_list('custom_blue', 
           ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
            '#4292c6', '#2171b5', '#08519c', '#08306b'], N=256)
    hb = ax1.hexbin(y_true, y_pred, gridsize=50, cmap=cmap, bins='log', alpha=0.8)
    fig.colorbar(hb, ax=ax1, label='Data Point Density (log scale)')
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--',
             lw=3, label='1:1 Line')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{dataset_name}: Density Scatter Plot')
    ax1.legend()

    # 右图：二维直方图
    hist = ax2.hist2d(y_true, y_pred, bins=50, cmap='viridis', alpha=0.8)
    fig.colorbar(hist[3], ax=ax2, label='Data Point Count')
    ax2.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'w--',
             lw=2, label='1:1 Line')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title(f'{dataset_name}: Histogram Heatmap Plot')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()

def create_marginal_density_plot(y_true, y_pred, dataset_name, filename):
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 4)
    
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # 主图：六边形密度图
    hd = ax_main.hexbin(y_true, y_pred, gridsize=40, 
                        cmap='plasma', bins='log', alpha=0.8)
    ax_main.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
                 'w--', lw=3, label='1:1 Line')
    ax_main.set_xlabel('True Values')
    ax_main.set_ylabel('Predicted Values')
    ax_main.set_title(f'{dataset_name}: Marginal Density Plot')
    ax_main.legend()
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.15, 0.1, 0.3, 0.02])  # [left, bottom, width, height]
    fig.colorbar(hd, cax=cbar_ax, orientation='horizontal', label='Density (log scale)')
    
    # 上方的边缘分布：真实值的密度图
    ax_xhist.tick_params(axis="x", labelbottom=False)
    sns.kdeplot(x=y_true, fill=True, color='skyblue', ax=ax_xhist)
    ax_xhist.set_ylabel('Density')
    
    # 右侧的边缘分布：预测值的密度图
    ax_yhist.tick_params(axis="y", labelleft=False)
    sns.kdeplot(y=y_pred, fill=True, color='salmon', ax=ax_yhist)
    ax_yhist.set_xlabel('Density')
    
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()

# 加载加利福尼亚住房数据
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['target'] = housing.target

# 添加一些seaborn可视化来探索数据
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制目标变量的分布
plt.figure(figsize=(10, 6))
sns.histplot(data['target'], kde=True, bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Price (in $100,000s)')
plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

X = data.drop(columns=["target"])
Y = data["target"]

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 训练 XGBoost 模型
xgb_model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, Y_train)

Y_train_pred = xgb_model.predict(X_train)
Y_test_pred = xgb_model.predict(X_test)

# 评估指标
# 计算R^2
train_metrics = {'R^2': r2_score(Y_train, Y_train_pred),
                 'MAE': mean_absolute_error(Y_train, Y_train_pred)}
test_metrics = {'R^2': r2_score(Y_test, Y_test_pred),
                'MAE': mean_absolute_error(Y_test, Y_test_pred)}

print("Training Metrics:", train_metrics)
print("Test Metrics:", test_metrics)

# 创建可视化
create_density_scatter_plot(Y_train, Y_train_pred, "Training Set", "train_density_scatter.png")
create_density_scatter_plot(Y_test, Y_test_pred, "Test Set", "test_density_scatter.png")
create_marginal_density_plot(Y_train, Y_train_pred, "Training Set", "train_marginal_density.png")
create_marginal_density_plot(Y_test, Y_test_pred, "Test Set", "test_marginal_density.png")

# 使用seaborn绘制残差图
plt.figure(figsize=(10, 6))
residuals = Y_test - Y_test_pred
sns.residplot(x=Y_test_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Test Set')
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 特征重要性可视化
feature_importance = pd.DataFrame({
    'feature': housing.feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()