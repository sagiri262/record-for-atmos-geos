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
    ax1.set_title(f'{dataset_name}: Density Scatter Plot'); ax1.legend()

    hist = ax2.hist2d(y_true, y_pred, bins=50, cmap='viridis', alpha=0.8)
    fig.colorbar(hist[3], ax=ax2, label='Data Point Count')
    ax2.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'w--',
             lw=2, abel='1:1 Line')
    ax2.set_title(f'{dataset_name}: Histogram Heatmap Plot'); ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()


def create_marginal_density_plot(y_true, y_pred, dataset_name, filename):
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 4)
    
    ax_main = fig.add_subplotgs(gs[1:4, 0:3])
    ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    hd = ax_main.hexbin(y_true, y_pred, gridsize=40, 
                        cmap='plasma', bins='log', alpha=0.8)
    ax_main.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
                 'w--', lw=3, lable='1:1 Line')
    sns.kdeplot(x=y_true, fill=True, color='skyblue', ax=ax_xhist)
    sns.kdeplot(y=y_pred, fill=True, color='salmon', ax=ax_yhist)
    plt.savefig(filename, dpi=600, bbox_inches='tight'); plt.show();


# load california housing data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['target'] = housing.target
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
                 'MAEN':mean_absolute_error(Y_train, Y_train_pred)}
test_metrics = {'R^2': r2_score(Y_test, Y_test_pred),
                'MAEN':mean_absolute_error(Y_test, Y_test_pred)}

print(train_metrics)
print(test_metrics)

