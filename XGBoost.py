import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据准备和清洗
# 加载数据
data = pd.read_csv('data/data_v0.csv')  # 替换为实际文件路径

# 检查缺失值并清洗
print("数据概况：")
print(data.info())
data = data.dropna()  # 删除缺失值
print("清洗后的数据形状：", data.shape)

# 定义特征列和目标列
feature_columns = ['PR', 'BH_avg', 'APAR', 'RD', 'poi_diversity',
                   'dominant_function_ratio', 'PD', 'ED', 'LPI', 'nli']
target_column = 'LST_mean'

X = data[feature_columns]
y = data[target_column]

# 2. 特征工程
# 标准化特征
scaler = StandardScaler() #! condition 1
X_scaled = scaler.fit_transform(X)

random_state = 42
# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

# 4. 模型选择与训练
# 初始化并训练 XGBoost 模型
xgb_model = XGBRegressor(random_state=random_state, n_estimators=100, learning_rate=0.1, max_depth=6) #! condition 2
xgb_model.fit(X_train, y_train)

# 5. 模型评估
# 预测测试集
y_pred = xgb_model.predict(X_test)

# 评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差 (MSE): {mse}")
print(f"决定系数 (R²): {r2}")

# 可视化真实值与预测值
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('True LST')
plt.ylabel('Predicted LST')
plt.title('True vs Predicted LST')
plt.show()

# 6. 特征重要性分析
# 提取特征重要性
feature_importances = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = [feature_columns[i] for i in sorted_idx]
sorted_importances = feature_importances[sorted_idx]

print("\n特征重要性排序:")
for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance:.4f}")

correlation_matrix = data.corr()
print(correlation_matrix['LST_mean'].sort_values(ascending=False))

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.show()

import matplotlib.pyplot as plt
for col in feature_columns:
    plt.scatter(data[col], data['LST_mean'], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel('LST_mean')
    plt.title(f'{col} vs LST_mean')
    plt.show()

import seaborn as sns
sns.histplot(data['LST_mean'], kde=True)
