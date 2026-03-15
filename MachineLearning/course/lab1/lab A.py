# -*- coding: utf-8 -*-
"""
实验1：基于逻辑回归的银行营销结果预测
数据集：bank-additional-full.csv

功能：
1. 训练逻辑回归（Logistic Regression）
2. 使用交叉验证（Cross-validation）和网格搜索（Grid Search）进行超参数调优
3. 输出分类报告，重点分析 Precision / Recall / F1-score
4. 对三个算法（逻辑回归、决策树、随机森林）绘制：
   - 混淆矩阵
   - ROC 曲线
   - 计算 AUC
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score
)

# =========================
# 1. 读取数据
# =========================
# 注意：bank-additional-full.csv 通常分隔符是 ";"
file_path = "bank-additional-full.csv"
df = pd.read_csv(file_path, sep=';')

print("数据集前5行：")
print(df.head())
print("\n数据集形状：", df.shape)
print("\n目标变量分布：")
print(df['y'].value_counts())

# =========================
# 2. 数据预处理
# =========================
# 目标变量转换：yes -> 1, no -> 0
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop('y', axis=1)
y = df['y']

# 区分数值特征和类别特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\n数值特征：", numeric_features)
print("\n类别特征：", categorical_features)

# 数值特征处理：缺失值填充 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别特征处理：缺失值填充 + OneHot编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =========================
# 3. 划分训练集和测试集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 保持类别比例一致
)

print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# =========================
# 4. 定义模型
# =========================
# 逻辑回归（核心模型）
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000, random_state=42))
])

# 决策树（对比模型1）
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
])

# 随机森林（对比模型2）
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# =========================
# 5. 逻辑回归：网格搜索 + 交叉验证
# =========================
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=log_reg_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',   # 类别不平衡时，优先关注F1
    n_jobs=-1,
    verbose=1
)

print("\n开始进行逻辑回归网格搜索和交叉验证...")
grid_search.fit(X_train, y_train)

print("\n逻辑回归最优参数：")
print(grid_search.best_params_)
print("逻辑回归最优交叉验证 F1：{:.4f}".format(grid_search.best_score_))

best_log_reg = grid_search.best_estimator_

# =========================
# 6. 训练另外两个对比模型
# =========================
dt_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# =========================
# 7. 定义评估函数
# =========================
def evaluate_model(model, X_test, y_test, model_name):
    """
    输出：
    - classification_report
    - confusion_matrix
    - ROC曲线数据
    - AUC
    """
    y_pred = model.predict(X_test)

    # 某些模型支持 predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # 极少情况下 fallback
        y_prob = model.decision_function(X_test)

    print("\n" + "=" * 60)
    print(f"{model_name} 分类报告：")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    return {
        "name": model_name,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score
    }

# =========================
# 8. 评估三个模型
# =========================
results = []
results.append(evaluate_model(best_log_reg, X_test, y_test, "Logistic Regression"))
results.append(evaluate_model(dt_pipeline, X_test, y_test, "Decision Tree"))
results.append(evaluate_model(rf_pipeline, X_test, y_test, "Random Forest"))

# =========================
# 9. 绘制三个模型的混淆矩阵
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, result in zip(axes, results):
    disp = ConfusionMatrixDisplay(confusion_matrix=result["cm"], display_labels=[0, 1])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{result["name"]}\nConfusion Matrix')

plt.tight_layout()
plt.show()

# =========================
# 10. 绘制三个模型的ROC曲线
# =========================
plt.figure(figsize=(10, 7))

for result in results:
    plt.plot(
        result["fpr"],
        result["tpr"],
        linewidth=2,
        label=f'{result["name"]} (AUC = {result["auc"]:.4f})'
    )

plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Three Models")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# =========================
# 11. 输出AUC汇总
# =========================
print("\n" + "=" * 60)
print("三个模型的AUC值：")
for result in results:
    print(f'{result["name"]}: AUC = {result["auc"]:.4f}')
