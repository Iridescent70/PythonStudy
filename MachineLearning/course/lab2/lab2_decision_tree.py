# -*- coding: utf-8 -*-
"""
实验2：基于决策树算法的电信客户流失预测
数据集：WA_Fn-UseC_-Telco-Customer-Churn.csv

核心任务：
1. 训练决策树模型
2. 使用交叉验证（Cross-validation）+ 网格搜索（Grid Search）调优树深度和叶子节点数
3. 输出 classification_report（重点关注 Precision / Recall / F1-Score）
4. 绘制混淆矩阵与 ROC 曲线，并计算 AUC
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


def load_and_preprocess_data(csv_path: Path):
    """读取并完成基础清洗。"""
    df = pd.read_csv(csv_path)

    # 清洗 TotalCharges：原始数据中有空格字符串，需要转成 NaN 再转数值
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # 删除无预测意义的 ID 列
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 目标变量编码：Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    """构建预处理+模型流水线。"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据集文件: {csv_path}")

    # 1) 读取与预处理
    X, y, numeric_features, categorical_features = load_and_preprocess_data(csv_path)

    print("数据集形状:", X.shape)
    print("目标变量分布:")
    print(y.value_counts())
    print("\n数值特征:", numeric_features)
    print("类别特征:", categorical_features)

    # 2) 划分训练集与测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) 构建流水线 + 网格搜索
    pipeline = build_pipeline(numeric_features, categorical_features)

    param_grid = {
        "classifier__max_depth": [3, 5, 7, 9, 12, None],
        "classifier__max_leaf_nodes": [10, 20, 30, 50, 80, None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    print("\n开始网格搜索与交叉验证...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("\n最优参数:", grid_search.best_params_)
    print("最优交叉验证 F1: {:.4f}".format(grid_search.best_score_))

    # 4) 测试集评估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n分类报告（重点关注 Precision / Recall / F1-Score）:")
    print(classification_report(y_test, y_pred, digits=4))

    auc_value = roc_auc_score(y_test, y_prob)
    print("AUC: {:.4f}".format(auc_value))

    # 5) 绘制并保存混淆矩阵
    cm_path = base_dir / "confusion_matrix_decision_tree.png"
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=ax1,
        colorbar=False,
    )
    ax1.set_title("Decision Tree - Confusion Matrix")
    fig1.tight_layout()
    fig1.savefig(cm_path, dpi=150)

    # 6) 绘制并保存 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_path = base_dir / "roc_curve_decision_tree.png"

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.plot(fpr, tpr, linewidth=2, label=f"Decision Tree (AUC = {auc_value:.4f})")
    ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Decision Tree - ROC Curve")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(roc_path, dpi=150)

    print(f"\n混淆矩阵已保存: {cm_path}")
    print(f"ROC 曲线已保存: {roc_path}")


if __name__ == "__main__":
    main()
