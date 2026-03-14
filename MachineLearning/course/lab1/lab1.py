from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data(csv_path: Path) -> pd.DataFrame:
    """读取银行营销数据集。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    return pd.read_csv(csv_path, sep=";")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """构建数值+类别特征的预处理器。"""
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )


def plot_confusion_matrices(confusion_results: dict[str, np.ndarray], output_path: Path) -> None:
    """将三个模型的混淆矩阵画在同一张图上。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (model_name, cm) in zip(axes, confusion_results.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{model_name}\nConfusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(roc_results: dict[str, tuple[np.ndarray, np.ndarray, float]], output_path: Path) -> None:
    """将三个模型的ROC曲线画在同一张图上。"""
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, roc_auc) in roc_results.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (3 Algorithms)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "bank-additional-full.csv"

    data = load_data(csv_path)

    # 目标变量编码：yes -> 1, no -> 0
    y = data["y"].map({"no": 0, "yes": 1})
    X = data.drop(columns=["y"])

    class_dist = y.value_counts(normalize=True).sort_index()
    print("类别分布(0=no, 1=yes):")
    print(class_dist)
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    # 三个算法：逻辑回归（核心）+ 随机森林 + SVM
    model_configs = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",  # 处理类别不平衡
                solver="liblinear",  # 避免lbfgs在稀疏高维下出现数值不稳定
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 3.0],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 10],
            },
        },
        "SVM": {
            "estimator": SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["linear", "rbf"],
            },
        },
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    confusion_results: dict[str, np.ndarray] = {}
    roc_results: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for model_name, config in model_configs.items():
        print(f"\n开始训练：{model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", config["estimator"]),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["param_grid"],
            scoring="f1",  # 类别不平衡下优先关注F1
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        print(f"{model_name} 最优参数: {grid.best_params_}")
        print(f"{model_name} CV最佳F1: {grid.best_score_:.4f}")

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        print(f"\n{model_name} classification_report:")
        print(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        confusion_results[model_name] = cm

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_results[model_name] = (fpr, tpr, roc_auc)

        print(f"{model_name} AUC: {roc_auc:.4f}")
        print("-" * 70)

    conf_path = base_dir / "three_algorithms_confusion_matrices.png"
    roc_path = base_dir / "three_algorithms_roc_curves.png"

    plot_confusion_matrices(confusion_results, conf_path)
    plot_roc_curves(roc_results, roc_path)

    print("\n实验完成，结果文件已生成：")
    print(f"1) {conf_path}")
    print(f"2) {roc_path}")


if __name__ == "__main__":
    main()
