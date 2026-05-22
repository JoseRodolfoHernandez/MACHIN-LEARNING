import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# DATASET
# =========================

data = pd.read_csv("classification_model_dataset.csv")

X = data[["Age", "Salary"]]
Y = data["Purchased"]

# =========================
# SPLIT
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)

# =========================
# TRAINING FUNCTION
# =========================

def train_models():

    results = []

    # =========================
    # DECISION TREE
    # =========================

    dt = DecisionTreeClassifier(max_depth=5)

    dt.fit(X_train, Y_train)

    pred_dt = dt.predict(X_test)

    acc_dt = accuracy_score(Y_test, pred_dt)

    results.append({
        "model": "Decision Tree",
        "accuracy": round(acc_dt * 100, 2)
    })

    # =========================
    # RANDOM FOREST
    # =========================

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5
    )

    rf.fit(X_train, Y_train)

    pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(Y_test, pred_rf)

    results.append({
        "model": "Random Forest",
        "accuracy": round(acc_rf * 100, 2)
    })

    # =========================
    # KNN
    # =========================

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, Y_train)

    pred_knn = knn.predict(X_test)

    acc_knn = accuracy_score(Y_test, pred_knn)

    results.append({
        "model": "KNN",
        "accuracy": round(acc_knn * 100, 2)
    })

    # =========================
    # GRAPH
    # =========================

    names = [
        "Decision Tree",
        "Random Forest",
        "KNN"
    ]

    accuracies = [
        acc_dt * 100,
        acc_rf * 100,
        acc_knn * 100
    ]

    plt.figure(figsize=(8,5))

    plt.bar(names, accuracies)

    plt.title("Model Accuracy Comparison")

    plt.ylabel("Accuracy %")

    plt.savefig("static/model_comparison.png")

    plt.close()

    return results