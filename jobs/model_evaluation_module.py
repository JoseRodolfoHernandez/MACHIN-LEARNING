import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "social_ads.csv")

static_path = os.path.join(BASE_DIR, "static")

os.makedirs(static_path, exist_ok=True)

# ==========================================
# DATASET
# ==========================================
data = pd.read_csv(data_path)

X = data[["Age", "EstimatedSalary"]]

y = data["Purchased"]

# ==========================================
# TRAIN / TEST
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

best_predictions = None
best_model_name = ""

# ==========================================
# TRAINING LOOP
# ==========================================
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2)
    })

# ==========================================
# BEST MODEL
# ==========================================
best_model = max(results, key=lambda x: x["Accuracy"])

best_model_name = best_model["Model"]

# ==========================================
# GET BEST PREDICTIONS
# ==========================================
best_model_object = models[best_model_name]

best_model_object.fit(X_train, y_train)

best_predictions = best_model_object.predict(X_test)

# ==========================================
# ACCURACY GRAPH
# ==========================================
models_names = [r["Model"] for r in results]

accuracy_values = [r["Accuracy"] for r in results]

plt.figure(figsize=(8, 5))

plt.bar(models_names, accuracy_values)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.grid(axis='y')

plt.savefig(
    os.path.join(static_path, "model_comparison.png")
)

plt.close()

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, best_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(f"Confusion Matrix - {best_model_name}")

plt.savefig(
    os.path.join(static_path, "confusion_matrix.png")
)

plt.close()

# ==========================================
# FUNCTION
# ==========================================
def get_results():

    return {
        "results": results,
        "best_model": best_model,
        "best_model_name": best_model_name
    }