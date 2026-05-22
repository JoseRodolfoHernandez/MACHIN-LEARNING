import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "social_ads.csv")

data = pd.read_csv(data_path)

# =========================
# FEATURES
# =========================

X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODELS
# =========================

def train_models():

    results = []

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "name": name,
            "accuracy": round(accuracy, 2)
        })

    generate_graph(results)

    return results

# =========================
# GRAPH
# =========================

def generate_graph(results):

    static_path = os.path.join(BASE_DIR, "static")

    os.makedirs(static_path, exist_ok=True)

    model_names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    plt.figure(figsize=(8,5))

    plt.bar(model_names, accuracies)

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")

    plt.savefig(os.path.join(static_path, "engineering_models.png"))

    plt.close()