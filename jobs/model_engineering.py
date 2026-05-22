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

data = pd.read_csv("Social_ads.csv")

# FEATURES
X = data[["Age", "EstimatedSalary"]]

# TARGET
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