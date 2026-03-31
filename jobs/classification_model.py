import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# =========================
# RUTA BASE (IMPORTANTE)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta del dataset
data_path = os.path.join(BASE_DIR, "social_ads.csv")

# =========================
# CARGAR DATASET
# =========================
data = pd.read_csv(data_path)

print(data.head())

# =========================
# VARIABLES
# =========================
X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]

# =========================
# DIVISIÓN
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELO
# =========================
model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================
# PREDICCIONES
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# MÉTRICAS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# =========================
# CARPETA STATIC SEGURA
# =========================
static_path = os.path.join(BASE_DIR, "static")
os.makedirs(static_path, exist_ok=True)

# =========================
# MATRIZ DE CONFUSIÓN
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Random Forest Confusion Matrix")
plt.colorbar()
plt.savefig(os.path.join(static_path, "rf_confusion_matrix.png"))
plt.close()

# =========================
# CURVA ROC
# =========================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.title("Random Forest ROC Curve")
plt.savefig(os.path.join(static_path, "rf_roc_curve.png"))
plt.close()

# =========================
# FUNCIONES PARA FLASK
# =========================
def predict_customer(data_input):
    return model.predict([data_input])[0]

def get_metrics():
    return accuracy, precision, recall, f1
