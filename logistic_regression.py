from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# =========================
# DATASET
# =========================
data = pd.read_csv("logistic_regression_dataset.csv")

X = data[["edad","ingreso","visitas","tiempo","compras","descuento"]]
y = data["target"]

# =========================
# TRAIN
# =========================
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# functions
# =========================
def predict_customer(data_input):
    return model.predict([data_input])[0]

def get_metrics():
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


# =========================
# HANDLE REQUEST
# =========================
def handle_request(form):

    edad = float(form["edad"])
    ingreso = float(form["ingreso"])
    visitas = float(form["visitas"])
    tiempo = float(form["tiempo"])
    compras = float(form["compras"])
    descuento = float(form["descuento"])

    data = [edad, ingreso, visitas, tiempo, compras, descuento]

    prediction = predict_customer(data)

    if prediction == 1:
        result = "Customer WILL BUY"
    else:
        result = "Customer WILL NOT BUY"

    accuracy, precision, recall, f1 = get_metrics()

    return result, accuracy, precision, recall, f1