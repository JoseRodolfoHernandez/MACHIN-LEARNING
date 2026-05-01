import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

data = pd.read_csv("logistic_regression_dataset.csv")

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

X = data[[
    "edad",
    "ingreso_mensual",
    "visitas_web_mes",
    "tiempo_sitio_min",
    "compras_previas",
    "descuento_usado"
]]

y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("static/confusion_matrix.png")
plt.close()


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.savefig("static/roc_curve.png")
plt.close()



def predict_customer(data_input):
    prediction = model.predict([data_input])
    return prediction[0]


def get_metrics():
    return accuracy, precision, recall, f1
