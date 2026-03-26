import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("dataset_regresion_logistica.csv")

print("Primeras filas del dataset:")
print(data.head())


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


accuracy = accuracy_score(y_test, y_pred)

print("\nExactitud del modelo:", accuracy)


print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

nuevo_cliente = [[30, 2000, 10, 5, 2, 1]]

prediccion = model.predict(nuevo_cliente)

if prediccion[0] == 1:
    print("\nEl cliente probablemente COMPRARÁ")
else:
    print("\nEl cliente probablemente NO comprará")
