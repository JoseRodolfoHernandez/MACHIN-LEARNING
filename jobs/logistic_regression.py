import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def predict_customer(data_input):

    data = pd.read_csv("dataset_regresion_logistica.csv")

    X = data[[
        "edad",
        "ingreso_mensual",
        "visitas_web_mes",
        "tiempo_sitio_min",
        "compras_previas",
        "descuento_usado"
    ]]

    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    prediction = model.predict([data_input])

    return prediction[0]
