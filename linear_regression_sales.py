from sklearn.linear_model import LinearRegression
import numpy as np

def predict_sales(investment):
    # Datos simulados
    X = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)
    Y = np.array([1000, 1500, 2000, 2500, 3000])

    model = LinearRegression()
    model.fit(X, Y)

    prediction = model.predict([[investment]])[0]

    return round(prediction, 2)
