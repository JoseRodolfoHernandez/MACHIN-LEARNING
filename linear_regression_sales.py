from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_model():
    data = pd.read_csv("advertising.csv")

    X = data[["TV"]]  
    Y = data["Sales"]

    model = LinearRegression()
    model.fit(X, Y)

    return model, X, Y

def get_equation():
    model, _, _ = train_model()
    m = model.coef_[0]
    b = model.intercept_

    return round(m, 2), round(b, 2)


def predict_sales(investment):
    model, _, _ = train_model()
    prediction = model.predict([[investment]])[0]
    return round(prediction, 2)


def generate_graph():
    model, X, Y = train_model()

    plt.scatter(X, Y)
    plt.plot(X, model.predict(X))
    plt.grid()
    plt.title("Sales vs Investment")
    plt.xlabel("Investment")
    plt.ylabel("Sales")

    plt.savefig("static/regression.png")
    plt.close()
