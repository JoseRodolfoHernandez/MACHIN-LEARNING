from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# DATASET
# =========================
data = pd.read_csv("advertising.csv")

# =========================
# TRAIN MODEL 
# =========================
def train_model():
    X = data[["TV"]]  
    Y = data["Sales"]

    model = LinearRegression()
    model.fit(X, Y)

    return model, X, Y

model, X, Y = train_model()

# =========================
# EQUATION
# =========================
def get_equation():
    m = model.coef_[0]
    b = model.intercept_
    return round(m, 2), round(b, 2)

# =========================
# PREDICTION
# =========================
def predict_sales(investment):
    prediction = model.predict(np.array([[investment]]))[0]
    return round(prediction, 2)

# =========================
# GRAPH
# =========================
def generate_graph():
    plt.scatter(X, Y)
    plt.plot(X, model.predict(X))
    plt.grid()
    plt.title("Sales vs Investment")
    plt.xlabel("Investment")
    plt.ylabel("Sales")

    plt.savefig("static/regression.png")
    plt.close()

# =========================
# HANDLE REQUEST 
# =========================
def handle_request(form):
    try:
        investment = float(form["investment"])
        prediction = predict_sales(investment)
        generate_graph()
        return prediction
    except:
        return "Invalid input"