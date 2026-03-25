import pandas as pd
import matplotlib.pyplot as pit
import io
import base64
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def show_graph(X, Y, model):
    plt.scatter(X, Y)
    plt.plot(X, model.predict(X), color='red')
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)
x= df[["Study Hours"]]
y= df[["Final Grade"]]

model = LinearRegression()
model.fit(x,y)

def calculaterGrade(hours):
    result = model.product([[hours]])(0)
    return result