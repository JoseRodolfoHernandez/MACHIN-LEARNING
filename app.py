from flask import Flask, render_template, request
import linear_regression_sales
from jobs import logistic_regression  # 👈 IMPORTANTE

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def firstPage():
    return render_template('index.html')

@app.route('/case_1_stock')
def case1Stock():
    return render_template('case_1_stock.html')

@app.route('/case2')
def case2advertising():
    return render_template('case_2_advertising.html')

@app.route('/case3')
def case3customer():
    return render_template('case_3_customer.html')

@app.route('/case4')
def case4inventory():
    return render_template('case_4_inventory.html')

@app.route('/supervised')
def supervised():
    return render_template('supervised_learning.html')

# ------------------ LINEAR REGRESSION ------------------

@app.route('/LinearRegression', methods=["GET", "POST"])
def calculateSales():
    result = None
    if request.method == "POST":
        investment = float(request.form["investment"])
        result = linear_regression_sales.predict_sales(investment)
    return render_template("linear_regression_app.html", result=result)

@app.route("/linear_theory")
def linear_theory():
    return render_template("linear_theory.html")

# ------------------ LOGISTIC REGRESSION ------------------

@app.route("/logistic", methods=["GET", "POST"])
def logistic():

    result = None

    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso"])
        visitas = float(request.form["visitas"])
        tiempo = float(request.form["tiempo"])
        compras = float(request.form["compras"])
        descuento = float(request.form["descuento"])

        data = [edad, ingreso, visitas, tiempo, compras, descuento]

        prediction = logistic_regression.predict_customer(data)

        if prediction == 1:
            result = "Customer WILL BUY"
        else:
            result = "Customer WILL NOT BUY"

    return render_template("logistic_regression.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
