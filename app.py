from flask import Flask, render_template, request
import linear_regression_sales
from jobs import logistic_regression  # 👈 correcto

app = Flask(__name__)

# =========================
# HOME
# =========================
@app.route('/')
@app.route('/home')
def firstPage():
    return render_template('index.html')


# =========================
# ML USE CASES
# =========================
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


# =========================
# SUPERVISED LEARNING
# =========================
@app.route('/supervised')
def supervised():
    return render_template('supervised_learning.html')


# =========================
# LINEAR REGRESSION
# =========================

# Concepts
@app.route("/linear_theory")
def linear_theory():
    return render_template("linear_theory.html")

# Application
@app.route('/LinearRegression', methods=["GET", "POST"])
def calculateSales():
    result = None

    if request.method == "POST":
        try:
            investment = float(request.form["investment"])
            result = linear_regression_sales.predict_sales(investment)
        except:
            result = "Invalid input"

    return render_template("linear_regression_app.html", result=result)


# =========================
# LOGISTIC REGRESSION
# =========================

# Concepts
@app.route("/logistic_theory")
def logistic_theory():
    return render_template("logistic_theory.html")

# Application
@app.route("/logistic", methods=["GET", "POST"])
def logistic():

    result = None
    accuracy = precision = recall = f1 = None

    if request.method == "POST":
        try:
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

            # Obtener métricas
            accuracy, precision, recall, f1 = logistic_regression.get_metrics()

        except:
            result = "Invalid input"

    return render_template(
        "logistic_regression.html",
        result=result,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )

# =========================
# CLASSIFICATION MODEL (ASSIGNED)
# =========================

# Concepts
@app.route("/classification_theory")
def classification_theory():
    return render_template("classification_theory.html")

# Application
@app.route("/classification_app")
def classification_app():
    return render_template("classification_app.html")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
