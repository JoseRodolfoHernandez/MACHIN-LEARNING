from flask import Flask, render_template, request
import linear_regression_sales
from jobs import logistic_regression
from jobs import classification_model
import clustering

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
@app.route("/linear_theory")
def linear_theory():
    return render_template("linear_theory.html")

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
@app.route("/logistic_theory")
def logistic_theory():
    return render_template("logistic_theory.html")

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
# CLASSIFICATION MODEL (FINAL)
# =========================
@app.route("/classification_theory")
def classification_theory():
    return render_template("classification_theory.html")

@app.route("/classification_app", methods=["GET", "POST"])
def classification_app():

    result = None
    accuracy = precision = recall = f1 = None

    if request.method == "POST":
        try:
            edad = float(request.form["edad"])
            salario = float(request.form["salario"])

            data = [edad, salario]

            prediction = classification_model.predict_customer(data)

            if prediction == 1:
                result = "Customer WILL BUY"
            else:
                result = "Customer WILL NOT BUY"

            accuracy, precision, recall, f1 = classification_model.get_metrics()

        except:
            result = "Invalid input"

    return render_template(
        "classification_app.html",
        result=result,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )

@app.route("/clustering")
def clustering():
    info = clustering.ApplyClusteringKMeans()
    return str(info["results"])




# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)

