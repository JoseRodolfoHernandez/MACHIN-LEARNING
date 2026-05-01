from flask import Flask, render_template, request
import linear_regression_sales
from jobs import logistic_regression
from jobs import classification_model
from jobs import clustering

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
        result = linear_regression_sales.handle_request(request.form)

    return render_template("linear_regression_app.html", result=result)


# =========================
# LOGISTIC REGRESSION
# =========================
@app.route("/logistic_theory")
def logistic_theory():
    return render_template("logistic_theory.html")

@app.route("/logistic", methods=["GET", "POST"])
def logistic():

    result = accuracy = precision = recall = f1 = None

    if request.method == "POST":
        result, accuracy, precision, recall, f1 = logistic_regression.handle_request(request.form)

    return render_template(
        "logistic_regression.html",
        result=result,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )


# =========================
# CLASSIFICATION MODEL
# =========================
@app.route("/classification_theory")
def classification_theory():
    return render_template("classification_theory.html")

@app.route("/classification_app", methods=["GET", "POST"])
def classification_app():

    result = accuracy = precision = recall = f1 = None

    if request.method == "POST":
        result, accuracy, precision, recall, f1 = classification_model.handle_request(request.form)

    return render_template(
        "classification_app.html",
        result=result,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )


# =========================
# UNSUPERVISED LEARNING
# =========================
@app.route("/clustering_theory")
def clustering_theory():
    return render_template("clustering_theory.html")

@app.route("/clustering")
def clustering_view():
    data, summary = clustering.apply_kmeans()
    return render_template("clustering.html", data=data, summary=summary)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)