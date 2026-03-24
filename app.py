from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    inicio = 'test'
    if 5 == 5:
        inicio = 'test2'
    return "hello Flask " + inicio 

@app.route('/FirstPage')
def firstPage():
    return render_template('index.html')

@app.route('/LinearRegression', methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = LinearRegression.calculaterGrade(hours)
    return render_template("linearRegressionGrades.html", result=calculateResult)


if __name__ == "__main__":
    app.run(debug=True)
