from flask import Flask, render_template, request, redirect
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("knn_credit.pkl")

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        checking_status = float(request.form['checking_status'])
        duration = float(request.form['duration'])
        credit_history = float(request.form['credit_history'])
        credit_amount = float(request.form['credit_amount'])
        savings_status = float(request.form['savings_status'])
        age = float(request.form['age'])

        datas = np.array((checking_status,duration,credit_history,credit_amount,savings_status,age))
        datas = np.reshape(datas, (1, -1))

        isDiabetes = model.predict(datas)

        if isDiabetes == 0:
            return render_template("index.html", data=isDiabetes)
        elif isDiabetes == 1:
            return render_template("index.html", data=isDiabetes)
    else:
        return render_template("index.html", prediction_text = "Error Clasification")

if __name__ == "__main__":
    app.run(debug=True)
