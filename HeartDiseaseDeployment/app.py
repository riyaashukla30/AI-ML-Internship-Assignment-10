from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    result = ""

    if request.method == "POST":
        try:
            data = [float(x) for x in request.form.values()]
            data = scaler.transform([data])

            pred = model.predict(data)

            if pred[0] == 1:
                result = "⚠️ Heart Disease Detected"
            else:
                result = "✅ No Heart Disease"

        except:
            result = "⚠️ Please enter valid input"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
