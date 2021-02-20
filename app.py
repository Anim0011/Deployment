import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import pickle


app = Flask(__name__, template_folder="C:/Users/Animesh/Deployment/Heroku/")
model = pickle.load(open("regmodel.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    return render_template("index.html", prediction_text="Chance $ {}".format(output))

if __name__ == "__main__":
    run_simple('localhost', 9000, app)