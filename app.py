from flask import Flask, request, jsonify
from classifier import getPrediction

app= Flask(__name__)
@app.route("/predict-alphabet",methods=["POST"])
def predictdata():
    image= request.files.get("alphabet")
    prediction=getPrediction(image)
    return jsonify({
        "prediction": prediction
    }), 200
if __name__== "__main__":
    app.run( debug= True)

