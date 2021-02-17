from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/add", methods=["GET"])
def add():
    request_data = request.get_json()
    return jsonify({"plus" : request_data["a"] + request_data["b"]})

@app.route("/substract/<int:a>/<int:b>", methods=["GET"])
def substract(a, b):
    return jsonify({"minus" : a - b})

###################################################################
#                 Client calling the API
###################################################################
# import requests
# add_endpoint = requests.get("http://127.0.0.1:5678/add", json={"a":99,"b":60})
# add_endpoint.json()
# substract_endpoint = requests.get("http://127.0.0.1:5678/substract/60/50")
# substract_endpoint.json()
