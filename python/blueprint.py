from flask import Flask, jsonify, request, Blueprint
add_api = Blueprint('predict_api', __name__)
substract_api =  Blueprint('worker_api', __name__)

@add_api.route("/add", methods=["GET"])
def add():
    request_data = request.get_json()
    return jsonify({"plus" : request_data["a"] + request_data["b"]})

@substract_api.route("/substract/<int:a>/<int:b>", methods=["GET"])
def substract(a, b):
    return jsonify({"minus" : a - b})

app = Flask(__name__)
app.register_blueprint(add_api)
app.register_blueprint(substract_api)

###################################################################
#                 Client calling the API
###################################################################
# import requests
# add_endpoint = requests.get("http://127.0.0.1:5678/add", json={"a":99,"b":60})
# add_endpoint.json()
# substract_endpoint = requests.get("http://127.0.0.1:5678/substract/60/50")
# substract_endpoint.json()
