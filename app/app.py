from flask import Flask, jsonify, request, render_template

from predictor import predict_transaction


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if not request.is_json:
        return jsonify({
            "error": "Request must use JSON."
        }), 400

    transaction = request.get_json()

    try:

        result = predict_transaction(
            transaction
        )

        return jsonify(result)

    except KeyError as error:

        return jsonify({
            "error": "Missing required field.",
            "field": str(error)
        }), 400

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        return jsonify({
            "error": "Prediction failed.",
            "details": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )