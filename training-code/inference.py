import json
import os
import joblib


def model_fn(model_dir):
    """Load the trained model once when the inference container starts."""
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


def input_fn(request_body, request_content_type):
    """Convert the incoming JSON request into a Python dictionary."""
    if "application/json" not in request_content_type:
        raise ValueError(
            f"Unsupported content type: {request_content_type}"
        )

    return json.loads(request_body)


def predict_fn(input_data, model):
    """Call the existing forecasting model."""
    cart_id = str(input_data["cart_id"])
    date = input_data["date"]

    predictions = model.predict(cart_id, date)

    return {
        "cart_id": cart_id,
        "date": date,
        "predictions": predictions,
    }


def output_fn(prediction, accept):
    """Convert prediction results into JSON."""
    if "application/json" not in accept:
        raise ValueError(f"Unsupported response type: {accept}")

    return json.dumps(prediction), "application/json"
