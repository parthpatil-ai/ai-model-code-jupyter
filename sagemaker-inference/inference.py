import csv
import json
import os
from io import StringIO
from datetime import datetime

import boto3
import joblib


S3_BUCKET = "yatrikart-ml-prod-ai-stn"
S3_PREFIX = "prediction/daily"
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

s3 = boto3.client("s3", region_name=AWS_REGION)


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


def save_predictions_to_s3(result):
    """Convert prediction results to CSV and upload to S3."""
    cart_id = str(result["cart_id"])
    prediction_date = result["date"]

    # Validate the requested date and build the destination key.
    date_obj = datetime.strptime(prediction_date, "%d-%m-%Y")
    month_folder = date_obj.strftime("%m-%Y")
    filename = date_obj.strftime("%d-%m-%Y") + ".csv"

    predictions = result["predictions"]

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list.")

    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)

    writer.writerow(["variantId", "productName", "quantity"])

    for item in predictions:
        writer.writerow([
            item.get("variant_id", item.get("variantId", "")),
            item.get("product_name", item.get("productName", "")),
            item.get("quantity", 0),
        ])

    s3_key = (
        f"{S3_PREFIX}/{month_folder}/{cart_id}/{filename}"
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    print(f"Prediction CSV uploaded to s3://{S3_BUCKET}/{s3_key}")


def predict_fn(input_data, model):
    """Generate predictions and save them to S3."""
    cart_id = str(input_data["cart_id"])
    prediction_date = input_data["date"]

    result = {
        "cart_id": cart_id,
        "date": prediction_date,
        "predictions": model.predict(cart_id, prediction_date),
    }

    # Upload the CSV before returning the API response.
    # If upload fails, the request will fail rather than silently
    # reporting success without saving the file.
    save_predictions_to_s3(result)

    return result


def output_fn(prediction, accept):
    """Return the existing JSON response."""
    if "application/json" not in accept:
        raise ValueError(f"Unsupported response type: {accept}")

    return json.dumps(prediction), "application/json"
