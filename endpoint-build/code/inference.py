import csv
import io
import json
import os
import tarfile
import threading
from io import StringIO
from datetime import datetime

import boto3
import joblib


S3_BUCKET = "yatrikart-ml-prod-ai-stn"
S3_PREFIX = "prediction/daily"
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# Each cart's live model is one fixed file, overwritten by every training run:
#   s3://<bucket>/models/live/cart_<id>.tar.gz
LIVE_MODEL_PREFIX = "models/live"

s3 = boto3.client("s3", region_name=AWS_REGION)


class LiveModels:
    """Loads each cart's model from S3 on first use and reloads it whenever
    the file in S3 changes (checked by ETag on every request), so a retrain
    takes effect on the next prediction without redeploying the endpoint."""

    def __init__(self):
        self._cache = {}  # cart_id -> (etag, training_job, model)
        self._lock = threading.Lock()

    def get(self, cart_id):
        key = f"{LIVE_MODEL_PREFIX}/cart_{cart_id}.tar.gz"
        try:
            etag = s3.head_object(Bucket=S3_BUCKET, Key=key)["ETag"]
        except s3.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise ValueError(f"No model for cart {cart_id} at s3://{S3_BUCKET}/{key}")
            raise

        with self._lock:
            cached = self._cache.get(cart_id)
            if cached and cached[0] == etag:
                return cached[1], cached[2]

            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            with tarfile.open(fileobj=io.BytesIO(obj["Body"].read()), mode="r:gz") as tar:
                model = joblib.load(tar.extractfile("model.pkl"))
            training_job = obj.get("Metadata", {}).get("training-job", "unknown")
            # Cache under the ETag of the object actually downloaded, in case
            # it was overwritten between the HEAD and the GET.
            self._cache[cart_id] = (obj["ETag"], training_job, model)
            print(f"Loaded cart {cart_id} model from s3://{S3_BUCKET}/{key} (training job {training_job})")
            return training_job, model


def model_fn(model_dir):
    """No model is baked into the endpoint; carts' models load from S3 on demand."""
    return LiveModels()


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


def predict_fn(input_data, models):
    """Generate predictions with the cart's latest model and save them to S3."""
    cart_id = str(input_data["cart_id"])
    prediction_date = input_data["date"]

    training_job, model = models.get(cart_id)
    result = {
        "cart_id": cart_id,
        "date": prediction_date,
        "model_training_job": training_job,
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
