import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
BUCKET = "yatrikart-ml-prod-ai-stn"
REGION = "ap-south-1"

ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)

MODEL_ARTIFACT = (
    "s3://yatrikart-ml-prod-ai-stn/models/cart_50/"
    "sagemaker-scikit-learn-2026-10-08-10-32-24-052/"
    "output/model.tar.gz"
)

SOURCE_DIR = "/home/sagemaker-user/sagemaker-inference"

# --------------------------------------------------
# 2. Get store and prediction date from command line
# --------------------------------------------------
if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python run_batch_transform.py <cart_id> <DD-MM-YYYY>\n"
        "Example: python run_batch_transform.py 50 12-10-2026"
    )

cart_id = str(sys.argv[1])
prediction_date = sys.argv[2]

datetime.strptime(prediction_date, "%d-%m-%Y")

date_obj = datetime.strptime(prediction_date, "%d-%m-%Y")
month_folder = date_obj.strftime("%m-%Y")
file_date = date_obj.strftime("%d-%m-%Y")

# Temporary Batch Transform input/output locations
input_key = (
    f"predictions/batch-transform-input/cart_{cart_id}/"
    f"{file_date}/request.json"
)

output_prefix = (
    f"s3://{BUCKET}/predictions/batch-transform-output/"
    f"cart_{cart_id}/{file_date}/"
)

output_key = (
    f"predictions/batch-transform-output/"
    f"cart_{cart_id}/{file_date}/request.json.out"
)

# Final location expected by the Node.js importer
csv_key = (
    f"prediction/daily/{month_folder}/"
    f"{cart_id}/{file_date}.csv"
)

work_dir = Path.home() / "prediction-test"
work_dir.mkdir(parents=True, exist_ok=True)

request_file = work_dir / "request.json"
csv_file = work_dir / f"{file_date}.csv"

# --------------------------------------------------
# 3. Create and upload the prediction request
# --------------------------------------------------
request = {
    "cart_id": cart_id,
    "date": prediction_date,
}

request_file.write_text(
    json.dumps(request),
    encoding="utf-8",
)

s3 = boto3.client("s3", region_name=REGION)

s3.upload_file(
    str(request_file),
    BUCKET,
    input_key,
)

input_uri = f"s3://{BUCKET}/{input_key}"

print(f"Prediction request uploaded: {input_uri}")

# --------------------------------------------------
# 4. Run SageMaker Batch Transform
# --------------------------------------------------
session = sagemaker.Session()

model = SKLearnModel(
    model_data=MODEL_ARTIFACT,
    role=ROLE,
    entry_point="inference.py",
    source_dir=SOURCE_DIR,
    framework_version="1.4-2",
    py_version="py3",
    sagemaker_session=session,
    name="yk-stn-cart-50-model",
)

transformer = model.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=output_prefix,
    strategy="SingleRecord",
    accept="application/json",
)

print("Starting SageMaker Batch Transform...")

transformer.transform(
    data=input_uri,
    content_type="application/json",
    wait=True,
)

print("Batch Transform completed.")

# --------------------------------------------------
# 5. Read the JSON prediction output from S3
# --------------------------------------------------
response = s3.get_object(
    Bucket=BUCKET,
    Key=output_key,
)

result = json.loads(
    response["Body"].read().decode("utf-8")
)

if str(result.get("cart_id")) != cart_id:
    raise ValueError("Prediction output cart_id does not match request.")

if result.get("date") != prediction_date:
    raise ValueError("Prediction output date does not match request.")

predictions = result.get("predictions")

if not isinstance(predictions, list) or not predictions:
    raise ValueError("No predictions returned. CSV was not uploaded.")

# --------------------------------------------------
# 6. Convert JSON predictions into CSV
# --------------------------------------------------
with csv_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    # Matches the existing Node.js CSV importer
    writer.writerow(["variantId", "productName", "quantity"])

    for item in predictions:
        for field in ("variant_id", "product_name", "quantity"):
            if field not in item:
                raise ValueError(
                    f"Prediction item is missing required field: {field}"
                )

        writer.writerow([
            item["variant_id"],
            item["product_name"],
            item["quantity"],
        ])

print(f"CSV created: {csv_file}")
print(f"Prediction rows: {len(predictions)}")

# --------------------------------------------------
# 7. Upload CSV to the final S3 location
# --------------------------------------------------
s3.upload_file(
    str(csv_file),
    BUCKET,
    csv_key,
    ExtraArgs={"ContentType": "text/csv"},
)

# Verify the uploaded file
uploaded = s3.head_object(
    Bucket=BUCKET,
    Key=csv_key,
)

print("\nSUCCESS: Prediction and CSV upload completed.")
print(f"S3 URI: s3://{BUCKET}/{csv_key}")
print(f"CSV size: {uploaded['ContentLength']} bytes")
