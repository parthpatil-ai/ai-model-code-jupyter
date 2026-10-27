import csv
import json
import boto3
from pathlib import Path

# Existing Batch Transform output
input_file = Path.home() / "prediction-test" / "result.json"

# AWS settings
bucket = "yatrikart-ml-prod-ai-stn"
region = "ap-south-1"

# Load JSON predictions
with input_file.open("r", encoding="utf-8") as f:
    result = json.load(f)

cart_id = str(result["cart_id"])
prediction_date = result["date"]
predictions = result["predictions"]

if not predictions:
    raise ValueError("No predictions found. CSV was not uploaded.")

# Convert DD-MM-YYYY to the expected S3 folder format
from datetime import datetime

date_obj = datetime.strptime(prediction_date, "%d-%m-%Y")
month_folder = date_obj.strftime("%m-%Y")
file_date = date_obj.strftime("%d-%m-%Y")

s3_key = (
    f"prediction/daily/{month_folder}/"
    f"{cart_id}/{file_date}.csv"
)

# Create CSV locally
output_file = Path.home() / "prediction-test" / f"{file_date}.csv"

with output_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # These columns match ImportAiPredictionCsv.ts
    writer.writerow(["variantId", "productName", "quantity"])

    for item in predictions:
        writer.writerow([
            item["variant_id"],
            item["product_name"],
            item["quantity"],
        ])

print(f"CSV created: {output_file}")
print(f"Prediction rows: {len(predictions)}")
print(f"First few rows:")

with output_file.open("r", encoding="utf-8") as f:
    for _, line in zip(range(4), f):
        print(line.rstrip())

# Upload CSV to S3
s3 = boto3.client("s3", region_name=region)

s3.upload_file(
    str(output_file),
    bucket,
    s3_key,
    ExtraArgs={"ContentType": "text/csv"},
)

# Verify that the uploaded object exists
response = s3.head_object(Bucket=bucket, Key=s3_key)

print("\nUpload successful!")
print(f"S3 URI: s3://{bucket}/{s3_key}")
print(f"Uploaded size: {response['ContentLength']} bytes")
