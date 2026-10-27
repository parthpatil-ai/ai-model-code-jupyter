"""Deploy ONE multi-model endpoint that serves every cart's forecast model.

All models live under s3://<bucket>/models/live/ (written by
training-code/promote_model.py). A request picks the cart's model with
TargetModel, e.g.:

    runtime.invoke_endpoint(
        EndpointName="yk-cart-forecast",
        TargetModel="cart_50/cart-50-train-20261021-112632.tar.gz",
        ContentType="application/json",
        Body=json.dumps({"cart_id": "50", "date": "28-10-2026"}),
    )

The scikit-learn container imports inference.py once, when the server
starts, from SAGEMAKER_SUBMIT_DIRECTORY; it does not read code from inside
each model archive. So this script bundles endpoint-build/code (inference.py,
app/ - needed to unpickle ForecastModel - and requirements.txt), uploads it,
and points the endpoint at it. The image is the same scikit-learn 1.4-2 image
the existing single-cart endpoint already runs.

Changing inference.py or app/ therefore means redeploying the endpoint;
promoting a new model does not.

This does not touch the existing yk-data-train endpoint.
"""

import tarfile
import tempfile
from pathlib import Path

import boto3

REGION = "ap-south-1"
BUCKET = "yatrikart-ml-prod-ai-stn"
ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)
IMAGE = "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
MODEL_DATA_PREFIX = f"s3://{BUCKET}/models/live/"
CODE_DIR = Path(__file__).resolve().parent / "code"
CODE_KEY = "code/yk-cart-forecast/sourcedir.tar.gz"

ENDPOINT_NAME = "yk-cart-forecast"
MODEL_NAME = "yk-cart-forecast-model"
CONFIG_NAME = "yk-cart-forecast-config"
INSTANCE_TYPE = "ml.m5.xlarge"

sm = boto3.client("sagemaker", region_name=REGION)

try:
    sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    raise RuntimeError(f"{ENDPOINT_NAME} already exists; stopping safely.")
except sm.exceptions.ClientError as exc:
    if exc.response["Error"]["Code"] != "ValidationException":
        raise

with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "sourcedir.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for item in sorted(CODE_DIR.iterdir()):
            tar.add(
                item,
                arcname=item.name,
                filter=lambda t: None if "__pycache__" in t.name or ".ipynb_checkpoints" in t.name else t,
            )
    boto3.client("s3", region_name=REGION).upload_file(str(bundle), BUCKET, CODE_KEY)
print(f"Uploaded inference code: s3://{BUCKET}/{CODE_KEY}")

sm.create_model(
    ModelName=MODEL_NAME,
    ExecutionRoleArn=ROLE,
    PrimaryContainer={
        "Image": IMAGE,
        "Mode": "MultiModel",
        "ModelDataUrl": MODEL_DATA_PREFIX,
        "Environment": {
            "AWS_REGION": REGION,
            "SAGEMAKER_REGION": REGION,
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": f"s3://{BUCKET}/{CODE_KEY}",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
    },
)
print("Created model:", MODEL_NAME)

sm.create_endpoint_config(
    EndpointConfigName=CONFIG_NAME,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }
    ],
)
print("Created endpoint config:", CONFIG_NAME)

sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=CONFIG_NAME)
print("Deploying endpoint:", ENDPOINT_NAME, "(takes ~5-10 minutes)")

sm.get_waiter("endpoint_in_service").wait(
    EndpointName=ENDPOINT_NAME,
    WaiterConfig={"Delay": 30, "MaxAttempts": 60},
)
print("Deployment completed:", ENDPOINT_NAME)
