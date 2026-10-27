"""Deploy ONE multi-model endpoint that serves every cart's forecast model.

All models live under s3://<bucket>/models/live/ (written by
training-code/promote_model.py). A request picks the cart's model with
TargetModel, e.g.:

    runtime.invoke_endpoint(
        EndpointName="yk-forecast-mme",
        TargetModel="cart_50/cart-50-train-20261021-112632.tar.gz",
        ContentType="application/json",
        Body=json.dumps({"cart_id": "50", "date": "28-10-2026"}),
    )

Each archive carries its own inference code (code/inference.py + code/app/),
so the endpoint itself only needs the container image. The image is the same
scikit-learn 1.4-2 image the existing single-cart endpoint already runs.

This does not touch the existing yk-data-train endpoint.
"""

import boto3

REGION = "ap-south-1"
BUCKET = "yatrikart-ml-prod-ai-stn"
ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)
IMAGE = "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
MODEL_DATA_PREFIX = f"s3://{BUCKET}/models/live/"

ENDPOINT_NAME = "yk-forecast-mme"
MODEL_NAME = "yk-forecast-mme-model"
CONFIG_NAME = "yk-forecast-mme-config"
INSTANCE_TYPE = "ml.m5.xlarge"

sm = boto3.client("sagemaker", region_name=REGION)

try:
    sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    raise RuntimeError(f"{ENDPOINT_NAME} already exists; stopping safely.")
except sm.exceptions.ClientError as exc:
    if exc.response["Error"]["Code"] != "ValidationException":
        raise

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
