import boto3
from sagemaker.sklearn.model import SKLearnModel

REGION = "ap-south-1"
BUCKET = "yatrikart-ml-prod-ai-stn"
ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)

MODEL_ARTIFACT = (
    f"s3://{BUCKET}/models/cart_50/"
    "cart-50-train-20261016-104812/output/model.tar.gz"
)

ENDPOINT_NAME = "yk-data-train"
MODEL_NAME = "yk-data-train-model"

sm = boto3.client("sagemaker", region_name=REGION)

try:
    sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    raise RuntimeError(
        f"{ENDPOINT_NAME} already exists; stopping safely."
    )
except sm.exceptions.ClientError as exc:
    if exc.response["Error"]["Code"] != "ValidationException":
        raise

model = SKLearnModel(
    model_data=MODEL_ARTIFACT,
    role=ROLE,
    entry_point="inference.py",
    source_dir="/home/sagemaker-user/endpoint-build/code",
    framework_version="1.4-2",
    py_version="py3",
    name=MODEL_NAME,
    env={"AWS_REGION": REGION},
)

print("Deploying endpoint:", ENDPOINT_NAME)
print("Using model artifact:", MODEL_ARTIFACT)

model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=ENDPOINT_NAME,
    wait=True,
)

print("Deployment completed:", ENDPOINT_NAME)
