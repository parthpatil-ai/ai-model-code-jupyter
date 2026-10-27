import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

BUCKET = "yatrikart-ml-prod-ai-stn"
REGION = "ap-south-1"

ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)

MODEL_ARTIFACT = (
    f"s3://{BUCKET}/models/cart_50/"
    "sagemaker-scikit-learn-2026-10-08-10-32-24-052/"
    "output/model.tar.gz"
)

SOURCE_DIR = "/home/sagemaker-user/sagemaker-inference"
ENDPOINT_NAME = "yk-stn-cart-50-endpoint"

session = sagemaker.Session()
sm = boto3.client("sagemaker", region_name=REGION)

# Check whether this endpoint name is already in use.
try:
    existing = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print("Endpoint already exists:", existing["EndpointStatus"])
    print("Endpoint name:", ENDPOINT_NAME)
    raise SystemExit(
        "Stopping to avoid accidentally creating or changing an existing endpoint."
    )
except sm.exceptions.ClientError as exc:
    code = exc.response["Error"]["Code"]
    if code not in ("ValidationException", "ResourceNotFound"):
        raise

# Register the existing trained artifact and inference code.
model = SKLearnModel(
    model_data=MODEL_ARTIFACT,
    role=ROLE,
    entry_point="inference.py",
    source_dir=SOURCE_DIR,
    framework_version="1.4-2",
    py_version="py3",
    sagemaker_session=session,
    name="yk-stn-cart-50-realtime-model",
)

print("Creating real-time endpoint. This may take several minutes...")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=ENDPOINT_NAME,
    wait=True,
)

print("\nEndpoint deployment completed.")
print("Endpoint name:", ENDPOINT_NAME)
print(
    "Endpoint ARN:",
    sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointArn"],
)
