"""Deploy (or update) the yk-cart-forecast endpoint, which serves every cart.

Callers send only the cart and date - no model name:

    runtime.invoke_endpoint(
        EndpointName="yk-cart-forecast",
        ContentType="application/json",
        Body=json.dumps({"cart_id": "50", "date": "28-10-2026"}),
    )

The endpoint has no model baked in. inference.py loads each cart's model from
its fixed live file, s3://<bucket>/models/live/cart_<id>.tar.gz, and reloads
it whenever that file changes. Training overwrites the file when it finishes,
so a retrain is live on the next prediction with no redeploy.

The scikit-learn container imports inference.py once, when the server starts,
from SAGEMAKER_SUBMIT_DIRECTORY. This script bundles endpoint-build/code
(inference.py, app/ - needed to unpickle ForecastModel - and requirements.txt),
uploads it, and points the endpoint at it. So changing inference.py or app/
means re-running this script; retraining a model does not.

If the endpoint already exists it is updated in place (new model + endpoint
config, then UpdateEndpoint: no downtime). The previous config is kept, so a
rollback is one update-endpoint call back to it.
"""

import tarfile
import tempfile
import time
from pathlib import Path

import boto3

REGION = "ap-south-1"
BUCKET = "yatrikart-ml-prod-ai-stn"
ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)
IMAGE = "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
CODE_DIR = Path(__file__).resolve().parent / "code"

ENDPOINT_NAME = "yk-cart-forecast"
INSTANCE_TYPE = "ml.m5.xlarge"

version = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
model_name = f"{ENDPOINT_NAME}-model-{version}"
config_name = f"{ENDPOINT_NAME}-config-{version}"
code_key = f"code/{ENDPOINT_NAME}/{version}/sourcedir.tar.gz"

sm = boto3.client("sagemaker", region_name=REGION)

try:
    current = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
except sm.exceptions.ClientError as exc:
    if exc.response["Error"]["Code"] != "ValidationException":
        raise
    current = None

if current and current["EndpointStatus"] != "InService":
    raise RuntimeError(f"{ENDPOINT_NAME} is {current['EndpointStatus']}; wait until it is InService.")

with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "sourcedir.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for item in sorted(CODE_DIR.iterdir()):
            tar.add(
                item,
                arcname=item.name,
                filter=lambda t: None if "__pycache__" in t.name or ".ipynb_checkpoints" in t.name else t,
            )
    boto3.client("s3", region_name=REGION).upload_file(str(bundle), BUCKET, code_key)
print(f"Uploaded inference code: s3://{BUCKET}/{code_key}")

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=ROLE,
    PrimaryContainer={
        "Image": IMAGE,
        "Environment": {
            "AWS_REGION": REGION,
            "SAGEMAKER_REGION": REGION,
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": f"s3://{BUCKET}/{code_key}",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
    },
)
print("Created model:", model_name)

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }
    ],
)
print("Created endpoint config:", config_name)

if current:
    print(f"Updating {ENDPOINT_NAME}: {current['EndpointConfigName']} -> {config_name}")
    print(f"  rollback: aws sagemaker update-endpoint --endpoint-name {ENDPOINT_NAME} "
          f"--endpoint-config-name {current['EndpointConfigName']} --region {REGION}")
    sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
else:
    print("Creating endpoint:", ENDPOINT_NAME)
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
print("Waiting for InService (~5-10 minutes)")

sm.get_waiter("endpoint_in_service").wait(
    EndpointName=ENDPOINT_NAME,
    WaiterConfig={"Delay": 30, "MaxAttempts": 60},
)
print("Deployment completed:", ENDPOINT_NAME)
