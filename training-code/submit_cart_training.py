import argparse
import os
import re
import time

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

BUCKET = "yatrikart-ml-prod-ai-stn"
REGION = "ap-south-1"
ROLE = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)
CODE_DIR = os.path.expanduser("~/training-code")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cart", type=int, required=True)
    args = parser.parse_args()

    if args.cart <= 0:
        parser.error("--cart must be a positive integer")

    cart_id = args.cart
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )

    # Unique job name, allowing separate runs for the same cart.
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    job_name = f"cart-{cart_id}-train-{timestamp}"

    estimator = SKLearn(
        entry_point="sagemaker_train.py",
        source_dir=CODE_DIR,
        role=ROLE,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        framework_version="1.4-2",
        py_version="py3",
        output_path=f"s3://{BUCKET}/models/cart_{cart_id}/",
        hyperparameters={"FORECAST_TUNE": "0"},
        sagemaker_session=session,
        base_job_name=f"cart-{cart_id}-train",
    )

    estimator.fit(
        {
            "training": f"s3://{BUCKET}/raw/cart_{cart_id}/cart_{cart_id}/"
        },
        job_name=job_name,
        wait=False,
        logs=False,
    )

    print(
        f"SUBMITTED job_name={job_name} "
        f"cart_id={cart_id} "
        f"training_s3=s3://{BUCKET}/raw/cart_{cart_id}/ "
        f"output_s3=s3://{BUCKET}/models/cart_{cart_id}/"
    )


if __name__ == "__main__":
    main()
