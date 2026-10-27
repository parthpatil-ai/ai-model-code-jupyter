"""Publish a past training job's model as a cart's live model (manual / rollback).

Each cart's live model is one fixed file, s3://<bucket>/models/live/cart_<id>.tar.gz.
Training overwrites it automatically when it finishes (sagemaker_train.py
--live-model-uri), so this script is only needed to put back an older model
or to publish a job that ran before that was in place. The yk-cart-forecast
endpoint notices the change on its next prediction.

Usage:
    python promote_model.py --cart 50                       # latest completed job
    python promote_model.py --cart 50 --job cart-50-train-20261021-112632
"""

import argparse
from datetime import datetime, timezone

import boto3

BUCKET = "yatrikart-ml-prod-ai-stn"
REGION = "ap-south-1"
LIVE_PREFIX = "models/live"


def latest_completed_job(sm, cart_id):
    prefix = f"cart-{cart_id}-train-"
    resp = sm.list_training_jobs(
        NameContains=prefix,
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    for job in resp["TrainingJobSummaries"]:
        if job["TrainingJobName"].startswith(prefix):
            return job["TrainingJobName"]
    raise RuntimeError(f"No completed training job found for cart {cart_id}")


def live_model_key(cart_id):
    return f"{LIVE_PREFIX}/cart_{cart_id}.tar.gz"


def promote(cart_id, job_name=None):
    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    job_name = job_name or latest_completed_job(sm, cart_id)
    if not job_name.startswith(f"cart-{cart_id}-train-"):
        raise ValueError(f"Job {job_name} does not belong to cart {cart_id}")

    detail = sm.describe_training_job(TrainingJobName=job_name)
    if detail["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"Job {job_name} is {detail['TrainingJobStatus']}, not Completed")

    artifact = detail["ModelArtifacts"]["S3ModelArtifacts"]
    src_bucket, src_key = artifact.removeprefix("s3://").split("/", 1)

    key = live_model_key(cart_id)
    s3.copy(
        {"Bucket": src_bucket, "Key": src_key},
        BUCKET,
        key,
        ExtraArgs={"Metadata": {"training-job": job_name}, "MetadataDirective": "REPLACE"},
    )
    return {
        "cart_id": cart_id,
        "training_job_name": job_name,
        "source_artifact": artifact,
        "live_model_uri": f"s3://{BUCKET}/{key}",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cart", type=int, required=True)
    parser.add_argument("--job", help="training job name (default: latest completed job for the cart)")
    args = parser.parse_args()

    result = promote(args.cart, args.job)
    print(f"PROMOTED job_name={result['training_job_name']} -> {result['live_model_uri']}")


if __name__ == "__main__":
    main()
