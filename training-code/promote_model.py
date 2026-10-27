"""Promote a finished training job's model to the multi-model endpoint.

The multi-model endpoint serves every cart from one S3 folder
(s3://<bucket>/models/live/). Each promoted model is one archive there,
cart_<id>/<training job name>.tar.gz, holding the trained model plus the
inference code needed to load it:

    model.pkl
    cleaning_log.csv
    code/inference.py
    code/app/...            (ForecastModel's package, needed to unpickle it)
    code/requirements.txt

Archives are never overwritten: the endpoint caches models it has loaded, so a
replaced file could keep serving the old model. Instead, a pointer file
(s3://<bucket>/models/live-pointers/cart_<id>.json) names the archive that is
live, and prediction requests send that name as TargetModel. Promoting a new
job just writes a new archive and moves the pointer; no redeploy needed.

Usage:
    python promote_model.py --cart 50                       # latest completed job
    python promote_model.py --cart 50 --job cart-50-train-20261021-112632
"""

import argparse
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

BUCKET = "yatrikart-ml-prod-ai-stn"
REGION = "ap-south-1"
LIVE_PREFIX = "models/live"
POINTER_PREFIX = "models/live-pointers"
INFERENCE_CODE_DIR = Path.home() / "endpoint-build" / "code"


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


def pointer_key(cart_id):
    return f"{POINTER_PREFIX}/cart_{cart_id}.json"


def current_target_model(cart_id, s3=None):
    """TargetModel of the cart's live model, or None if it was never promoted."""
    s3 = s3 or boto3.client("s3", region_name=REGION)
    try:
        body = s3.get_object(Bucket=BUCKET, Key=pointer_key(cart_id))["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    return json.loads(body)["target_model"]


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

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trained = tmp / "trained.tar.gz"
        s3.download_file(src_bucket, src_key, str(trained))

        packaged = tmp / "packaged.tar.gz"
        with tarfile.open(packaged, "w:gz") as out:
            with tarfile.open(trained, "r:gz") as src:
                for member in src.getmembers():
                    if member.isfile():
                        out.addfile(member, src.extractfile(member))
            out.add(
                INFERENCE_CODE_DIR,
                arcname="code",
                filter=lambda t: None if "__pycache__" in t.name or ".ipynb_checkpoints" in t.name else t,
            )

        target_model = f"cart_{cart_id}/{job_name}.tar.gz"
        s3.upload_file(str(packaged), BUCKET, f"{LIVE_PREFIX}/{target_model}")

    result = {
        "cart_id": cart_id,
        "training_job_name": job_name,
        "source_artifact": artifact,
        "live_model_uri": f"s3://{BUCKET}/{LIVE_PREFIX}/{target_model}",
        "target_model": target_model,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    # Written last, so the pointer never names an archive that isn't uploaded.
    s3.put_object(
        Bucket=BUCKET,
        Key=pointer_key(cart_id),
        Body=json.dumps(result).encode("utf-8"),
        ContentType="application/json",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cart", type=int, required=True)
    parser.add_argument("--job", help="training job name (default: latest completed job for the cart)")
    args = parser.parse_args()

    result = promote(args.cart, args.job)
    print(f"PROMOTED job_name={result['training_job_name']} -> {result['live_model_uri']}")


if __name__ == "__main__":
    main()
