import os
import re
import sys
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import json

import boto3
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

BUCKET = "yatrikart-ml-prod-ai-stn"
REGION = "ap-south-1"

BASE_DIR = Path.home() / "sagemaker-inference"
EXTRACTOR = BASE_DIR / "cart_data_pipeline.py"
SUBMITTER = Path.home() / "training-code" / "submit_cart_training.py"

# One multi-model endpoint serves every cart (endpoint-build/deploy_multimodel_endpoint.py).
PREDICT_ENDPOINT = os.environ.get("PREDICT_ENDPOINT", "yk-forecast-mme")

sys.path.append(str(SUBMITTER.parent))
from promote_model import current_target_model, promote  # noqa: E402

# Use the system Python for extraction (pymongo/dotenv are installed there),
# and the SageMaker virtualenv Python for training submission.
EXTRACT_PYTHON = "/opt/conda/bin/python"
TRAIN_PYTHON = sys.executable

app = FastAPI(title="Multi-Cart Training API", version="2.0.0")

jobs = {}
active_carts = set()
lock = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_extraction(cart_id: int):
    """Run cart_data_pipeline.py for one cart: Mongo -> CSVs -> validate -> S3.

    No --s3-prefix is passed: the script's own default ("raw") is correct,
    since it appends cart_{cart_id}/ itself - passing "raw/cart_{cart_id}"
    here previously produced a doubled path (raw/cart_21/cart_21/).
    """
    result = subprocess.run(
        [
            EXTRACT_PYTHON,
            str(EXTRACTOR),
            "--cart", str(cart_id),
            "--bucket", BUCKET,
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Data extraction failed. "
            + (result.stderr[-4000:] or result.stdout[-4000:])
        )


def run_data_prep(cart_id: int, job_id: str):
    try:
        with lock:
            jobs[job_id]["status"] = "extracting"
            jobs[job_id]["updated_at"] = utc_now()

        run_extraction(cart_id)

        with lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["data_s3_uri"] = f"s3://{BUCKET}/raw/cart_{cart_id}/"
            jobs[job_id]["updated_at"] = utc_now()

    except Exception as exc:
        with lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
            jobs[job_id]["updated_at"] = utc_now()
    finally:
        with lock:
            active_carts.discard(cart_id)


def training_data_ready(cart_id: int) -> bool:
    """True when /prepare-data has uploaded the cart's sales.csv to S3."""
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.head_object(Bucket=BUCKET, Key=f"raw/cart_{cart_id}/sales.csv")
        return True
    except s3.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def run_training(cart_id: int, job_id: str):
    """Submit a SageMaker training job on the data already in S3 (no extraction)."""
    try:
        with lock:
            jobs[job_id]["status"] = "submitting_training"
            jobs[job_id]["updated_at"] = utc_now()

        # Submit SageMaker job; --wait=False is handled by the submitter.
        result = subprocess.run(
            [
                TRAIN_PYTHON,
                str(SUBMITTER),
                "--cart", str(cart_id),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "SageMaker submission failed. "
                + (result.stderr[-4000:] or result.stdout[-4000:])
            )

        match = re.search(r"SUBMITTED job_name=([^\s]+)", result.stdout)
        if not match:
            raise RuntimeError(
                "Submitter did not return a training job name. "
                + result.stdout[-2000:]
            )

        job_name = match.group(1)

        with lock:
            jobs[job_id]["status"] = "training"
            jobs[job_id]["training_job_name"] = job_name
            jobs[job_id]["submission_output"] = result.stdout[-2000:]
            jobs[job_id]["updated_at"] = utc_now()

    except Exception as exc:
        with lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
            jobs[job_id]["updated_at"] = utc_now()
    finally:
        with lock:
            active_carts.discard(cart_id)


@app.get("/health")
def health():
    return {"status": "ok", "service": "multi-cart-training-api"}


@app.post("/carts/{cart_id}/prepare-data", status_code=202)
def start_data_prep(cart_id: int, background_tasks: BackgroundTasks):
    """Endpoint 1: extract Mongo data, build + validate CSVs, upload to S3. No training submitted."""
    if cart_id <= 0:
        raise HTTPException(status_code=400, detail="cart_id must be positive")

    if not EXTRACTOR.is_file():
        raise HTTPException(status_code=500, detail="Extractor script not found")

    with lock:
        if cart_id in active_carts:
            raise HTTPException(
                status_code=409,
                detail=f"Cart {cart_id} already has an active workflow",
            )

        job_id = f"cart-{cart_id}-prep-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        active_carts.add(cart_id)
        jobs[job_id] = {
            "job_id": job_id,
            "cart_id": cart_id,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "data_s3_uri": None,
            "error": None,
        }

    background_tasks.add_task(run_data_prep, cart_id, job_id)
    return {
        "message": "Data preparation workflow accepted",
        "job_id": job_id,
        "cart_id": cart_id,
        "status": "queued",
        "data_s3_prefix": f"s3://{BUCKET}/raw/cart_{cart_id}/",
    }


@app.post("/carts/{cart_id}/train", status_code=202)
def start_training(cart_id: int, background_tasks: BackgroundTasks):
    """Endpoint 2: train on the data /prepare-data already put in S3. No extraction."""
    if cart_id <= 0:
        raise HTTPException(status_code=400, detail="cart_id must be positive")

    if not SUBMITTER.is_file():
        raise HTTPException(status_code=500, detail="Training submitter not found")

    if not training_data_ready(cart_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"No training data at s3://{BUCKET}/raw/cart_{cart_id}/sales.csv; "
                f"run POST /carts/{cart_id}/prepare-data first"
            ),
        )

    with lock:
        if cart_id in active_carts:
            raise HTTPException(
                status_code=409,
                detail=f"Cart {cart_id} already has an active workflow",
            )

        job_id = f"cart-{cart_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        active_carts.add(cart_id)
        jobs[job_id] = {
            "job_id": job_id,
            "cart_id": cart_id,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "training_job_name": None,
            "error": None,
        }

    background_tasks.add_task(run_training, cart_id, job_id)
    return {
        "message": "Training workflow accepted",
        "job_id": job_id,
        "cart_id": cart_id,
        "status": "queued",
        "data_s3_prefix": f"s3://{BUCKET}/raw/cart_{cart_id}/",
        "model_s3_prefix": f"s3://{BUCKET}/models/cart_{cart_id}/",
    }


@app.get("/training-jobs/{job_id}")
def get_training_job(job_id: str):
    with lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found in this API process",
        )

    response = dict(job)

    # Once submitted, query SageMaker for the real training state.
    training_job_name = response.get("training_job_name")
    if training_job_name:
        try:
            sm = boto3.client("sagemaker", region_name=REGION)
            detail = sm.describe_training_job(
                TrainingJobName=training_job_name
            )
            response["sagemaker_status"] = detail["TrainingJobStatus"]
            response["training_job_arn"] = detail.get("TrainingJobArn")
            response["model_artifact_s3_uri"] = (
                detail.get("ModelArtifacts", {}).get("S3ModelArtifacts")
            )
            if detail.get("FailureReason"):
                response["failure_reason"] = detail["FailureReason"]
        except Exception as exc:
            response["status_lookup_error"] = str(exc)

    return response


@app.get("/carts/{cart_id}/training-jobs")
def list_cart_jobs(cart_id: int):
    with lock:
        result = [
            dict(job)
            for job in jobs.values()
            if job["cart_id"] == cart_id
        ]
    return {"cart_id": cart_id, "jobs": result}


@app.post("/carts/{cart_id}/promote")
def promote_model(cart_id: int, training_job_name: str | None = None):
    """Endpoint 3: make a completed training job's model the one /predict uses.

    Defaults to the cart's latest completed training job.
    """
    if cart_id <= 0:
        raise HTTPException(status_code=400, detail="cart_id must be positive")
    try:
        return promote(cart_id, training_job_name)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class PredictRequest(BaseModel):
    date: str  # DD-MM-YYYY


@app.post("/carts/{cart_id}/predict")
def predict(cart_id: int, request: PredictRequest):
    """Endpoint 4: forecast one day for a cart via the multi-model endpoint.

    The endpoint also saves the CSV to s3://<bucket>/prediction/daily/<MM-YYYY>/<cart>/.
    """
    if cart_id <= 0:
        raise HTTPException(status_code=400, detail="cart_id must be positive")
    try:
        datetime.strptime(request.date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be DD-MM-YYYY")

    target_model = current_target_model(cart_id)
    if target_model is None:
        raise HTTPException(
            status_code=409,
            detail=f"Cart {cart_id} has no promoted model; run POST /carts/{cart_id}/promote first",
        )

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=PREDICT_ENDPOINT,
            TargetModel=target_model,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps({"cart_id": str(cart_id), "date": request.date}),
        )
    except runtime.exceptions.ModelError as exc:
        raise HTTPException(status_code=502, detail=f"Model error: {exc}")

    result = json.loads(resp["Body"].read())
    result["target_model"] = target_model
    return result
