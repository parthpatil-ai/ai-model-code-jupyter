"""SageMaker training entry point for ForecastModel.

SageMaker copies the "training" input channel (an S3 folder holding
sales.csv and, optionally, catalog.csv / variant_mapping.csv / holidays.csv)
to SM_CHANNEL_TRAINING, runs this script, and uploads everything written to
SM_MODEL_DIR to S3 as model.tar.gz (model.pkl + cleaning_log.csv).

Publishing: with --live-model-uri (s3://.../models/live/cart_<id>.tar.gz), the
finished model is also uploaded there, overwriting the cart's live model; the
yk-cart-forecast endpoint picks it up on its next prediction.

Settings: any FORECAST_* setting from app/config.py can be passed as a
hyperparameter (e.g. {"FORECAST_TUNE": "0", "FORECAST_MODE": "typical"}).

Run locally the same way SageMaker does:

    SM_CHANNEL_TRAINING=sample_data/files SM_MODEL_DIR=model_out python sagemaker_train.py --FORECAST_TUNE 0
"""

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "data"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model_out"))
    parser.add_argument("--sales-file", default="sales.csv", help="file name inside --data-dir")
    parser.add_argument("--live-model-uri", help="s3:// path to overwrite with the trained model")
    args, extra = parser.parse_known_args()

    # Hyperparameters arrive as "--FORECAST_TUNE 0"; app/config.py reads them
    # from the environment, so set them before importing the app.
    for flag, value in zip(extra[::2], extra[1::2]):
        name = flag.lstrip("-")
        if not name.startswith("FORECAST_"):
            raise SystemExit(f"Unknown argument {flag}; settings must be FORECAST_* (see app/config.py)")
        os.environ[name] = str(value)
    return args


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import joblib
    import pandas as pd

    from app.forecasting.cleaning import write_cleaning_log
    from app.forecasting.model import ForecastModel

    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    model = ForecastModel()
    model.load_auxiliary_files(data_dir)
    model.train(pd.read_csv(data_dir / args.sales_file))

    joblib.dump(model, model_dir / "model.pkl")
    if model.cleaning_log is not None:
        write_cleaning_log(model.cleaning_log, model_dir / "cleaning_log.csv")

    print(
        f"Trained in {time.time() - started:.0f}s | mode={model.forecast_mode} | "
        f"data {model.history.index[0].date()}..{model.last_date.date()} | "
        f"{model.history.shape[1]} products | saved {model_dir / 'model.pkl'}"
    )

    if args.live_model_uri:
        publish(model_dir, args.live_model_uri)


def publish(model_dir: Path, uri: str) -> None:
    """Overwrite the cart's live model with this run's model.tar.gz."""
    import tarfile
    import tempfile

    import boto3

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    job = os.environ.get("TRAINING_JOB_NAME", "local")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(model_dir.iterdir()):
                tar.add(item, arcname=item.name)
        boto3.client("s3").upload_file(
            str(archive), bucket, key, ExtraArgs={"Metadata": {"training-job": job}}
        )
    print(f"Published live model -> {uri} (training job {job})")


if __name__ == "__main__":
    main()
