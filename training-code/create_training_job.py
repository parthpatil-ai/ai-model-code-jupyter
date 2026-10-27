import sagemaker
from sagemaker.sklearn.estimator import SKLearn

session = sagemaker.Session()

role = "arn:aws:iam::773475891997:role/service-role/AmazonSageMaker-ExecutionRole-20260821T124146"

estimator = SKLearn(
    entry_point="sagemaker_train.py",
    source_dir="/home/sagemaker-user/training-code",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.4-2",
    py_version="py3",
    output_path="s3://yatrikart-ml-prod-ai-stn/models/cart_50/",
    hyperparameters={
        "FORECAST_TUNE": "0",
    },
)

estimator.fit(
    {
        "training": "s3://yatrikart-ml-prod-ai-stn/raw/cart_50/"
    }
)
