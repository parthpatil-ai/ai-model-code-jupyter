import sagemaker
from sagemaker.sklearn.model import SKLearnModel

session = sagemaker.Session()

role = (
    "arn:aws:iam::773475891997:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20260821T124146"
)

model = SKLearnModel(
    model_data=(
        "s3://yatrikart-ml-prod-ai-stn/models/cart_50/"
        "sagemaker-scikit-learn-2026-10-08-10-32-24-052/"
        "output/model.tar.gz"
    ),
    role=role,
    entry_point="inference.py",
    source_dir="/home/sagemaker-user/sagemaker-inference",
    framework_version="1.4-2",
    py_version="py3",
    sagemaker_session=session,
    name="yk-stn-cart-50-model",
)

model.create(instance_type="ml.m5.xlarge")

print("SageMaker model created successfully.")
print("Model name:", model.name)
