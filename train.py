# train.py
import boto3
import sagemaker
from sagemaker.estimator import Estimator
import os


# Retrieve environment variables
role = os.getenv('SAGEMAKER_ROLE')
bucket_name = os.getenv('S3_BUCKET')
train_s3_path = os.getenv('TRAIN_S3_PATH')
validation_s3_path = os.getenv('VALIDATION_S3_PATH')
region_name = os.getenv('AWS_REGION')
endpoint_name = os.getenv('ENDPOINT_NAME')

# Define the container for the algorithm (e.g., XGBoost)
container = sagemaker.image_uris.retrieve('xgboost', region_name, version='1.2-1')

# Create the Estimator
xgb_estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket_name}/output',
    sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region_name))
)

# Set hyperparameters
xgb_estimator.set_hyperparameters(
    objective='multi:softmax',
    num_class=3,  # Number of classes in the Iris dataset
    num_round=5
)

# Specify input data sources
train_input = sagemaker.inputs.TrainingInput(train_s3_path, content_type='csv')
validation_input = sagemaker.inputs.TrainingInput(validation_s3_path, content_type='csv')

# Start training
xgb_estimator.fit({'train': train_input, 'validation': validation_input})

# Deploy the model
model = xgb_estimator.create_model()

# Create a predictor and deploy to an endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name
)