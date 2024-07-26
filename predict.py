import boto3
import os


# Define endpoint name
endpoint_name = os.getenv('ENDPOINT_NAME')
region_name = os.getenv('AWS_REGION')

# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region_name)

# Define your request parameters
content_type = 'text/csv'  # Assuming the model expects CSV format
accept = 'application/json'  # Assuming you want a JSON response

# Define your payload (4 float features)
# Example: 0.5, 1.2, 3.4, 0.8
payload = '0.5,1.2,3.4,5.0\n'

# Make the request
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Accept=accept,
    Body=payload
)

# Print the response
print(response['Body'].read().decode('utf-8'))
