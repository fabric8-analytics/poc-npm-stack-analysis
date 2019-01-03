import os

bucket_name = 'avgupta-stack-analysis-dev'
access_key = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')
secret_key = os.environ.get('AWS_S3_SECRET_KEY_ID', '')
