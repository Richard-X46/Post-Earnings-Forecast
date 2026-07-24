# Appendix: Automated Cloud Storage Synchronization Pipeline
import os
import boto3
from botocore.exceptions import ClientError

def sync_local_cache_to_s3(local_file_path: str, bucket_name: str, s3_key: str):
    """
    Synchronizes compressed multi-modal Parquet tables from local staging 
    directly to an isolated enterprise AWS S3 bucket framework.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    try:
        print(f"Initiating multi-part upload protocol for: {local_file_path}")
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        print(f"Successfully committed target artifact to cloud space: s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        print(f"Cloud write operations failure: {e}")
        raise e