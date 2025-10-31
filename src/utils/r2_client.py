import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "viewmax-assets")

if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    raise ValueError("Missing R2 credentials. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")


def _get_r2_client():
    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )


def download_from_r2(r2_url: str, local_path: str) -> str:
    print(f"Downloading from R2: {r2_url}")

    if '.r2.dev/' in r2_url:
        r2_key = r2_url.split('.r2.dev/')[-1]
    elif '.r2.cloudflarestorage.com/' in r2_url:
        parts = r2_url.split('.r2.cloudflarestorage.com/')[-1].split('/', 1)
        r2_key = parts[1] if len(parts) > 1 else parts[0]
    else:
        r2_key = r2_url.split('/')[-1]

    print(f"  R2 key: {r2_key}")
    print(f"  Local path: {local_path}")

    try:
        client = _get_r2_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(R2_BUCKET_NAME, r2_key, local_path)
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Downloaded successfully ({file_size:.1f} MB)")
        return local_path
    except ClientError as e:
        raise Exception(f"R2 download failed [{e.response['Error']['Code']}]: {e.response['Error']['Message']}")
    except Exception as e:
        raise Exception(f"R2 download failed: {str(e)}")


def upload_to_r2(local_path: str, r2_key: str = None, job_id: str = None) -> str:
    print(f"Uploading to R2: {local_path}")

    if r2_key is None:
        if job_id is None:
            job_id = str(uuid.uuid4())
        filename = Path(local_path).name
        r2_key = f"caption-remover/{job_id}/{filename}"

    print(f"  R2 key: {r2_key}")

    try:
        client = _get_r2_client()
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  File size: {file_size:.1f} MB")
        extra_args = {'ContentType': 'video/mp4'}
        client.upload_file(local_path, R2_BUCKET_NAME, r2_key, ExtraArgs=extra_args)
        public_url = f"https://{R2_BUCKET_NAME}.r2.dev/{r2_key}"
        print("Uploaded successfully")
        print(f"  URL: {public_url}")
        return public_url
    except ClientError as e:
        raise Exception(f"R2 upload failed [{e.response['Error']['Code']}]: {e.response['Error']['Message']}")
    except Exception as e:
        raise Exception(f"R2 upload failed: {str(e)}")


def generate_presigned_url(r2_key: str, expiration: int = 3600) -> str:
    try:
        client = _get_r2_client()
        return client.generate_presigned_url(
            'get_object',
            Params={'Bucket': R2_BUCKET_NAME, 'Key': r2_key},
            ExpiresIn=expiration
        )
    except ClientError as e:
        raise Exception(f"Failed to generate presigned URL: {str(e)}")


if __name__ == "__main__":
    print("R2 Client Configuration:")
    print(f"  Account ID: {R2_ACCOUNT_ID}")
    print(f"  Bucket: {R2_BUCKET_NAME}")
    print(f"  Access Key: {R2_ACCESS_KEY_ID[:10]}...")
    print("Ready to use. Replace credentials with actual values.")
