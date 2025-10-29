"""
Cloudflare R2 Storage Client
Handles upload/download operations for video processing pipeline.
"""

import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# R2 CREDENTIALS - Load from environment variables
# ============================================================================
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "viewmax-assets")  # Default to viewmax-assets

# Validate credentials on import
if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    raise ValueError(
        "Missing R2 credentials. Set environment variables: "
        "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
    )

# ============================================================================
# R2 Client Functions
# ============================================================================

def _get_r2_client():
    """Initialize and return boto3 S3 client configured for R2."""
    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'  # R2 uses 'auto' region
    )
    return client


def download_from_r2(r2_url: str, local_path: str) -> str:
    """
    Download file from R2 to local path.
    
    Args:
        r2_url: Full R2 URL (e.g., https://bucket.r2.dev/path/to/file.mp4)
        local_path: Local file path to save to
    
    Returns:
        Local path to downloaded file
    
    Raises:
        Exception: If download fails
    """
    print(f"Downloading from R2: {r2_url}")
    
    # Extract R2 key from URL
    # Format: https://bucket.r2.dev/path/to/file.mp4 -> path/to/file.mp4
    # Or: https://account.r2.cloudflarestorage.com/bucket/path/to/file.mp4 -> path/to/file.mp4
    if '.r2.dev/' in r2_url:
        r2_key = r2_url.split('.r2.dev/')[-1]
    elif '.r2.cloudflarestorage.com/' in r2_url:
        # Format: https://account.r2.cloudflarestorage.com/bucket/key
        parts = r2_url.split('.r2.cloudflarestorage.com/')[-1].split('/', 1)
        if len(parts) > 1:
            r2_key = parts[1]
        else:
            r2_key = parts[0]
    else:
        # Assume it's just the key
        r2_key = r2_url.split('/')[-1]
    
    print(f"  R2 key: {r2_key}")
    print(f"  Local path: {local_path}")
    
    try:
        client = _get_r2_client()
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        client.download_file(R2_BUCKET_NAME, r2_key, local_path)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"✓ Downloaded successfully ({file_size:.1f} MB)")
        
        return local_path
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        raise Exception(f"R2 download failed [{error_code}]: {error_msg}")
    except Exception as e:
        raise Exception(f"R2 download failed: {str(e)}")


def upload_to_r2(local_path: str, r2_key: str = None, job_id: str = None) -> str:
    """
    Upload file to R2 and return public URL.
    
    Args:
        local_path: Local file path to upload
        r2_key: Optional R2 key (path in bucket). If None, auto-generates.
        job_id: Optional job ID for organizing files. If None, generates UUID.
    
    Returns:
        Public R2 URL to the uploaded file
    
    Raises:
        Exception: If upload fails
    """
    print(f"Uploading to R2: {local_path}")
    
    # Auto-generate R2 key if not provided
    # Modal storage structure: caption-remover/job-id/filename.mp4
    if r2_key is None:
        if job_id is None:
            job_id = str(uuid.uuid4())
        filename = Path(local_path).name
        r2_key = f"caption-remover/{job_id}/{filename}"
    
    print(f"  R2 key: {r2_key}")
    
    try:
        client = _get_r2_client()
        
        # Get file size for logging
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.1f} MB")
        
        # Upload with proper content type
        extra_args = {'ContentType': 'video/mp4'}
        client.upload_file(
            local_path,
            R2_BUCKET_NAME,
            r2_key,
            ExtraArgs=extra_args
        )
        
        # Generate public URL
        # Format: https://<bucket-name>.r2.dev/<key>
        public_url = f"https://{R2_BUCKET_NAME}.r2.dev/{r2_key}"
        
        print(f"✓ Uploaded successfully")
        print(f"  URL: {public_url}")
        
        return public_url
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        raise Exception(f"R2 upload failed [{error_code}]: {error_msg}")
    except Exception as e:
        raise Exception(f"R2 upload failed: {str(e)}")


def generate_presigned_url(r2_key: str, expiration: int = 3600) -> str:
    """
    Generate temporary signed URL for private buckets.
    
    Args:
        r2_key: R2 key (path in bucket)
        expiration: URL expiration time in seconds (default 1 hour)
    
    Returns:
        Presigned URL
    """
    try:
        client = _get_r2_client()
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': R2_BUCKET_NAME, 'Key': r2_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        raise Exception(f"Failed to generate presigned URL: {str(e)}")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("R2 Client Configuration:")
    print(f"  Account ID: {R2_ACCOUNT_ID}")
    print(f"  Bucket: {R2_BUCKET_NAME}")
    print(f"  Access Key: {R2_ACCESS_KEY_ID[:10]}...")
    print("\nReady to use. Replace credentials at top of file with actual values.")
