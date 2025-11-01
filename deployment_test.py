#!/usr/bin/env python3

import boto3
import requests
import time
from pathlib import Path

# this file is using direct creds, it will be deleted and is in .gitignore
R2_ACCOUNT_ID = "0f344107b933c5a2fbf668c10c344e0d"
R2_ACCESS_KEY_ID = "466971de9adf6d56c73b6e03f3256a28"
R2_SECRET_ACCESS_KEY = "658400319f9c19f2d672ea6fcfa6954b3c571991aec6e154a6c65fb1fe63754f"
R2_BUCKET_NAME = "viewmax-assets"

MODAL_API_URL = "https://caption-remover--caption-remover-fastapi-app.modal.run" #TODO: 

INPUT_VIDEO = "long.mp4"
MODEL = "diffueraser"


def upload_to_r2(local_file: str) -> str:
    print(f"\n[1/4] Uploading {local_file} to R2...")
    r2_client = boto3.client(
        's3',
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )

    file_path = Path(local_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {local_file}")

    timestamp = int(time.time())
    s3_key = f"test-inputs/{timestamp}_{file_path.name}"

    print(f"  Uploading to: {R2_BUCKET_NAME}/{s3_key}")
    r2_client.upload_file(str(file_path), R2_BUCKET_NAME, s3_key)

    r2_url = f"https://{R2_BUCKET_NAME}.r2.cloudflarestorage.com/{s3_key}"
    print(f"Uploaded: {r2_url}")
    return r2_url


def call_modal_api(video_url: str, model: str) -> str:
    print(f"\n[2/4] Submitting job to Modal API...")
    print(f"  Endpoint: {MODAL_API_URL}")
    print(f"  Video: {video_url}")
    print(f"  Model: {model}")

    payload = {
        "video_url": video_url,
        "model": model
    }

    # Submit job
    print("\n  Submitting job...")
    start_time = time.time()
    submit_url = f"{MODAL_API_URL}/submit"
    response = requests.post(submit_url, json=payload, timeout=30)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(f"Job submission failed: {response.status_code}")

    result = response.json()
    job_id = result.get('job_id')
    
    if not job_id:
        raise Exception("No job_id in response")
    
    print(f"  Job submitted successfully!")
    print(f"  Job ID: {job_id}")
    print(f"  Status: {result.get('status')}")
    
    # Poll for completion
    print(f"\n  Polling for completion (checking every 10 seconds)...")
    poll_count = 0
    status_url = f"{MODAL_API_URL}/status/{job_id}"
    print(f"  Status URL: {status_url}")
    
    while True:
        poll_count += 1
        elapsed = time.time() - start_time
        
        # Check status
        status_response = requests.get(status_url, timeout=30)
        
        if status_response.status_code not in [200, 202]:
            print(f"\n  Warning: Status check failed (HTTP {status_response.status_code})")
            time.sleep(10)
            continue
        
        status_data = status_response.json()
        current_status = status_data.get('status')
        
        if current_status == 'completed':
            result_url = status_data.get('result_url')
            print(f"\n✓ Job completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"  Result URL: {result_url}")
            return result_url
            
        elif current_status == 'error':
            error_msg = status_data.get('error', 'Unknown error')
            raise Exception(f"Job failed: {error_msg}")
            
        elif current_status == 'processing':
            print(f"  [{poll_count}] Still processing... ({elapsed:.0f}s elapsed)")
            time.sleep(10)
            
        else:
            print(f"  Unknown status: {current_status}")
            time.sleep(10)


def download_from_r2(result_url: str, output_file: str):
    print(f"\n[3/4] Downloading result from R2...")
    r2_client = boto3.client(
        's3',
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )

    s3_key = result_url.split('.r2.cloudflarestorage.com/')[-1]

    print(f"  Downloading: {R2_BUCKET_NAME}/{s3_key}")
    print(f"  Saving to: {output_file}")

    r2_client.download_file(R2_BUCKET_NAME, s3_key, output_file)

    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"Downloaded: {output_file} ({file_size:.1f} MB)")


def main():
    print("Caption Remover - Modal Deployment Test")

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        print("\nError: Please fill in all R2 credentials at the top of this script")
        return

    try:
        video_url = upload_to_r2(INPUT_VIDEO)
        result_url = call_modal_api(video_url, MODEL)
        output_file = f"result_{MODEL}_{Path(INPUT_VIDEO).stem}.mp4"
        download_from_r2(result_url, output_file)

        print("\nTEST COMPLETED SUCCESSFULLY!")
        print(f"\nInput:  {INPUT_VIDEO}")
        print(f"Output: {output_file}")
        print(f"Model:  {MODEL}")

    except Exception as e:
        print("\nTEST FAILED")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
