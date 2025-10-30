# test_r2_roundtrip.py
from src.utils.r2_client import upload_to_r2, download_from_r2
import os

# 1. Upload local file
print("1. Uploading test video...")
test_video = "data/examples/long.mp4"
uploaded_url = upload_to_r2(test_video, job_id="test-roundtrip")
print(f"   URL: {uploaded_url}")

# 2. Download it back
print("\n2. Downloading from R2...")
download_path = "/tmp/downloaded_test.mp4"
download_from_r2(uploaded_url, download_path)

# 3. Verify file exists
if os.path.exists(download_path):
    size_mb = os.path.getsize(download_path) / (1024*1024)
    print(f"   ✓ Downloaded successfully ({size_mb:.1f} MB)")
else:
    print("   ✗ Download failed")