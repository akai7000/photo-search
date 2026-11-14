"""Upload installers to Cloudflare R2."""
import boto3
import os
from pathlib import Path

def main():
    version = os.environ.get('APP_VERSION', 'unknown')
    r2_account_id = os.environ.get('R2_ACCOUNT_ID')
    r2_bucket_name = os.environ.get('R2_BUCKET_NAME')
    r2_access_key = os.environ.get('R2_ACCESS_KEY_ID')
    r2_secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
    
    if not all([r2_account_id, r2_bucket_name, r2_access_key, r2_secret_key]):
        print("⚠️ R2 credentials not configured. Skipping upload.")
        return
    
    r2_url = f"https://{r2_account_id}.r2.cloudflarestorage.com"
    
    print(f"Uploading installers for v{version} to R2...")
    print(f"R2 URL: {r2_url}")
    print(f"Bucket: {r2_bucket_name}")
    
    # Initialize S3 client for R2
    s3 = boto3.client(
        's3',
        endpoint_url=r2_url,
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key,
        region_name='auto'
    )
    
    # Upload files
    installer_dir = Path('dist/installers')
    if installer_dir.exists():
        uploaded_count = 0
        for file in installer_dir.glob('*'):
            if file.is_file():
                r2_key = f'installers/v{version}/{file.name}'
                print(f'Uploading: {file.name} -> {r2_key}')
                try:
                    with open(file, 'rb') as f:
                        s3.upload_fileobj(f, r2_bucket_name, r2_key)
                    print(f'✅ Uploaded: {r2_key}')
                    uploaded_count += 1
                except Exception as e:
                    print(f'❌ Error uploading {file.name}: {e}')
        
        if uploaded_count > 0:
            print(f"\n✅ Successfully uploaded {uploaded_count} file(s) to R2")
        else:
            print("\n⚠️ No files were uploaded")
    else:
        print('⚠️ No installers found in dist/installers')

if __name__ == '__main__':
    main()
