#!/usr/bin/env python3
"""Upload installers to Cloudflare R2."""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import boto3


ROOT = Path(__file__).resolve().parent


def load_r2_config():
    """Load R2 configuration from .env.r2.local or environment."""
    env_file = ROOT / ".env.r2.local"
    if env_file.exists():
        load_dotenv(env_file)
    
    config = {
        "account_id": os.getenv("R2_ACCOUNT_ID"),
        "bucket": os.getenv("R2_BUCKET"),
        "access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "public_base_url": os.getenv("R2_PUBLIC_BASE_URL"),
    }
    
    if not all(config.values()):
        missing = [k for k, v in config.items() if not v]
        raise ValueError(
            f"Missing R2 configuration: {', '.join(missing)}. "
            f"Please set environment variables or create .env.r2.local"
        )
    
    return config


def upload_installers(pattern="*", dry_run=False):
    """Upload installers to R2."""
    try:
        config = load_r2_config()
    except ValueError as e:
        print(f"[ERROR] {e}")
        return False
    
    installer_dir = ROOT / "dist" / "installers"
    
    if not installer_dir.exists():
        print(f"[ERROR] No installers found in {installer_dir}")
        return False
    
    files = list(installer_dir.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching pattern '{pattern}' in {installer_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Uploading {len(files)} installer(s) to R2...")
    print(f"Bucket: {config['bucket']}")
    print(f"Endpoint: {config['endpoint_url']}")
    print(f"{'='*60}\n")
    
    if dry_run:
        for file in files:
            print(f"[DRY-RUN] Would upload: {file.name}")
        return True
    
    s3 = boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["access_key_id"],
        aws_secret_access_key=config["secret_access_key"],
        region_name="auto",
    )
    
    uploaded_count = 0
    for file in files:
        r2_key = f"installers/{file.name}"
        print(f"Uploading: {file.name}")
        
        try:
            with open(file, "rb") as f:
                s3.upload_fileobj(f, config["bucket"], r2_key)
            
            public_url = f"{config['public_base_url']}/{r2_key}"
            print(f"[OK] Uploaded to: {public_url}\n")
            uploaded_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to upload {file.name}: {e}\n")
    
    if uploaded_count > 0:
        print(f"{'='*60}")
        print(f"[OK] Successfully uploaded {uploaded_count} file(s)")
        print(f"{'='*60}")
        return True
    else:
        print(f"[ERROR] No files were uploaded")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload installers to Cloudflare R2")
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*.exe",
        help="File pattern to upload (default: *.exe)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    
    args = parser.parse_args()
    
    try:
        success = upload_installers(pattern=args.pattern, dry_run=args.dry_run)
        return 0 if success else 1
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
