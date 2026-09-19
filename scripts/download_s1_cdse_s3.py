#!/usr/bin/env python3
"""Download Sentinel-1 products from Copernicus Data Space via S3.

Credentials are read from CDSE_USERNAME and CDSE_PASSWORD environment variables.
No credentials are stored in this repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import boto3
import requests
from tqdm import tqdm

AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
S3_ENDPOINT = "https://eodata.cloudferro.com"
S3_CREDENTIALS_URL = "https://s3-keys-manager.cloudferro.com/api/user/credentials"
DEFAULT_TIMEOUT = 60


def get_access_token(username: str, password: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Authenticate with CDSE and return an access token."""
    response = requests.post(
        AUTH_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_product(product_name: str, token: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Return the first CDSE product whose name starts with product_name."""
    # Keep the query behavior of the thesis script while safely encoding the value.
    filter_value = quote(f"startswith(Name,'{product_name}')", safe="(),'$ ")
    url = f"{ODATA_URL}?$filter={filter_value}&$top=1"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    items = response.json().get("value", [])
    if not items:
        raise RuntimeError(f"Product not found: {product_name}")
    return items[0]


def get_s3_credentials(token: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Request temporary S3 credentials from CDSE."""
    response = requests.post(
        S3_CREDENTIALS_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def download_product(
    product_name: str,
    output_dir: Path,
    username: str,
    password: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """Download all S3 objects belonging to one CDSE product."""
    token = get_access_token(username, password, timeout=timeout)
    product = get_product(product_name, token, timeout=timeout)

    base_path = product["S3Path"].strip("/")
    parts = base_path.split("/", 1)
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected S3Path returned by CDSE: {product['S3Path']}")
    bucket_name, prefix = parts

    creds = get_s3_credentials(token, timeout=timeout)
    s3 = boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=creds["access_id"],
        aws_secret_access_key=creds["secret"],
    )

    out_dir = output_dir / product_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket = s3.Bucket(bucket_name)
    objects = list(bucket.objects.filter(Prefix=prefix))
    if not objects:
        raise RuntimeError(f"No S3 objects found for prefix: {prefix}")

    for obj in tqdm(objects, desc=f"Downloading {product_name}"):
        relative_path = obj.key[len(prefix):].lstrip("/")
        if not relative_path:
            continue
        local_path = out_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket.download_file(obj.key, str(local_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Sentinel-1 product from Copernicus Data Space via S3."
    )
    parser.add_argument("--product-name", required=True, help="Sentinel-1 product name or prefix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./sentinel1_raw"),
        help="Directory in which the product will be downloaded.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD as environment variables.")

    download_product(args.product_name, args.output_dir, username, password, args.timeout)


if __name__ == "__main__":
    main()
