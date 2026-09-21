#!/usr/bin/env python3
"""Download Sentinel-1 products from the Copernicus Data Space Ecosystem.

The workflow used in the thesis is:

1. Authenticate against the Copernicus Data Space Ecosystem (CDSE).
2. Query Sentinel-1 product metadata through the CDSE OData API.
3. Request temporary S3 credentials.
4. Download all objects belonging to the selected product.

Credentials are supplied through environment variables and are never stored in
this script.

Required environment variables
------------------------------
CDSE_USERNAME
    CDSE account username/email.
CDSE_PASSWORD
    CDSE account password.

Dependencies
------------
requests
boto3
tqdm
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
from requests import Session
from tqdm import tqdm

AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
S3_ENDPOINT = "https://eodata.cloudferro.com"
S3_CREDENTIALS_URL = (
    "https://s3-keys-manager.cloudferro.com/api/user/credentials"
)

DEFAULT_DOWNLOAD_DIR = Path("sentinel1_raw")
REQUEST_TIMEOUT = 60

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging for the command-line application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_credentials_from_environment() -> tuple[str, str]:
    """Read CDSE credentials from environment variables.

    Raises
    ------
    RuntimeError
        If either required environment variable is missing.
    """
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Set CDSE_USERNAME and CDSE_PASSWORD as environment variables."
        )

    return username, password


def get_access_token(session: Session, username: str, password: str) -> str:
    """Authenticate with CDSE and return an access token."""
    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }

    response = session.post(AUTH_URL, data=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    try:
        return response.json()["access_token"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError("CDSE authentication response did not contain an access token.") from exc


def get_product(
    session: Session,
    product_name: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    """Return the first CDSE product whose name starts with ``product_name``."""
    params = {
        "$filter": f"startswith(Name,'{product_name}')",
        "$top": 1,
    }

    response = session.get(
        ODATA_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    try:
        items = response.json().get("value", [])
    except ValueError as exc:
        raise RuntimeError("CDSE product query returned invalid JSON.") from exc

    if not items:
        raise RuntimeError(f"Product not found: {product_name}")

    return items[0]


def get_s3_credentials(
    session: Session,
    headers: dict[str, str],
) -> dict[str, str]:
    """Request temporary S3 credentials from CDSE."""
    response = session.post(
        S3_CREDENTIALS_URL,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    try:
        credentials = response.json()
        return {
            "access_id": credentials["access_id"],
            "secret": credentials["secret"],
        }
    except (KeyError, ValueError) as exc:
        raise RuntimeError("CDSE S3 credential response was invalid.") from exc


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Extract bucket and object prefix from an S3 URI."""
    parsed = urlparse(s3_path)

    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Unexpected S3 path returned by CDSE: {s3_path}")

    return parsed.netloc, parsed.path.lstrip("/")


def download_product(
    session: Session,
    product_name: str,
    output_dir: Path,
) -> Path:
    """Download all S3 objects belonging to a CDSE Sentinel-1 product."""
    username, password = get_credentials_from_environment()
    token = get_access_token(session, username, password)
    headers = {"Authorization": f"Bearer {token}"}

    product = get_product(session, product_name, headers)
    s3_path = product.get("S3Path")
    if not s3_path:
        raise RuntimeError(f"CDSE product has no S3Path: {product_name}")

    bucket_name, prefix = parse_s3_path(s3_path)
    credentials = get_s3_credentials(session, headers)

    s3 = boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=credentials["access_id"],
        aws_secret_access_key=credentials["secret"],
    )

    product_output_dir = output_dir / product_name
    product_output_dir.mkdir(parents=True, exist_ok=True)

    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=prefix)

    LOGGER.info("Downloading product: %s", product_name)
    LOGGER.info("S3 bucket: %s", bucket_name)
    LOGGER.info("Output directory: %s", product_output_dir)

    downloaded = 0
    for obj in tqdm(objects, desc="Downloading", unit="file"):
        relative_path = obj.key.removeprefix(prefix).lstrip("/")
        if not relative_path:
            continue

        local_path = product_output_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket.download_file(obj.key, str(local_path))
        downloaded += 1

    LOGGER.info("Download complete: %d files", downloaded)
    return product_output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download a Sentinel-1 product from the Copernicus Data Space Ecosystem."
    )
    parser.add_argument(
        "--product-name",
        required=True,
        help="Sentinel-1 product name or unique name prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory where the downloaded product will be stored.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    with requests.Session() as session:
        download_product(
            session=session,
            product_name=args.product_name,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
