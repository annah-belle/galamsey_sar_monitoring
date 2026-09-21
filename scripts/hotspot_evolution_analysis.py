"""
Hotspot Evolution Analysis for Galamsey Monitoring (2015–2025).

This script implements a spatial–temporal hotspot evolution analysis
based on annual CMF-restricted binary galamsey maps.

Hotspot categories:
    1 = Persistent hotspots
    2 = Emerging hotspots
    3 = Abandoned hotspots
    0 = No hotspot

The analysis follows the methodological definitions described in the thesis.

Author: Anabel Bonsu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio


YEARS = list(range(2015, 2026))
EXPECTED_COUNT = len(YEARS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify galamsey hotspot evolution from annual binary CMF rasters."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing annual CMF binary GeoTIFFs for 2015–2025.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where hotspot output GeoTIFFs will be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser.parse_args()


def extract_year(path: Path) -> int | None:
    """Extract a supported year from an annual raster filename."""
    for year in YEARS:
        if str(year) in path.stem:
            return year
    return None


def discover_binary_rasters(input_dir: Path) -> list[Path]:
    """Find and validate one annual binary raster for each year 2015–2025."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    candidates = sorted(input_dir.glob("*.tif"))
    year_to_file: dict[int, Path] = {}

    for path in candidates:
        year = extract_year(path)
        if year is None:
            logging.warning("Skipping TIFF without a supported year: %s", path.name)
            continue
        if year in year_to_file:
            raise ValueError(
                f"Multiple TIFF files found for {year}: "
                f"{year_to_file[year].name} and {path.name}"
            )
        year_to_file[year] = path

    missing = [year for year in YEARS if year not in year_to_file]
    if missing:
        raise ValueError(
            "Missing annual binary rasters for: "
            + ", ".join(map(str, missing))
        )

    if len(year_to_file) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} annual binary rasters (2015–2025), "
            f"found {len(year_to_file)} supported-year rasters."
        )

    return [year_to_file[year] for year in YEARS]


def read_raster_stack(binary_files: list[Path]) -> tuple[np.ndarray, dict]:
    """Read annual rasters and verify that their spatial grids match."""
    arrays: list[np.ndarray] = []
    reference_meta: dict | None = None
    reference_shape: tuple[int, int] | None = None

    for path in binary_files:
        with rasterio.open(path) as src:
            data = src.read(1)
            meta = src.meta.copy()

            if reference_shape is None:
                reference_shape = data.shape
                reference_meta = meta
            else:
                if data.shape != reference_shape:
                    raise ValueError(
                        f"Raster dimensions do not match: {path.name} has "
                        f"{data.shape}; expected {reference_shape}."
                    )
                if src.crs != reference_meta["crs"]:
                    raise ValueError(
                        f"CRS does not match reference raster: {path.name}."
                    )
                if src.transform != reference_meta["transform"]:
                    raise ValueError(
                        f"Affine transform does not match reference raster: {path.name}."
                    )

            arrays.append(data)

    if reference_meta is None:
        raise ValueError("No rasters were available to read.")

    stack = np.stack(arrays, axis=0)
    return stack, reference_meta


def classify_hotspots(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the thesis temporal definitions to the annual binary stack."""
    if stack.shape[0] != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} annual layers, got {stack.shape[0]}."
        )

    # Index mapping: 0 = 2015, 1 = 2016, ..., 10 = 2025.
    early_period = stack[0:5]       # 2015–2019
    late_period = stack[5:11]       # 2020–2025
    recent_period = stack[7:11]     # 2022–2025

    # Persistent: active in at least 8 out of 11 years.
    activity_count = np.sum(stack == 1, axis=0)
    persistent_hotspots = activity_count >= 8

    # Emerging: no activity in 2015–2019 and active in at least 2 years
    # during the 2020–2025 period, following the thesis implementation.
    early_activity = np.sum(early_period == 1, axis=0)
    late_activity = np.sum(late_period == 1, axis=0)
    emerging_hotspots = (early_activity == 0) & (late_activity >= 2)

    # Abandoned: active in at least 2 years during 2015–2019 and
    # no activity during 2022–2025.
    recent_activity = np.sum(recent_period == 1, axis=0)
    abandoned_hotspots = (early_activity >= 2) & (recent_activity == 0)

    # Priority order: Persistent (1) > Emerging (2) > Abandoned (3).
    hotspot_evolution = np.zeros(activity_count.shape, dtype=np.uint8)
    hotspot_evolution[abandoned_hotspots] = 3
    hotspot_evolution[emerging_hotspots] = 2
    hotspot_evolution[persistent_hotspots] = 1

    return {
        "persistent_hotspots.tif": persistent_hotspots,
        "emerging_hotspots.tif": emerging_hotspots,
        "abandoned_hotspots.tif": abandoned_hotspots,
        "hotspot_evolution.tif": hotspot_evolution,
    }


def write_outputs(
    outputs: dict[str, np.ndarray],
    output_dir: Path,
    reference_meta: dict,
) -> None:
    """Write hotspot classification rasters as uint8 GeoTIFFs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = reference_meta.copy()
    meta.update(dtype="uint8", count=1, nodata=0)

    for filename, data in outputs.items():
        output_path = output_dir / filename
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data.astype("uint8"), 1)
        logging.info("Written: %s", output_path)


def main() -> None:
    """Run the hotspot evolution analysis."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    binary_files = discover_binary_rasters(args.input_dir)

    logging.info("Loaded annual binary rasters:")
    for path in binary_files:
        logging.info("  %s", path.name)

    stack, reference_meta = read_raster_stack(binary_files)
    logging.info("Raster stack shape: %s", stack.shape)

    outputs = classify_hotspots(stack)
    write_outputs(outputs, args.output_dir, reference_meta)

    print("Hotspot evolution analysis completed successfully.")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
