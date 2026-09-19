#!/usr/bin/env python3
"""Classify annual binary galamsey maps into hotspot-evolution categories.

The supplied thesis script defines:
- Persistent: active in at least 8 of 11 years (2015–2025).
- Emerging: no activity in 2015–2019 and active in at least 2 years in 2020–2025.
- Abandoned: active in at least 2 years in 2015–2019 and no activity in 2022–2025.

Combined class codes remain:
0 = no hotspot
1 = persistent
2 = emerging
3 = abandoned
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import rasterio

LOGGER = logging.getLogger("hotspot_evolution")
REQUIRED_YEARS = tuple(range(2015, 2026))
YEAR_PATTERN = re.compile(r"(?<!\d)(201[5-9]|202[0-5])(?!\d)")


def year_from_filename(path: Path) -> int:
    """Extract one 2015–2025 year from a raster filename."""
    matches = [int(value) for value in YEAR_PATTERN.findall(path.name)]
    if len(matches) != 1:
        raise ValueError(f"Could not identify exactly one year in filename: {path.name}")
    return matches[0]


def discover_rasters(input_dir: Path) -> list[Path]:
    """Find one GeoTIFF for each required year and return them chronologically."""
    candidates = list(input_dir.glob("*.tif"))
    by_year: dict[int, Path] = {}
    for path in candidates:
        year = year_from_filename(path)
        if year in by_year:
            raise ValueError(f"Multiple rasters found for {year}: {by_year[year].name}, {path.name}")
        by_year[year] = path

    missing = sorted(set(REQUIRED_YEARS) - set(by_year))
    extra = sorted(set(by_year) - set(REQUIRED_YEARS))
    if missing or extra:
        raise ValueError(f"Expected exactly 2015–2025. Missing={missing}; extra={extra}")
    return [by_year[year] for year in REQUIRED_YEARS]


def read_aligned_stack(paths: list[Path]) -> tuple[np.ndarray, dict]:
    """Read rasters after verifying dimensions and spatial alignment."""
    arrays = []
    reference_meta = None
    reference_shape = None

    for path in paths:
        with rasterio.open(path) as src:
            if reference_meta is None:
                reference_meta = src.meta.copy()
                reference_shape = src.shape
            else:
                if src.shape != reference_shape:
                    raise ValueError(f"Raster shape mismatch: {path.name}")
                if src.crs != reference_meta["crs"]:
                    raise ValueError(f"CRS mismatch: {path.name}")
                if src.transform != reference_meta["transform"]:
                    raise ValueError(f"Transform mismatch: {path.name}")
            arrays.append(src.read(1))

    return np.stack(arrays, axis=0), reference_meta


def classify_hotspots(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the thesis hotspot definitions to an annual binary stack."""
    if stack.shape[0] != 11:
        raise ValueError("Hotspot classification requires 11 annual rasters (2015–2025).")

    early_period = stack[0:5]       # 2015–2019
    late_period = stack[5:11]       # 2020–2025
    recent_period = stack[7:11]     # 2022–2025

    activity_count = np.sum(stack == 1, axis=0)
    early_activity = np.sum(early_period == 1, axis=0)
    late_activity = np.sum(late_period == 1, axis=0)
    recent_activity = np.sum(recent_period == 1, axis=0)

    persistent = activity_count >= 8
    emerging = (early_activity == 0) & (late_activity >= 2)
    abandoned = (early_activity >= 2) & (recent_activity == 0)

    # Preserve the original priority: Persistent > Emerging > Abandoned.
    evolution = np.zeros(activity_count.shape, dtype=np.uint8)
    evolution[abandoned] = 3
    evolution[emerging] = 2
    evolution[persistent] = 1

    return {
        "persistent_hotspots.tif": persistent,
        "emerging_hotspots.tif": emerging,
        "abandoned_hotspots.tif": abandoned,
        "hotspot_evolution.tif": evolution,
    }


def write_outputs(outputs: dict[str, np.ndarray], output_dir: Path, metadata: dict) -> None:
    """Write hotspot rasters using the first input raster's spatial metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata.copy()
    metadata.update(dtype="uint8", count=1, nodata=0)

    for filename, data in outputs.items():
        output_path = output_dir / filename
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(data.astype("uint8"), 1)
        LOGGER.info("Written: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse galamsey hotspot evolution from annual binary rasters.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing 2015–2025 annual binary GeoTIFFs.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for hotspot classification rasters.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    raster_paths = discover_rasters(args.input_dir)
    LOGGER.info("Annual rasters: %s", ", ".join(path.name for path in raster_paths))
    stack, metadata = read_aligned_stack(raster_paths)
    LOGGER.info("Raster stack shape: %s", stack.shape)

    outputs = classify_hotspots(stack)
    write_outputs(outputs, args.output_dir, metadata)
    LOGGER.info("Hotspot evolution analysis completed successfully.")


if __name__ == "__main__":
    main()
