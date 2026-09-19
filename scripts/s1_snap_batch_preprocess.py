#!/usr/bin/env python3
"""Batch Sentinel-1 GRD preprocessing with ESA SNAP.

Thesis batch chain:
1. Apply Orbit File
2. Remove GRD Border Noise
3. Thermal Noise Removal
4. Calibration to Sigma0 (VV/VH)
5. Terrain Correction (SRTM 3Sec, 10 m)
6. Convert Sigma0 to dB
7. Export GeoTIFF-BigTIFF

The processing logic is based on the supplied thesis batch script, with local
paths moved to command-line arguments and operational checks added.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from esa_snappy import GPF, HashMap, ProductIO, jpy

LOGGER = logging.getLogger("s1_snap_batch")


def free_space_gb(path: Path) -> float:
    """Return free disk space in GiB for the filesystem containing path."""
    return shutil.disk_usage(path).free / (1024**3)


def load_status(path: Path) -> dict:
    """Load processing status JSON, returning an empty mapping if absent."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_status(path: Path, status: dict) -> None:
    """Atomically write processing status JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    temporary.replace(path)


def record_status(status: dict, status_path: Path, scene: str, success: bool, message: str) -> None:
    """Record the latest result for one scene."""
    status[scene] = {
        "done": bool(success),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": message,
    }
    save_status(status_path, status)


def safe_dispose(product) -> None:
    """Dispose a SNAP product without masking the original error."""
    try:
        if product:
            product.dispose()
    except Exception:
        LOGGER.debug("Could not dispose SNAP product", exc_info=True)


def preprocess_scene(input_path: Path, output_path: Path, aoi_wkt: str | None = None) -> None:
    """Process one Sentinel-1 scene using the thesis batch chain."""
    product = orbit = border = tnr = cal = tc = db = None
    try:
        product = ProductIO.readProduct(str(input_path))
        if product is None:
            raise RuntimeError(f"SNAP could not read product: {input_path}")

        orbit = GPF.createProduct("Apply-Orbit-File", HashMap(), product)
        border = GPF.createProduct("Remove-GRD-Border-Noise", HashMap(), orbit)

        params = HashMap()
        params.put("removeThermalNoise", True)
        tnr = GPF.createProduct("ThermalNoiseRemoval", params, border)

        params = HashMap()
        params.put("outputSigmaBand", True)
        params.put("selectedPolarisations", "VV,VH")
        params.put("outputImageInDecibel", False)
        cal = GPF.createProduct("Calibration", params, tnr)

        params = HashMap()
        params.put("demName", "SRTM 3Sec")
        params.put("pixelSpacingInMeter", 10.0)
        params.put("mapProjection", "AUTO:42001")
        params.put("saveSigmaNought", True)
        params.put("alignToStandardGrid", True)
        if aoi_wkt:
            params.put("geoRegion", aoi_wkt)
        tc = GPF.createProduct("Terrain-Correction", params, cal)

        band_descriptor = jpy.get_type(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor"
        )
        target_bands = []
        for polarization in ("VV", "VH"):
            name = f"Sigma0_{polarization}"
            if name in tc.getBandNames():
                descriptor = band_descriptor()
                descriptor.name = f"{name}_db"
                descriptor.type = "float32"
                descriptor.expression = f"10*log10({name})"
                target_bands.append(descriptor)

        if not target_bands:
            raise RuntimeError("No Sigma0 VV/VH bands found after terrain correction.")

        band_array = jpy.array(band_descriptor, len(target_bands))
        for index, descriptor in enumerate(target_bands):
            band_array[index] = descriptor

        params = HashMap()
        params.put("targetBands", band_array)
        db = GPF.createProduct("BandMaths", params, tc)

        ProductIO.writeProduct(db, str(output_path), "GeoTIFF-BigTIFF")
    finally:
        for item in (db, tc, cal, tnr, border, orbit, product):
            safe_dispose(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch preprocess Sentinel-1 GRD products with ESA SNAP.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing Sentinel-1 ZIP/SAFE scenes.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for processed GeoTIFFs.")
    parser.add_argument("--status-file", type=Path, default=None, help="Optional status JSON path.")
    parser.add_argument("--aoi-wkt", default=None, help="Optional AOI polygon WKT.")
    parser.add_argument("--retries", type=int, default=3, help="Maximum attempts per scene.")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="Minimum free disk space before processing a scene.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Seconds between failed attempts.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    status_path = args.status_file or (args.output_dir / "processing_status.json")
    status = load_status(status_path)
    scenes = sorted(p for p in args.input_dir.iterdir() if p.name.endswith((".zip", ".SAFE")))
    if not scenes:
        raise RuntimeError(f"No Sentinel-1 .zip or .SAFE scenes found in {args.input_dir}")

    LOGGER.info("Found %d Sentinel-1 scenes.", len(scenes))
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    for index, input_path in enumerate(scenes, start=1):
        scene = input_path.name
        if status.get(scene, {}).get("done"):
            LOGGER.info("[%d/%d] Skipping completed scene: %s", index, len(scenes), scene)
            continue

        if free_space_gb(args.output_dir) < args.min_free_gb:
            raise RuntimeError(f"Insufficient disk space in {args.output_dir}")

        output_path = args.output_dir / f"{input_path.stem}_ARD.tif"
        for attempt in range(1, args.retries + 1):
            try:
                LOGGER.info("[%d/%d] %s (attempt %d)", index, len(scenes), scene, attempt)
                preprocess_scene(input_path, output_path, args.aoi_wkt)
                record_status(status, status_path, scene, True, "OK")
                break
            except Exception as exc:
                LOGGER.exception("Failed: %s", scene)
                if attempt == args.retries:
                    record_status(status, status_path, scene, False, str(exc))
                else:
                    time.sleep(args.retry_delay)

    LOGGER.info("Batch processing completed.")


if __name__ == "__main__":
    main()
