"""
Batch preprocessing of Sentinel-1 GRD products using ESA SNAP.

Processing chain
----------------
1. Apply Orbit File
2. Remove GRD Border Noise
3. Thermal Noise Removal
4. Radiometric Calibration to Sigma0 (VV/VH)
5. Terrain Correction using SRTM 3Sec at 10 m
6. Conversion of Sigma0 bands to dB
7. GeoTIFF-BigTIFF export

The workflow is based on the supplied thesis batch-processing script.
It preserves the original SNAP operators and processing parameters,
including the ``AUTO:42001`` terrain-correction projection.

Requirements
------------
- ESA SNAP with a compatible ``esa-snappy`` Python environment
- SRTM DEM available to SNAP
- Sentinel-1 GRD products (``.zip`` or ``.SAFE``)

Notes
-----
The supplied single-scene preprocessing script and this batch script
do not implement identical processing chains: this batch workflow
includes GRD border-noise removal. That difference is intentionally
preserved here rather than silently changing the research workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the batch workflow."""
    parser = argparse.ArgumentParser(
        description="Batch-preprocess Sentinel-1 GRD products with ESA SNAP."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Sentinel-1 .zip or .SAFE products.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for processed GeoTIFF outputs and status file.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Optional JSON status file. Defaults to <output-dir>/processing_status.json.",
    )
    parser.add_argument(
        "--aoi-shapefile",
        type=Path,
        default=None,
        help="Optional AOI shapefile passed to SNAP Terrain-Correction as geoRegion.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum processing attempts per scene (default: 3).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Minimum required free disk space before processing each scene (default: 5 GB).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between failed attempts (default: 2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def free_space_gb(path: Path) -> float:
    """Return free disk space at *path* in GiB."""
    return shutil.disk_usage(path).free / (1024**3)


def load_status(status_file: Path) -> dict[str, dict[str, Any]]:
    """Load previously recorded scene-processing status."""
    if not status_file.exists():
        return {}

    try:
        with status_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid status file: {status_file}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Status file must contain a JSON object: {status_file}")

    return data


def save_status(status: dict[str, dict[str, Any]], status_file: Path) -> None:
    """Atomically save scene-processing status."""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = status_file.with_suffix(status_file.suffix + ".tmp")

    with temporary_file.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)

    temporary_file.replace(status_file)


def record_status(
    status: dict[str, dict[str, Any]],
    status_file: Path,
    scene: str,
    success: bool,
    message: str,
) -> None:
    """Record the outcome of a scene and persist the status file."""
    status[scene] = {
        "done": bool(success),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message": message,
    }
    save_status(status, status_file)


def safe_dispose(product: Any) -> None:
    """Dispose a SNAP product without masking the original exception."""
    if product is None:
        return

    try:
        product.dispose()
    except Exception:
        LOGGER.debug("Could not dispose SNAP product.", exc_info=True)


def validate_paths(
    input_dir: Path,
    output_dir: Path,
    aoi_shapefile: Path | None,
    retries: int,
    min_free_gb: float,
) -> None:
    """Validate batch-processing inputs."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if aoi_shapefile is not None and not aoi_shapefile.is_file():
        raise FileNotFoundError(f"AOI shapefile does not exist: {aoi_shapefile}")

    if retries < 1:
        raise ValueError("--retries must be at least 1.")

    if min_free_gb < 0:
        raise ValueError("--min-free-gb cannot be negative.")


def import_snap():
    """Import the ESA SNAP Python bridge only when processing is requested."""
    try:
        from esa_snappy import GPF, HashMap, ProductIO, jpy
    except ImportError as exc:
        raise RuntimeError(
            "Could not import esa_snappy. Run this script in a compatible "
            "ESA SNAP/esa-snappy environment."
        ) from exc

    return ProductIO, GPF, HashMap, jpy


def preprocess_scene(
    input_path: Path,
    output_path: Path,
    aoi_shapefile: Path | None = None,
) -> None:
    """
    Preprocess one Sentinel-1 GRD product using the supplied SNAP chain.

    The original processing parameters are retained, including:
    SRTM 3Sec, 10 m pixel spacing, AUTO:42001 projection, VV/VH
    calibration, and GeoTIFF-BigTIFF export.
    """
    ProductIO, GPF, HashMap, jpy = import_snap()

    product = orbit = border = thermal_noise = calibration = terrain_correction = None
    db_product = None

    try:
        LOGGER.debug("Reading Sentinel-1 product: %s", input_path)
        product = ProductIO.readProduct(str(input_path))

        if product is None:
            raise RuntimeError(f"SNAP could not open product: {input_path}")

        orbit = GPF.createProduct("Apply-Orbit-File", HashMap(), product)

        border = GPF.createProduct(
            "Remove-GRD-Border-Noise",
            HashMap(),
            orbit,
        )

        params = HashMap()
        params.put("removeThermalNoise", True)
        thermal_noise = GPF.createProduct(
            "ThermalNoiseRemoval",
            params,
            border,
        )

        params = HashMap()
        params.put("outputSigmaBand", True)
        params.put("selectedPolarisations", "VV,VH")
        params.put("outputImageInDecibel", False)
        calibration = GPF.createProduct(
            "Calibration",
            params,
            thermal_noise,
        )

        params = HashMap()
        params.put("demName", "SRTM 3Sec")
        params.put("pixelSpacingInMeter", 10.0)
        params.put("mapProjection", "AUTO:42001")
        params.put("saveSigmaNought", True)
        params.put("alignToStandardGrid", True)

        if aoi_shapefile is not None:
            params.put("geoRegion", str(aoi_shapefile))

        terrain_correction = GPF.createProduct(
            "Terrain-Correction",
            params,
            calibration,
        )

        band_descriptor_type = jpy.get_type(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor"
        )

        target_bands = []

        for polarisation in ("VV", "VH"):
            source_band = f"Sigma0_{polarisation}"

            if source_band not in terrain_correction.getBandNames():
                LOGGER.warning(
                    "Expected band '%s' was not found in %s.",
                    source_band,
                    input_path.name,
                )
                continue

            descriptor = band_descriptor_type()
            descriptor.name = f"{source_band}_db"
            descriptor.type = "float32"
            descriptor.expression = f"10*log10({source_band})"
            target_bands.append(descriptor)

        if not target_bands:
            raise RuntimeError(
                f"No expected Sigma0 VV/VH bands found after calibration: "
                f"{input_path.name}"
            )

        target_band_array = jpy.array(
            band_descriptor_type,
            len(target_bands),
        )

        for index, descriptor in enumerate(target_bands):
            target_band_array[index] = descriptor

        params = HashMap()
        params.put("targetBands", target_band_array)

        db_product = GPF.createProduct(
            "BandMaths",
            params,
            terrain_correction,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Writing %s", output_path)
        ProductIO.writeProduct(
            db_product,
            str(output_path),
            "GeoTIFF-BigTIFF",
        )

    finally:
        for snap_product in (
            db_product,
            terrain_correction,
            calibration,
            thermal_noise,
            border,
            orbit,
            product,
        ):
            safe_dispose(snap_product)


def find_sentinel1_scenes(input_dir: Path) -> list[Path]:
    """Return sorted Sentinel-1 .zip and .SAFE products."""
    scenes = sorted(
        path
        for path in input_dir.iterdir()
        if path.name.endswith((".zip", ".SAFE"))
    )
    return scenes


def output_name(scene: Path) -> str:
    """Return the processed GeoTIFF filename for a Sentinel-1 product."""
    return f"{scene.stem}_ARD.tif"


def process_batch(
    input_dir: Path,
    output_dir: Path,
    status_file: Path,
    aoi_shapefile: Path | None,
    retries: int,
    min_free_gb: float,
    retry_delay: float,
) -> None:
    """Process all Sentinel-1 scenes that are not already completed."""
    status = load_status(status_file)
    scenes = find_sentinel1_scenes(input_dir)

    LOGGER.info("Found %d Sentinel-1 scenes.", len(scenes))

    for index, scene in enumerate(scenes, start=1):
        scene_name = scene.name

        if status.get(scene_name, {}).get("done") is True:
            LOGGER.info(
                "[%d/%d] Skipping completed scene: %s",
                index,
                len(scenes),
                scene_name,
            )
            continue

        available_gb = free_space_gb(output_dir)
        if available_gb < min_free_gb:
            raise RuntimeError(
                f"Insufficient free disk space: {available_gb:.2f} GB "
                f"available; {min_free_gb:.2f} GB required."
            )

        output_path = output_dir / output_name(scene)
        succeeded = False

        for attempt in range(1, retries + 1):
            try:
                LOGGER.info(
                    "[%d/%d] Processing %s (attempt %d/%d)",
                    index,
                    len(scenes),
                    scene_name,
                    attempt,
                    retries,
                )

                preprocess_scene(
                    input_path=scene,
                    output_path=output_path,
                    aoi_shapefile=aoi_shapefile,
                )

                record_status(
                    status,
                    status_file,
                    scene_name,
                    True,
                    "OK",
                )
                succeeded = True
                break

            except Exception as exc:
                LOGGER.exception(
                    "Processing failed for %s on attempt %d/%d.",
                    scene_name,
                    attempt,
                    retries,
                )

                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    record_status(
                        status,
                        status_file,
                        scene_name,
                        False,
                        str(exc),
                    )

        if not succeeded:
            LOGGER.error("Scene failed after %d attempts: %s", retries, scene_name)

    LOGGER.info("Batch processing completed.")


def main() -> None:
    """Run the command-line batch-processing workflow."""
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = args.output_dir
    status_file = args.status_file or (
        output_dir / "processing_status.json"
    )

    validate_paths(
        input_dir=args.input_dir,
        output_dir=output_dir,
        aoi_shapefile=args.aoi_shapefile,
        retries=args.retries,
        min_free_gb=args.min_free_gb,
    )

    process_batch(
        input_dir=args.input_dir,
        output_dir=output_dir,
        status_file=status_file,
        aoi_shapefile=args.aoi_shapefile,
        retries=args.retries,
        min_free_gb=args.min_free_gb,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
