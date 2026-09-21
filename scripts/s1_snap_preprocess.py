"""
Sentinel-1 GRD preprocessing to analysis-ready GeoTIFF using ESA SNAP.

Processing chain implemented by this supplied single-scene workflow:
1. Apply Orbit File
2. Thermal Noise Removal
3. Calibration to Sigma0 (VV/VH)
4. Terrain Correction (Range-Doppler; default DEM: SRTM 3Sec)
5. Convert Sigma0 bands to dB
6. Export GeoTIFF-BigTIFF

Example
-------
python s1_snap_preprocess.py \
    --input /path/to/S1.zip \
    --output ./out/S1_ARD.tif

Notes
-----
This script reflects the supplied single-scene thesis implementation. It does
NOT include the GRD Border Noise Removal operator used by the supplied batch
preprocessing script. That difference is intentionally preserved here rather
than silently changing the original workflow.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from snappy import GPF, HashMap, ProductIO

LOGGER = logging.getLogger(__name__)
DEFAULT_DEM = "SRTM 3Sec"
DEFAULT_PIXEL_SPACING = 10.0
DEFAULT_PROJECTION = "AUTO:42001"
DEFAULT_POLARISATIONS = "VV,VH"


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging for the command-line workflow."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def read_product(path: Path):
    """Read a Sentinel-1 product using ESA SNAP's ProductIO."""
    if not path.exists():
        raise FileNotFoundError(f"Input product not found: {path}")

    LOGGER.info("Reading product: %s", path)
    product = ProductIO.readProduct(str(path))
    if product is None:
        raise RuntimeError(f"SNAP could not read the input product: {path}")
    return product


def create_operator(name: str, parameters: HashMap, source_product):
    """Apply a SNAP operator to a source product."""
    LOGGER.debug("Running SNAP operator: %s", name)
    return GPF.createProduct(name, parameters, source_product)


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Preprocess a Sentinel-1 GRD product with ESA SNAP."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Sentinel-1 GRD input product (ZIP, SAFE, or supported SNAP product).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output GeoTIFF path (GeoTIFF-BigTIFF).",
    )
    parser.add_argument(
        "--aoi-wkt",
        default=None,
        help="Optional AOI polygon supplied as WKT for terrain-correction subsetting.",
    )
    parser.add_argument(
        "--dem",
        default=DEFAULT_DEM,
        help=f"SNAP DEM name (default: {DEFAULT_DEM!r}).",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=DEFAULT_PIXEL_SPACING,
        help=f"Terrain-correction pixel spacing in metres (default: {DEFAULT_PIXEL_SPACING}).",
    )
    parser.add_argument(
        "--map-projection",
        default=DEFAULT_PROJECTION,
        help=f"SNAP map projection string (default: {DEFAULT_PROJECTION}).",
    )
    parser.add_argument(
        "--polarisations",
        default=DEFAULT_POLARISATIONS,
        help=f"Comma-separated input polarisations (default: {DEFAULT_POLARISATIONS}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line values before starting SNAP processing."""
    if args.pixel_spacing <= 0:
        raise ValueError("--pixel-spacing must be greater than zero.")

    polarisations = [item.strip().upper() for item in args.polarisations.split(",")]
    if not polarisations or any(not item for item in polarisations):
        raise ValueError("--polarisations must contain at least one non-empty value.")


def process_product(args: argparse.Namespace) -> Path:
    """Run the supplied single-scene Sentinel-1 preprocessing chain."""
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Ensure SNAP's operator registry is available.
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    product = read_product(args.input)

    try:
        # 1) Apply Orbit File
        product_orbit = create_operator("Apply-Orbit-File", HashMap(), product)

        # 2) Thermal Noise Removal
        thermal_params = HashMap()
        thermal_params.put("removeThermalNoise", True)
        product_tnr = create_operator(
            "ThermalNoiseRemoval", thermal_params, product_orbit
        )

        # 3) Calibration to Sigma0
        calibration_params = HashMap()
        calibration_params.put("outputSigmaBand", True)
        calibration_params.put("selectedPolarisations", args.polarisations)
        calibration_params.put("outputImageInDecibel", False)
        product_cal = create_operator(
            "Calibration", calibration_params, product_tnr
        )

        # 4) Terrain Correction
        terrain_params = HashMap()
        terrain_params.put("demName", args.dem)
        terrain_params.put("demResamplingMethod", "Bilinear")
        terrain_params.put("imgResamplingMethod", "Bilinear")
        terrain_params.put("pixelSpacingInMeter", float(args.pixel_spacing))
        terrain_params.put("mapProjection", args.map_projection)
        terrain_params.put("saveSigmaNought", True)
        terrain_params.put("alignToStandardGrid", True)
        if args.aoi_wkt:
            terrain_params.put("geoRegion", args.aoi_wkt)
        product_tc = create_operator(
            "Terrain-Correction", terrain_params, product_cal
        )

        # 5) Convert Sigma0 VV/VH to dB.
        db_expression = (
            "Sigma0_VV_db = 10*log10(Sigma0_VV); "
            "Sigma0_VH_db = 10*log10(Sigma0_VH)"
        )
        bandmath_params = HashMap()
        bandmath_params.put("targetBands", db_expression)
        product_db = create_operator("BandMaths", bandmath_params, product_tc)

        # 6) Export to GeoTIFF-BigTIFF.
        LOGGER.info("Writing output: %s", args.output)
        ProductIO.writeProduct(
            product_db,
            str(args.output),
            "GeoTIFF-BigTIFF",
        )
    finally:
        # SNAP products can hold native resources. Dispose of them when possible.
        for name, snap_product in (
            ("product_db", locals().get("product_db")),
            ("product_tc", locals().get("product_tc")),
            ("product_cal", locals().get("product_cal")),
            ("product_tnr", locals().get("product_tnr")),
            ("product_orbit", locals().get("product_orbit")),
            ("product", product),
        ):
            if snap_product is not None and hasattr(snap_product, "dispose"):
                try:
                    snap_product.dispose()
                    LOGGER.debug("Disposed SNAP product: %s", name)
                except Exception:  # pragma: no cover - depends on SNAP runtime
                    LOGGER.debug("Could not dispose SNAP product: %s", name, exc_info=True)

    LOGGER.info("Completed: %s", args.output.resolve())
    return args.output.resolve()


def main() -> None:
    """Run the command-line preprocessing workflow."""
    parser = build_argparser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    try:
        validate_arguments(args)
        process_product(args)
    except Exception as exc:
        LOGGER.error("Preprocessing failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
