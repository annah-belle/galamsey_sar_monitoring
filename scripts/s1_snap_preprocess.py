#!/usr/bin/env python3
"""Preprocess one Sentinel-1 GRD product with ESA SNAP.

Original thesis single-scene chain preserved:
1. Apply Orbit File
2. Thermal Noise Removal
3. Calibration to Sigma0 (VV/VH)
4. Terrain Correction (SRTM 3Sec, 10 m)
5. Convert Sigma0 to dB
6. Export GeoTIFF-BigTIFF

Note: the thesis batch script additionally applies Remove-GRD-Border-Noise.
This single-scene script intentionally preserves the original single-scene chain;
see docs/methodology.md for the discrepancy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from snappy import GPF, HashMap, ProductIO


def read_product(path: Path):
    """Read a SNAP product from disk."""
    product = ProductIO.readProduct(str(path))
    if product is None:
        raise RuntimeError(f"SNAP could not read product: {path}")
    return product


def create_product(operator: str, parameters: HashMap, source_product):
    """Run a SNAP graph operator."""
    return GPF.createProduct(operator, parameters, source_product)


def preprocess_scene(
    input_path: Path,
    output_path: Path,
    aoi_wkt: str | None = None,
    dem: str = "SRTM 3Sec",
    pixel_spacing: float = 10.0,
    map_projection: str = "AUTO:42001",
    polarisations: str = "VV,VH",
) -> None:
    """Run the original single-scene preprocessing chain."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    product = orbit = tnr = cal = tc = db = None
    try:
        product = read_product(input_path)

        orbit = create_product("Apply-Orbit-File", HashMap(), product)

        params = HashMap()
        params.put("removeThermalNoise", True)
        tnr = create_product("ThermalNoiseRemoval", params, orbit)

        params = HashMap()
        params.put("outputSigmaBand", True)
        params.put("selectedPolarisations", polarisations)
        params.put("outputImageInDecibel", False)
        cal = create_product("Calibration", params, tnr)

        params = HashMap()
        params.put("demName", dem)
        params.put("demResamplingMethod", "Bilinear")
        params.put("imgResamplingMethod", "Bilinear")
        params.put("pixelSpacingInMeter", float(pixel_spacing))
        params.put("mapProjection", map_projection)
        params.put("saveSigmaNought", True)
        params.put("alignToStandardGrid", True)
        if aoi_wkt:
            params.put("geoRegion", aoi_wkt)
        tc = create_product("Terrain-Correction", params, cal)

        expression = "Sigma0_VV_db = 10*log10(Sigma0_VV); Sigma0_VH_db = 10*log10(Sigma0_VH)"
        params = HashMap()
        params.put("targetBands", expression)
        db = create_product("BandMaths", params, tc)

        ProductIO.writeProduct(db, str(output_path), "GeoTIFF-BigTIFF")
    finally:
        for item in (db, tc, cal, tnr, orbit, product):
            try:
                if item:
                    item.dispose()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess one Sentinel-1 GRD product with ESA SNAP.")
    parser.add_argument("--input", required=True, type=Path, help="Sentinel-1 GRD input (ZIP/SAFE/.dim).")
    parser.add_argument("--output", required=True, type=Path, help="Output GeoTIFF-BigTIFF path.")
    parser.add_argument("--aoi-wkt", default=None, help="Optional AOI polygon WKT for subsetting.")
    parser.add_argument("--dem", default="SRTM 3Sec", help="SNAP DEM name.")
    parser.add_argument("--pixel-spacing", type=float, default=10.0, help="Terrain-correction pixel spacing in metres.")
    parser.add_argument("--map-projection", default="AUTO:42001", help="SNAP map projection string.")
    parser.add_argument("--polarisations", default="VV,VH", help="Comma-separated polarisations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_scene(
        args.input,
        args.output,
        aoi_wkt=args.aoi_wkt,
        dem=args.dem,
        pixel_spacing=args.pixel_spacing,
        map_projection=args.map_projection,
        polarisations=args.polarisations,
    )
    print(f"Done -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
