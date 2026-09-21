# Data

Large and source datasets are not stored in this repository.

## Expected local inputs

The workflows expect the following local data:

- **Sentinel-1 GRD products** downloaded from the Copernicus Data Space Ecosystem.
- **Sentinel-1-derived sample CSV** containing `plot_id`, `Class`, `VV`, and `VH` (or `BAND_1`, `BAND_2`).
- **Annual CMF-restricted binary GeoTIFFs** for 2015–2025 for hotspot evolution analysis.

## Reference sample fields

| Field | Description |
|---|---|
| `plot_id` | Reference plot identifier used for spatial grouping and validation. |
| `Class` | Target class used by the modelling workflow: galamsey or non-galamsey. |
| `VV` | Sentinel-1 VV backscatter value. |
| `VH` | Sentinel-1 VH backscatter value. |
| `BAND_1` / `BAND_2` | Alternative band names accepted by the training script and mapped to the VV/VH inputs. |

The modelling workflow derives the following predictors from VV and VH:

- `VV_VH_diff` — VV minus VH
- `VV_VH_ratio` — linear-power VV/VH ratio
- `NPI` — Normalized Polarisation Index

The machine-learning workflow therefore uses five predictors:

`VV`, `VH`, `VV_VH_diff`, `VV_VH_ratio`, and `NPI`.

## Data handling

Keep the following outside Git history:

- credentials and access tokens;
- raw Sentinel-1 `.SAFE` directories;
- Sentinel-1 ZIP archives;
- large raster collections;
- other large source datasets that are not required for the public repository.

The repository documents the processing and analysis workflows without distributing the full source-data archive.
