"""Survey the structure of the CUGN climatology NetCDF files.

Walks every ``*.nc`` file in ``$OS_SPRAY/CUGN/Climatology`` and reports, for
each file: byte size, dimensions, coordinates, data variables (with dims,
dtype, shape, and units/long_name where present) and global attributes.

The per-file details are written as JSON to ``climatology_structure.json`` in
this directory so the human-readable report can be regenerated without
re-reading the (~1.6 GB) data. A short summary is printed to stdout.

Run with the project convention (see CLAUDE.md):

    conda run -n ocean14 python inspect_climatology_files.py

The files are NetCDF-4/HDF5, so xarray is opened with the ``h5netcdf`` engine.
"""

import os
import json
import glob

import numpy as np
import xarray as xr


def _jsonable(value):
    """Coerce numpy / xarray attribute values into JSON-serialisable types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value


def describe_file(path):
    """Return a dict describing one NetCDF file's structure."""
    info = {
        "filename": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
    }
    # ``decode_times=False`` keeps time coords as raw numbers so we can report
    # their units/calendar attributes rather than have xarray consume them.
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as ds:
        info["dims"] = {str(k): int(v) for k, v in ds.sizes.items()}
        info["coords"] = sorted(str(c) for c in ds.coords)

        variables = {}
        for name, da in ds.data_vars.items():
            variables[str(name)] = {
                "dims": [str(d) for d in da.dims],
                "shape": [int(s) for s in da.shape],
                "dtype": str(da.dtype),
                "long_name": _jsonable(da.attrs.get("long_name", "")),
                "units": _jsonable(da.attrs.get("units", "")),
            }
        info["data_vars"] = variables

        # Coordinate detail (units / range) helps interpret axes later.
        coord_detail = {}
        for name in ds.coords:
            da = ds[name]
            entry = {
                "dims": [str(d) for d in da.dims],
                "shape": [int(s) for s in da.shape],
                "dtype": str(da.dtype),
                "units": _jsonable(da.attrs.get("units", "")),
                "long_name": _jsonable(da.attrs.get("long_name", "")),
            }
            # Numeric coords: record min/max for context (skip if not numeric).
            if np.issubdtype(da.dtype, np.number) and da.size:
                vals = da.values
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    entry["min"] = float(np.min(finite))
                    entry["max"] = float(np.max(finite))
            coord_detail[str(name)] = entry
        info["coord_detail"] = coord_detail

        info["global_attrs"] = {
            str(k): _jsonable(v) for k, v in ds.attrs.items()
        }
    return info


def main():
    os_spray = os.environ.get("OS_SPRAY")
    if not os_spray:
        raise SystemExit("OS_SPRAY environment variable is not set.")

    clim_dir = os.path.join(os_spray, "CUGN", "Climatology")
    paths = sorted(glob.glob(os.path.join(clim_dir, "*.nc")))
    if not paths:
        raise SystemExit(f"No .nc files found in {clim_dir}")

    print(f"Found {len(paths)} NetCDF files in {clim_dir}\n")

    records = []
    for path in paths:
        rec = describe_file(path)
        records.append(rec)
        nvars = len(rec["data_vars"])
        dims = ", ".join(f"{k}={v}" for k, v in rec["dims"].items())
        print(f"{rec['filename']:28s}  {rec['size_bytes']/1e6:8.2f} MB  "
              f"{nvars:2d} vars  dims: {dims}")

    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "climatology_structure.json")
    with open(out_json, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nWrote structure JSON -> {out_json}")


if __name__ == "__main__":
    main()
