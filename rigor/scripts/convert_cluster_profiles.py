#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert cluster profile tables (ACCEPT/ABELL ASCII or VizieR ASU FITS) into
per-cluster CSVs expected by rigor/scripts/cluster_logtail_test.py.

Outputs per cluster under data/clusters/<CLUSTER>/:
- gas_profile.csv: r_kpc, n_e_cm3
- temp_profile.csv: r_kpc, kT_keV, kT_err_keV (if available)

Usage examples:
  python rigor/scripts/convert_cluster_profiles.py \
      --in data/ABELL_0426_profiles.dat.txt --cluster ABELL_0426

  python rigor/scripts/convert_cluster_profiles.py \
      --in data/asu.fit --cluster COMA

Notes:
- r_kpc is computed as the annulus mid-point: r_kpc = 0.5 * (Rin + Rout) * 1000.
- The ASCII parser expects header lines beginning with '#Name' and '###' (ACCEPT style).
- FITS tables are read via astropy; install it if missing:  py -m pip install --user astropy
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _infer_format(p: Path, fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower()
    ext = p.suffix.lower()
    if ext in {".fits", ".fit", ".fits.gz"}:
        return "fits"
    return "ascii"


def _read_ascii_accept(path: Path) -> pd.DataFrame:
    # Parse column names from the first non-empty header line starting with '#'
    header_names = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("#") and not line.strip().startswith("###"):
                # likely the names line, e.g. '#Name Rin Rout nelec ...'
                header_names = line.strip().lstrip("#").split()
                break
    if not header_names:
        raise RuntimeError(f"Could not locate a '#Name ...' header in {path}")
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        names=header_names,
        engine="python",
    )
    # Drop any fully empty rows
    df = df.dropna(how="all")
    return df


def _read_fits(path: Path) -> pd.DataFrame:
    try:
        from astropy.io import fits
        from astropy.table import Table
    except Exception as e:
        raise RuntimeError(
            "astropy is required to read FITS. Install with: py -m pip install --user astropy"
        ) from e
    with fits.open(path) as hdul:
        # find the first BinTableHDU with data
        table_hdu = None
        for h in hdul:
            if hasattr(h, "data") and h.data is not None and getattr(h, "columns", None) is not None:
                table_hdu = h
                break
        if table_hdu is None:
            raise RuntimeError(f"No table HDU found in {path}")
        tbl = Table(table_hdu.data)
        df = tbl.to_pandas()
        # normalize column names (lowercase)
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, *cands: str) -> str:
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        lc = c.lower()
        if lc in cols:
            return cols[lc]
    raise KeyError(f"Expected one of {cands} in columns={list(df.columns)}")


def _sanitize_label(s: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s.strip()]
    lab = "".join(keep).strip("_")
    return lab or "CLUSTER"


def convert_to_csv(df: pd.DataFrame, out_dir: Path, cluster_label: str) -> Tuple[Path, Path]:
    # Compute mid radius in kpc
    rin = df[_col(df, "Rin")].astype(float)
    rout = df[_col(df, "Rout")].astype(float)
    r_kpc = 0.5 * (rin + rout) * 1000.0

    # Gas (n_e)
    ne_col = _col(df, "nelec", "n_e", "ne", "NELEC")
    gas = pd.DataFrame({
        "r_kpc": r_kpc.to_numpy(),
        "n_e_cm3": pd.to_numeric(df[ne_col], errors="coerce").to_numpy(),
    })

    # Temperature (if present)
    temp_path = None
    tx_col = None
    try:
        tx_col = _col(df, "Tx", "kT", "T_keV")
    except KeyError:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    gas_path = out_dir / "gas_profile.csv"
    gas.to_csv(gas_path, index=False)

    if tx_col is not None:
        tx = pd.to_numeric(df[tx_col], errors="coerce")
        try:
            txe_col = _col(df, "Txerr", "kT_err", "T_err")
            txe = pd.to_numeric(df[txe_col], errors="coerce")
        except KeyError:
            txe = None
        tdf = pd.DataFrame({"r_kpc": r_kpc.to_numpy(), "kT_keV": tx.to_numpy()})
        if txe is not None:
            tdf["kT_err_keV"] = txe.to_numpy()
        temp_path = out_dir / "temp_profile.csv"
        tdf.to_csv(temp_path, index=False)

    return gas_path, temp_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input file path (.dat/.txt ASCII or .fit/.fits)")
    ap.add_argument("--cluster", dest="cluster", default=None, help="Cluster label for output directory and optional row filter")
    ap.add_argument("--format", dest="fmt", default=None, choices=["ascii", "fits"], help="Override input format detection")
    ap.add_argument("--out_base", default="data/clusters", help="Base directory for per-cluster outputs")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    fmt = _infer_format(in_path, args.fmt)
    if fmt == "ascii":
        df = _read_ascii_accept(in_path)
    else:
        df = _read_fits(in_path)

    # Determine label and optional row filter
    cluster_label = args.cluster
    name_col = None
    try:
        name_col = _col(df, "Name")
    except KeyError:
        name_col = None

    if cluster_label and name_col:
        mask = df[name_col].astype(str).str.upper() == cluster_label.upper()
        if mask.any():
            df = df[mask].reset_index(drop=True)
    if cluster_label is None:
        if name_col and df[name_col].nunique() == 1:
            cluster_label = str(df[name_col].iloc[0])
        else:
            # fallback to file stem
            cluster_label = in_path.stem
    cluster_label = _sanitize_label(cluster_label)

    out_dir = Path(args.out_base) / cluster_label
    gas_path, temp_path = convert_to_csv(df, out_dir, cluster_label)

    print("{\n  \"cluster\": \"%s\",\n  \"out_dir\": \"%s\",\n  \"gas_profile\": \"%s\",\n  \"temp_profile\": %s\n}" % (
        cluster_label,
        str(out_dir).replace("\\", "/"),
        str(gas_path).replace("\\", "/"),
        ("\"%s\"" % str(temp_path).replace("\\", "/")) if temp_path else "null",
    ))


if __name__ == "__main__":
    main()