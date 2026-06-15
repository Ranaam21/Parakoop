"""
scripts/extract_stl_features.py

Extract geometric features from DrivAerNet++ STL mesh files and save to CSV.

Reads directly from the (possibly truncated) Globus-downloaded ZIP archives
without needing the ZIP central directory. Processes all available mesh ZIPs
in the meshes/ folder.

Output: data/drivaernet/geometry_features.csv
  columns: design_id, config, cd, length_mm, width_mm, height_mm,
           frontal_area_mm2, roof_height_mm, rear_slant_deg,
           windshield_deg, cabin_length_frac, n_vertices

Usage
─────
    cd parakoop/
    python scripts/extract_stl_features.py                  # all designs
    python scripts/extract_stl_features.py --max-per-zip 50 # quick test (50/zip)
    python scripts/extract_stl_features.py --zip N_S_WW_WM  # single zip only
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_pipeline.stl_features import extract_all_features

MESH_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'drivaernet', 'meshes')
CD_CSV    = os.path.join(os.path.dirname(__file__), '..', 'data', 'drivaernet',
                         'DrivAerNetPlusPlus_Cd_8k_Updated.csv')
OUT_CSV   = os.path.join(os.path.dirname(__file__), '..', 'data', 'drivaernet',
                         'geometry_features.csv')


def parse_args():
    pa = argparse.ArgumentParser(description='Extract STL geometry features')
    pa.add_argument('--max-per-zip', type=int, default=None,
                    help='Max designs per ZIP (default: all). Use 50 for quick test.')
    pa.add_argument('--zip', type=str, default=None,
                    help='Process only this zip stem (e.g. N_S_WW_WM)')
    pa.add_argument('--out', type=str, default=OUT_CSV,
                    help=f'Output CSV path (default: {OUT_CSV})')
    return pa.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(MESH_DIR):
        print(f'ERROR: mesh directory not found: {MESH_DIR}')
        sys.exit(1)
    if not os.path.exists(CD_CSV):
        print(f'ERROR: Cd CSV not found: {CD_CSV}')
        sys.exit(1)

    print(f'Mesh dir : {MESH_DIR}')
    print(f'Cd CSV   : {CD_CSV}')
    print(f'Output   : {args.out}')
    if args.max_per_zip:
        print(f'Cap      : {args.max_per_zip} designs/zip')
    if args.zip:
        print(f'Filter   : {args.zip} only')
    print()

    t0 = time.time()

    if args.zip:
        # Single-zip mode: temporarily rename other zips by passing a filtered mesh_dir
        import shutil, tempfile, pathlib
        src = pathlib.Path(MESH_DIR) / f'{args.zip}.zip'
        if not src.exists():
            print(f'ERROR: {src} not found')
            sys.exit(1)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(src, pathlib.Path(tmp) / src.name)
            df = extract_all_features(
                mesh_dir=tmp, cd_csv=CD_CSV, out_csv=args.out,
                max_per_zip=args.max_per_zip, verbose=True,
            )
    else:
        df = extract_all_features(
            mesh_dir=MESH_DIR, cd_csv=CD_CSV, out_csv=args.out,
            max_per_zip=args.max_per_zip, verbose=True,
        )

    elapsed = time.time() - t0
    print(f'\nExtraction complete: {len(df)} designs in {elapsed:.0f}s')
    print(f'Saved to: {args.out}')


if __name__ == '__main__':
    main()
