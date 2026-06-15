"""
Download DrivAerNet++ dataset into data/drivaernet/
Source: NeurIPS 2024, arXiv:2406.09624
GitHub:  https://github.com/Mehtab08/DrivAerNet
AICrowd: https://www.aicrowd.com/challenges/drivaernet

DrivAerNet++ is a large dataset (~39 TB full, but subsets available):
  - 8,000 car-body geometries (STL/OBJ meshes)
  - Steady-state RANS CFD (k-omega SST, Re ~ 3-5M)
  - Fields: 3D velocity, pressure, wall shear stress
  - Scalars: Cd, Cl per geometry

ACCESS NOTE:
  DrivAerNet++ requires registration on AICrowd to get the full CFD fields.
  The geometry-only subset and Cd/Cl scalars are available via the GitHub
  repo and may be available as a Hugging Face dataset (see below).

  Full data access steps:
    1. Register at https://www.aicrowd.com/challenges/drivaernet
    2. Accept the data license
    3. Download the API token from your AICrowd profile
    4. Set env var: export AICROWD_API_KEY=<your_token>
    5. Re-run this script — it will use the aicrowd-cli if token is present.

Usage:
    python scripts/download_drivaernet.py [--subset geometry_only]

Downloads to: data/drivaernet/
"""

import os
import subprocess
import sys
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "drivaernet")
GITHUB_URL = "https://github.com/Mehtab08/DrivAerNet"
HF_REPO = "neashton/DrivAerNet"  # community mirror — verify availability before use


def download(subset="geometry_only"):
    os.makedirs(DATA_DIR, exist_ok=True)

    api_key = os.environ.get("AICROWD_API_KEY", "")

    if api_key:
        _download_aicrowd(api_key, subset)
    else:
        print("=" * 70)
        print("AICROWD_API_KEY not set — falling back to public subset.")
        print("For full CFD fields (required for training):")
        print("  1. Register at https://www.aicrowd.com/challenges/drivaernet")
        print("  2. Accept the data license")
        print("  3. export AICROWD_API_KEY=<your_token>")
        print("  4. Re-run this script")
        print("=" * 70)
        _download_public_subset()


def _download_aicrowd(api_key, subset):
    try:
        import aicrowd_cli  # noqa
    except ImportError:
        print("Installing aicrowd-cli...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aicrowd-cli"])

    print(f"Downloading DrivAerNet++ via AICrowd (subset: {subset})...")
    print(f"Target: {os.path.abspath(DATA_DIR)}")

    env = os.environ.copy()
    env["AICROWD_API_KEY"] = api_key
    subprocess.check_call(
        ["aicrowd", "dataset", "download",
         "--challenge", "drivaernet",
         "--output-directory", DATA_DIR],
        env=env
    )
    print(f"\nDrivAerNet++ download complete -> {os.path.abspath(DATA_DIR)}")


def _download_public_subset():
    """
    Downloads the geometry-only public subset from the GitHub repo.
    This gives STL meshes + Cd/Cl scalars but NOT the full 3D CFD fields.
    Sufficient for geometry representation work (Stage 1) but NOT for
    training A(theta)'s steady fixed-point readout (Stage 3) — for that,
    you need the full AICrowd CFD fields.
    """
    print("Cloning DrivAerNet GitHub repo (geometry + scalar data)...")
    target = os.path.join(DATA_DIR, "DrivAerNet")
    if os.path.exists(target):
        print(f"Already exists at {target} — skipping clone.")
    else:
        subprocess.check_call(["git", "clone", "--depth=1", GITHUB_URL, target])

    print(f"\nPublic geometry subset ready at: {os.path.abspath(target)}")
    print("\nCONTENTS:")
    print("  - STL meshes for a subset of car geometries")
    print("  - Cd/Cl scalar values")
    print("  - MISSING: full 3D velocity/pressure fields (need AICrowd access)")
    print("\nNext step: register at AICrowd and set AICROWD_API_KEY for full data.")


def _print_summary(path):
    files = []
    for root, _, fnames in os.walk(path):
        for f in fnames:
            files.append(os.path.join(root, f))
    total_mb = sum(os.path.getsize(f) for f in files) / 1e6
    print(f"Files: {len(files)}  |  Total size: {total_mb:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="geometry_only",
                        choices=["geometry_only", "full"],
                        help="geometry_only (no AICrowd key needed) or full (requires key)")
    args = parser.parse_args()
    download(args.subset)
