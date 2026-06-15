"""
scripts/globus_drivaernet_download.py

Download DrivAerNet++ from Harvard Dataverse via Globus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THE DATASET LOOKS LIKE (discovered via Dataverse API)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DrivAerNet++ is split into 7 separate Harvard Dataverse datasets,
each containing ZIP archives grouped by body-style configuration:

  Dataset                DOI                    Size    File IDs
  ─────────────────────────────────────────────────────────────────
  3D Meshes (STL)        doi:10.7910/DVN/OYU2FG  522 GB  15 ZIPs
  Pressure (surface VTK) doi:10.7910/DVN/K7PWNJ  201 GB  15 ZIPs
  CFD (volume VTK)       doi:10.7910/DVN/EEYHUA   ~TB    30 ZIPs
  Wall Shear Stress      doi:10.7910/DVN/PCZYL4  ~200 GB  15 ZIPs
  Annotations            doi:10.7910/DVN/CAWRXI   small   7 files
  Sketches               doi:10.7910/DVN/JRHNAX   small   2 files
  Renderings             doi:10.7910/DVN/XKW8WI   small   2 files

Each ZIP is named by configuration, e.g.:
  F_D_WM_WW_1.zip  →  Fastback, Detailed, With-Mirrors, With-Wheels, part 1
  N_S_WWC_WM.zip   →  Notchback, Simplified, Wheels-Coarse, With-Mirrors
  E_S_WWC_WM.zip   →  Estateback, Simplified, Wheels-Coarse, With-Mirrors

For ParaKoop we need:
  ✓ 3D Meshes    — STL geometry for shape encoding
  ✓ Pressure     — surface Cp/p for phi_surface observables
  ✗ CFD VTK      — 3D volume, NOT needed (AhmedML covers wake volume)
  ✗ WSS          — optional; skip for now

Recommended download (all 3 body styles represented):
  3D Meshes   : F_D_WM_WW_1.zip (10.4 GB)  +  N_S_WWC_WM.zip (28.7 GB)  +  E_S_WWC_WM.zip (52.5 GB)
  Pressure    : F_D_WM_WW_1.zip (12.8 GB)  +  N_S_WWC_WM.zip (10.2 GB)  +  E_S_WWC_WM.zip (18.4 GB)
  Total       : ~133 GB — one representative ZIP per body style per data type

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP (two things needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. Harvard Dataverse API token
   1. Go to https://dataverse.harvard.edu/
   2. Log in (or create an account — free, instant)
   3. Top-right menu → "API Token" → "Create Token"
   4. Paste into DATAVERSE_API_TOKEN below

   NOTE: If you see "Request Access" on the dataset page, click it first
   and agree to CC BY-NC 4.0 terms — approval is usually immediate.

B. Globus Connect Personal on your Mac
   1. Download: https://www.globus.org/globus-connect-personal
   2. Install → log in with your Globus account
   3. Menu bar app → Preferences → shows YOUR_ENDPOINT_ID (UUID)
   4. Paste into YOUR_ENDPOINT_ID below

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
import globus_sdk
from globus_sdk.scopes import TransferScopes

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILL THESE IN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATAVERSE_API_TOKEN = "FILL_IN_YOUR_HARVARD_DATAVERSE_API_TOKEN"
YOUR_ENDPOINT_ID    = "FILL_IN_YOUR_GLOBUS_CONNECT_PERSONAL_UUID"

# Globus tutorial app CLIENT_ID — fine for personal use
CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"

DEST_PATH = "/Users/amit21/Desktop/Car_CFD/parakoop/data/drivaernet/"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset DOIs and the specific file IDs we want
# (file IDs obtained from the Dataverse API — hardcoded here for speed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATAVERSE_BASE = "https://dataverse.harvard.edu"

# File IDs per dataset (from /api/datasets/:persistentId/versions/:latest/files)
# Format: (doi, filename, file_id, size_gb)
DATASET_FILES = {
    # ── 3D Meshes (STL) ── doi:10.7910/DVN/OYU2FG ──────────────────
    "meshes": [
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_1.zip", 10816695, 10.4),   # Fastback detailed ×~700 designs
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_2.zip", 10816697, 10.4),
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_3.zip", 10816696, 10.4),
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_4.zip", 10816694, 10.4),
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_5.zip", 10816692, 10.4),
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_6.zip", 10816704, 10.4),
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_7.zip", 10816693, 10.4),
        ("doi:10.7910/DVN/OYU2FG", "F_D_WM_WW_8.zip", 10816705,  9.7),
        ("doi:10.7910/DVN/OYU2FG", "F_S_WWC_WM.zip",  10816700, 52.0),   # Fastback simplified
        ("doi:10.7910/DVN/OYU2FG", "F_S_WWS_WM.zip",  10816691, 63.8),
        ("doi:10.7910/DVN/OYU2FG", "N_S_WWC_WM.zip",  10816698, 28.7),   # Notchback
        ("doi:10.7910/DVN/OYU2FG", "N_S_WWS_WM.zip",  10816703, 31.5),
        ("doi:10.7910/DVN/OYU2FG", "N_S_WW_WM.zip",   10816702, 64.2),
        ("doi:10.7910/DVN/OYU2FG", "E_S_WWC_WM.zip",  10816701, 52.5),   # Estateback
        ("doi:10.7910/DVN/OYU2FG", "E_S_WW_WM.zip",   10816699, 67.6),
    ],
    # ── Surface Pressure VTK ── doi:10.7910/DVN/K7PWNJ ─────────────
    "pressure": [
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_1.zip", 10816653, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_2.zip", 10816656, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_3.zip", 10816652, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_4.zip", 10816655, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_5.zip", 10816646, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_6.zip", 10816647, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_7.zip", 10816645, 12.8),
        ("doi:10.7910/DVN/K7PWNJ", "F_D_WM_WW_8.zip", 10816658, 11.9),
        ("doi:10.7910/DVN/K7PWNJ", "F_S_WWC_WM.zip",  10816651, 18.4),
        ("doi:10.7910/DVN/K7PWNJ", "F_S_WWS_WM.zip",  10816649, 18.2),
        ("doi:10.7910/DVN/K7PWNJ", "N_S_WWC_WM.zip",  10816648, 10.2),
        ("doi:10.7910/DVN/K7PWNJ", "N_S_WWS_WM.zip",  10816654,  9.0),
        ("doi:10.7910/DVN/K7PWNJ", "N_S_WW_WM.zip",   10816657, 18.1),
        ("doi:10.7910/DVN/K7PWNJ", "E_S_WWC_WM.zip",  10816644, 18.4),
        ("doi:10.7910/DVN/K7PWNJ", "E_S_WW_WM.zip",   10816650, 18.8),
    ],
}

# Preset download bundles
PRESETS = {
    # One ZIP per body style (F + N + E) for both meshes + pressure
    # ~133 GB total — represents all 3 body styles
    "minimal": {
        "meshes":   ["F_D_WM_WW_1.zip", "N_S_WWC_WM.zip", "E_S_WWC_WM.zip"],
        "pressure": ["F_D_WM_WW_1.zip", "N_S_WWC_WM.zip", "E_S_WWC_WM.zip"],
    },
    # All fastback ZIPs (8 parts) + one N + one E for meshes + pressure
    # ~370 GB — good fastback coverage + style diversity
    "fastback_full": {
        "meshes":   ["F_D_WM_WW_1.zip", "F_D_WM_WW_2.zip", "F_D_WM_WW_3.zip",
                     "F_D_WM_WW_4.zip", "F_D_WM_WW_5.zip", "F_D_WM_WW_6.zip",
                     "F_D_WM_WW_7.zip", "F_D_WM_WW_8.zip",
                     "N_S_WWC_WM.zip", "E_S_WWC_WM.zip"],
        "pressure": ["F_D_WM_WW_1.zip", "F_D_WM_WW_2.zip", "F_D_WM_WW_3.zip",
                     "F_D_WM_WW_4.zip", "F_D_WM_WW_5.zip", "F_D_WM_WW_6.zip",
                     "F_D_WM_WW_7.zip", "F_D_WM_WW_8.zip",
                     "N_S_WWC_WM.zip", "E_S_WWC_WM.zip"],
    },
    # Meshes only for all configurations — just geometry, no VTK
    "meshes_only": {
        "meshes":   "all",
    },
    # Everything: meshes + pressure for all 15 ZIPs each
    "everything": {
        "meshes":   "all",
        "pressure": "all",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Get Globus collection UUID via Dataverse API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_globus_collection_id(doi: str) -> str | None:
    """
    Ask Harvard Dataverse for the Globus collection UUID for a dataset.

    Calls /api/datasets/:persistentId/globus-download-parameters
    with the user's API token. Returns the managed_endpoint_id UUID.
    """
    url = (f"{DATAVERSE_BASE}/api/datasets/:persistentId/"
           f"globus-download-parameters?persistentId={doi}")
    headers = {"X-Dataverse-key": DATAVERSE_API_TOKEN}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json().get("data", {})
            cid = data.get("managed_endpoint_id") or data.get("endpoint_id")
            if cid:
                return cid
            # Some versions return a different structure
            for key in ("collection_id", "globus_endpoint", "endpoint"):
                if key in data:
                    return data[key]
        elif r.status_code == 401:
            print(f"  [ERROR] Unauthorized — check your DATAVERSE_API_TOKEN")
        elif r.status_code == 403:
            print(f"  [ERROR] Forbidden — you may need to request access to {doi}")
            print(f"          Visit: {DATAVERSE_BASE}/dataset.xhtml?persistentId={doi}")
            print(f"          Click 'Request Access' then rerun.")
        else:
            print(f"  [WARN] Globus params API returned {r.status_code} for {doi}")
            print(f"  Response: {r.text[:300]}")
    except Exception as e:
        print(f"  [WARN] Could not fetch Globus collection for {doi}: {e}")
    return None


def get_direct_download_url(file_id: int) -> str:
    """Dataverse direct HTTP download URL for a file (fallback, for small files)."""
    return f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Globus authentication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def authenticate() -> globus_sdk.TransferClient:
    """Interactive Globus OAuth2 — prints URL, waits for code."""
    auth  = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    auth.oauth2_start_flow(requested_scopes=TransferScopes.all)
    url   = auth.oauth2_get_authorize_url()

    print("\n" + "═" * 62)
    print("GLOBUS AUTHENTICATION")
    print("═" * 62)
    print("\n1. Open this URL in your browser and log in with Globus:\n")
    print(f"   {url}\n")
    code = input("2. Paste the code shown after login: ").strip()

    tok = auth.oauth2_exchange_code_for_tokens(code)
    tc  = globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(
            tok.by_resource_server["transfer.api.globus.org"]["access_token"]
        )
    )
    print("✓ Globus authenticated.\n")
    return tc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Build and submit transfer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_file_list(preset_name: str) -> list[tuple]:
    """
    Returns list of (doi, filename, file_id, size_gb) tuples to download.
    """
    preset = PRESETS[preset_name]
    items  = []
    for data_type, wanted in preset.items():
        all_files = DATASET_FILES[data_type]
        if wanted == "all":
            items.extend(all_files)
        else:
            items.extend(f for f in all_files if f[1] in wanted)
    return items


def transfer_via_globus(
    tc:          globus_sdk.TransferClient,
    files:       list[tuple],
    dry_run:     bool = False,
) -> None:
    """
    For each file, get its Globus collection UUID from the Dataverse API,
    then submit a Globus transfer to your local endpoint.
    """
    os.makedirs(DEST_PATH, exist_ok=True)

    # Group files by DOI to minimise API calls
    by_doi: dict[str, list] = {}
    for doi, fname, fid, size_gb in files:
        by_doi.setdefault(doi, []).append((fname, fid, size_gb))

    total_gb = sum(f[3] for f in files)
    print(f"\n  Files to transfer : {len(files)}")
    print(f"  Estimated size    : ~{total_gb:.1f} GB")
    print(f"  Destination       : {DEST_PATH}\n")

    task_ids = []

    for doi, doi_files in by_doi.items():
        print(f"  Dataset {doi}")

        # Get Globus collection for this dataset
        collection_id = get_globus_collection_id(doi)
        if not collection_id:
            print(f"    [SKIP] Could not get Globus collection ID. "
                  f"Trying direct HTTP download instead.")
            _http_download_fallback(doi_files, doi)
            continue

        print(f"    Globus collection : {collection_id}")

        task_data = globus_sdk.TransferData(
            source_endpoint      = collection_id,
            destination_endpoint = YOUR_ENDPOINT_ID,
            label                = f"DrivAerNet++ {doi.split('/')[-1]}",
            sync_level           = "checksum",
            notify_on_succeeded  = True,
            notify_on_failed     = True,
        )

        for fname, fid, size_gb in doi_files:
            # Source path on the Globus collection
            # (Dataverse Globus paths are typically /doi-part/filename)
            doi_path  = doi.replace("doi:", "").replace("/", "/")
            src_path  = f"/{doi_path}/{fname}"
            dest_path = os.path.join(DEST_PATH, fname)
            task_data.add_item(src_path, dest_path)
            print(f"    + {fname}  ({size_gb:.1f} GB)")

        if dry_run:
            print(f"    [DRY RUN] Would submit transfer for {len(doi_files)} files")
            continue

        task_doc  = tc.submit_transfer(task_data)
        task_id   = task_doc["task_id"]
        task_ids.append(task_id)
        print(f"    ✓ Transfer submitted: {task_id}")
        print(f"      Monitor: https://app.globus.org/activity/{task_id}")

    if not dry_run and task_ids:
        print(f"\nAll transfers submitted. Watching the first one ...")
        _poll(tc, task_ids[0])


def _http_download_fallback(doi_files: list, doi: str) -> None:
    """
    Fallback: download via HTTP using Dataverse file access API.
    Practical for small files (<5 GB each); for large ZIPs Globus is preferred.
    """
    import urllib.request
    os.makedirs(DEST_PATH, exist_ok=True)
    for fname, fid, size_gb in doi_files:
        dest = os.path.join(DEST_PATH, fname)
        if os.path.exists(dest):
            print(f"    [SKIP] {fname} already exists")
            continue
        if size_gb > 5:
            print(f"    [SKIP] {fname} ({size_gb:.1f} GB) — too large for HTTP; needs Globus")
            continue
        url = get_direct_download_url(fid)
        print(f"    Downloading {fname} ({size_gb:.1f} GB) via HTTP ...")
        headers = {"X-Dataverse-key": DATAVERSE_API_TOKEN}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            while chunk := resp.read(8 * 1024 * 1024):
                f.write(chunk)
        print(f"    ✓ {fname}")


def _poll(tc: globus_sdk.TransferClient, task_id: str) -> None:
    print("  (Ctrl-C to stop watching — transfer keeps running on Globus servers)")
    try:
        while True:
            t = tc.get_task(task_id)
            done  = t.get("files_transferred", 0)
            total = t.get("files", "?")
            bps   = t.get("effective_bytes_per_second", 0)
            speed = f"  {bps/1e6:.0f} MB/s" if bps else ""
            print(f"  {t['status']}  {done}/{total} files{speed}        ", end="\r")
            if t["status"] in ("SUCCEEDED", "FAILED"):
                print(f"\n  Final: {t['status']}")
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print(f"\n  Transfer still running: https://app.globus.org/activity/{task_id}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _check_config() -> bool:
    ok = True
    if DATAVERSE_API_TOKEN == "FILL_IN_YOUR_HARVARD_DATAVERSE_API_TOKEN":
        print("ERROR: Set DATAVERSE_API_TOKEN")
        print("  → https://dataverse.harvard.edu → login → top-right menu → API Token")
        ok = False
    if YOUR_ENDPOINT_ID == "FILL_IN_YOUR_GLOBUS_CONNECT_PERSONAL_UUID":
        print("ERROR: Set YOUR_ENDPOINT_ID")
        print("  → https://www.globus.org/globus-connect-personal → install → UUID in Preferences")
        ok = False
    return ok


def _show_presets() -> None:
    print("\nAvailable presets and estimated sizes:")
    print()
    info = {
        "minimal":      ("~133 GB", "1 ZIP per body style (F+N+E), meshes + pressure"),
        "fastback_full":("~370 GB", "All fastback ZIPs + 1 N + 1 E, meshes + pressure"),
        "meshes_only":  ("~522 GB", "All STL ZIPs, no VTK"),
        "everything":   ("~723 GB", "All meshes + all pressure VTK"),
    }
    for name, (size, desc) in info.items():
        print(f"  {name:<16} {size:<10}  {desc}")
    print()


def main():
    pa = argparse.ArgumentParser(description="Download DrivAerNet++ via Globus")
    pa.add_argument("--preset", default="minimal",
                    choices=list(PRESETS.keys()),
                    help="Download bundle (default: minimal ~133 GB)")
    pa.add_argument("--list-presets", action="store_true",
                    help="Show all presets and sizes, then exit")
    pa.add_argument("--dry-run", action="store_true",
                    help="Show what would be transferred without submitting")
    pa.add_argument("--test-token", action="store_true",
                    help="Test that your Dataverse API token works, then exit")
    args = pa.parse_args()

    if args.list_presets:
        _show_presets()
        return

    if not _check_config():
        sys.exit(1)

    if args.test_token:
        # Quick check: can we reach the Dataverse API?
        doi = "doi:10.7910/DVN/OYU2FG"
        print(f"Testing API token against {doi} ...")
        cid = get_globus_collection_id(doi)
        if cid:
            print(f"✓ Token works. Globus collection ID: {cid}")
        else:
            print("✗ Could not retrieve Globus collection ID. Check token and access.")
        return

    files = build_file_list(args.preset)
    tc    = authenticate()
    transfer_via_globus(tc, files, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
