"""
Selective AhmedML download — skips PNG images, gets only what Koopman training needs:
  - ahmed_X.stl          geometry mesh
  - volume_X.vtu         full 3D velocity/pressure fields (Koopman observables)
  - force_mom_X.csv      Cd, Cl per run
  - geo_parameters_X.csv geometry parameters
  - force_mom_all.csv    combined scalars (root level)
  - geo_parameters_all.csv combined parameters (root level)
"""

import os
from huggingface_hub import hf_hub_download, list_repo_files

REPO   = "neashton/ahmedml"
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "data", "ahmedml")
KEEP   = (".stl", ".vtu", ".csv")   # skip .vtp slices and .png images for now

def should_download(path):
    if "/images/" in path:
        return False
    if "/slices/" in path:   # skip cross-section slices — volume_X.vtu is enough
        return False
    return any(path.endswith(ext) for ext in KEEP)

def download():
    os.makedirs(OUTDIR, exist_ok=True)
    all_files = list(list_repo_files(REPO, repo_type="dataset"))
    targets   = [f for f in all_files if should_download(f)]

    print(f"AhmedML: {len(all_files)} total files → downloading {len(targets)} (skipping images/slices)")
    print(f"Target : {os.path.abspath(OUTDIR)}\n")

    for i, path in enumerate(targets, 1):
        dest = os.path.join(OUTDIR, path)
        if os.path.exists(dest):
            print(f"[{i}/{len(targets)}] skip (exists): {path}")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[{i}/{len(targets)}] {path}")
        hf_hub_download(
            repo_id=REPO,
            filename=path,
            repo_type="dataset",
            local_dir=OUTDIR,
        )

    print(f"\nDone. Files in {OUTDIR}:")
    total = sum(os.path.getsize(os.path.join(r,f))
                for r,_,fs in os.walk(OUTDIR) for f in fs)
    print(f"  Total size: {total/1e9:.2f} GB")

if __name__ == "__main__":
    download()
