# Dataset Guide

## What's included in this repo

The following small files are committed and sufficient to run the Streamlit app and inference:

| File | Size | Purpose |
|---|---|---|
| `data/drivaernet/geometry_features.csv` | 120 KB | Geometry → Cd predictions (Koopman input) |
| `data/drivaernet/DrivAerNetPlusPlus_Cd_8k_Updated.csv` | 268 KB | Full Cd/Cl scalar lookup (8 121 designs) |
| `data/drivaernet/balanced_sample_500.csv` | 20 KB | Balanced 500-design training sample |
| `data/ahmedml/force_mom_all.csv` | 20 KB | AhmedML consolidated force/moment data |
| `data/ahmedml/force_mom_varref_all.csv` | 16 KB | AhmedML variable-reference force data |
| `data/ahmedml/geo_parameters_all.csv` | 76 KB | AhmedML geometry parameters (all runs) |
| `checkpoints/unified/parakoop_unified_best.pt` | 57 MB | Trained unified Koopman model |
| `checkpoints/drivaernet/parakoop_drivaernet_best.pt` | 321 KB | Trained DrivAerNet Koopman model |

## Full datasets (required for re-training)

### DrivAerNet++ (~39 TB full / ~100 GB practical subset)

- **GitHub (geometry + Cd scalars, no key needed):**
  ```
  python scripts/download_drivaernet.py --subset geometry_only
  ```
- **Full CFD fields (velocity, pressure, WSS):**
  1. Register at <https://www.aicrowd.com/challenges/drivaernet>
  2. Accept the data licence and copy your API token
  3. `export AICROWD_API_KEY=<your_token>`
  4. `python scripts/download_drivaernet.py --subset full`
- **Author's repo:** <https://github.com/Mehtab08/DrivAerNet>

Expected layout after download:
```
data/drivaernet/
  meshes/          # STL/OBJ car bodies (~21 GB extracted)
  pressure/        # 3D pressure fields (~25 GB)
  annotations/     # per-design metadata (~49 GB)
  renderings/      # render images (~1.8 GB)
  geometry_features.csv
  DrivAerNetPlusPlus_Cd_8k_Updated.csv
```

### AhmedML (~389 GB)

- **Hugging Face (automated download):**
  ```
  python scripts/download_ahmedml.py
  ```
  Dataset: `neashton/ahmedml` on Hugging Face Hub (downloads STL + VTU + CSV, skips images/slices)

Expected layout after download:
```
data/ahmedml/
  force_mom_all.csv
  force_mom_varref_all.csv
  geo_parameters_all.csv
  run_1/ … run_76/
    ahmed_X.stl          # geometry mesh
    volume_X.vtu         # full 3D fields (Koopman observables)
    force_mom_X.csv
    geo_parameters_X.csv
```

## Personal backup (owner only)

Full mesh zips and DrivAerNet CSVs are also stored at:
<https://drive.google.com/drive/folders/1M6ZB3IqNsPrLAb1lb6Fs8pFo55Qi0e6K>
