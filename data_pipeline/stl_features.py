"""
data_pipeline/stl_features.py

Streaming geometric feature extractor for DrivAerNet++ STL meshes.

Reads directly from truncated ZIP files (no central directory needed)
by scanning local PK headers. Extracts aerodynamically relevant shape
parameters from each design's ASCII STL surface mesh.

Features extracted
──────────────────
  length_mm          : overall car length (X span)
  width_mm           : car width  (Y span)
  height_mm          : car height (Z span)
  frontal_area_mm2   : approx frontal area (W × H)
  roof_height_mm     : peak roof height from ground
  rear_slant_deg     : rear deck/windscreen slope angle from horizontal
  windshield_deg     : A-pillar / windshield slope from horizontal
  cabin_length_frac  : cabin length as fraction of total body length
  n_vertices         : mesh resolution

Usage
─────
    from data_pipeline.stl_features import extract_all_features

    df = extract_all_features(
        mesh_dir='data/drivaernet/meshes/',
        cd_csv  ='data/drivaernet/DrivAerNetPlusPlus_Cd_8k_Updated.csv',
        out_csv ='data/drivaernet/geometry_features.csv',
        max_per_zip=200,   # cap designs per zip for speed
    )
"""

from __future__ import annotations

import io
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd

# ── Streaming ZIP reader ──────────────────────────────────────────────────────

LOCAL_HEADER_SIG = b'PK\x03\x04'
_HEADER_FMT      = '<4sHHHHHIIIHH'
_HEADER_LEN      = struct.calcsize(_HEADER_FMT)


def _stream_stl_files(
    zip_path: str,
    max_files: Optional[int] = None,
    chunk_size: int = 512 * 1024 * 1024,  # 512 MB read window
) -> Generator[tuple[str, bytes], None, None]:
    """
    Yield (filename, raw_bytes) for each STL found in a (possibly truncated)
    ZIP file by scanning local file headers sequentially.

    Works without the central directory — handles incomplete Globus downloads.
    """
    yielded = 0
    with open(zip_path, 'rb') as fh:
        # Read in chunks to avoid loading the entire multi-GB ZIP
        buffer   = b''
        file_pos = 0

        while True:
            chunk  = fh.read(chunk_size)
            if not chunk:
                break
            buffer = buffer + chunk

            pos = 0
            while pos < len(buffer) - _HEADER_LEN:
                idx = buffer.find(LOCAL_HEADER_SIG, pos)
                if idx == -1:
                    # No more headers in buffer — keep tail in case header
                    # straddles the chunk boundary
                    buffer = buffer[max(0, len(buffer) - 30):]
                    break

                try:
                    (sig, ver, flags, comp, mtime, mdate, crc,
                     comp_size, uncomp_size, fname_len, extra_len
                     ) = struct.unpack_from(_HEADER_FMT, buffer, idx)
                except struct.error:
                    pos = idx + 1
                    continue

                fname_start = idx + _HEADER_LEN
                fname_end   = fname_start + fname_len
                extra_end   = fname_end + extra_len
                data_end    = extra_end + comp_size

                if data_end > len(buffer):
                    # File data not yet in buffer — stop scanning, read more
                    buffer = buffer[idx:]
                    break

                fname = buffer[fname_start:fname_end].decode('utf-8', errors='replace')

                if fname.endswith('.stl') and comp_size > 0:
                    comp_data = buffer[extra_end:data_end]
                    try:
                        if comp == 8:    # deflate
                            raw = zlib.decompress(comp_data, -15)
                        elif comp == 0:  # stored
                            raw = comp_data
                        else:
                            pos = extra_end
                            continue
                        yield fname, raw
                        yielded += 1
                        if max_files and yielded >= max_files:
                            return
                    except zlib.error:
                        pass

                pos = data_end if comp_size > 0 else idx + 1


# ── Geometry feature extraction ───────────────────────────────────────────────

_VERTEX_PATTERN = re.compile(
    rb'vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)'
)


def _is_binary_stl(stl_bytes: bytes) -> bool:
    """Binary STL: 80-byte header + 4-byte triangle count + 50×N bytes.
    ASCII STL starts with 'solid'. We use the triangle-count cross-check."""
    if len(stl_bytes) < 84:
        return False
    if stl_bytes[:5].lower() == b'solid':
        return False
    n_tri = struct.unpack_from('<I', stl_bytes, 80)[0]
    expected = 84 + 50 * n_tri
    return abs(expected - len(stl_bytes)) < 10


def _parse_vertices(stl_bytes: bytes) -> np.ndarray:
    """Extract vertex coordinates from ASCII or binary STL → (N, 3) float32."""
    if _is_binary_stl(stl_bytes):
        n_tri = struct.unpack_from('<I', stl_bytes, 80)[0]
        if n_tri == 0 or 84 + 50 * n_tri > len(stl_bytes):
            return np.zeros((0, 3), dtype=np.float32)
        # Each 50-byte block: 12B normal + 12B v1 + 12B v2 + 12B v3 + 2B attr
        # Extract all 3 vertices per triangle
        data = np.frombuffer(stl_bytes, dtype=np.float32,
                             count=n_tri * 12, offset=80 + 4)
        # data shape: (n_tri × 12,) floats — nx ny nz | v1x v1y v1z | v2x v2y v2z | v3x v3y v3z
        # but attribute bytes interrupt: 50 bytes = 12 floats + 2 bytes  → can't use frombuffer directly
        # Parse struct per block instead
        # Structured dtype: 50 bytes per triangle exactly
        _tri_dtype = np.dtype([
            ('normal', np.float32, (3,)),
            ('v1',     np.float32, (3,)),
            ('v2',     np.float32, (3,)),
            ('v3',     np.float32, (3,)),
            ('attr',   np.uint16),
        ])
        tris  = np.frombuffer(stl_bytes, dtype=_tri_dtype, count=n_tri, offset=84)
        verts = np.concatenate([tris['v1'], tris['v2'], tris['v3']])
        return verts.astype(np.float32)
    # ASCII STL
    matches = _VERTEX_PATTERN.findall(stl_bytes)
    if not matches:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(
        [[float(x), float(y), float(z)] for x, y, z in matches],
        dtype=np.float32,
    )


def extract_features(stl_bytes: bytes, design_id: str = '') -> dict:
    """
    Extract aerodynamic geometry features from ASCII or binary STL files.

    Coordinate convention (DrivAerNet++):
        X : longitudinal (front → rear)
        Y : lateral      (left → right, symmetric)
        Z : vertical     (bottom → top)

    Simplified meshes use mm; detailed (F_D) meshes use meters — auto-detected.
    Output is always in mm.
    """
    verts = _parse_vertices(stl_bytes)
    if len(verts) < 100:
        return {}

    mn, mx = verts.min(axis=0), verts.max(axis=0)
    L = float(mx[0] - mn[0])   # length
    W = float(mx[1] - mn[1])   # width
    H = float(mx[2] - mn[2])   # height

    # Auto-detect meters vs mm: a real car length is 4–6m or 4000–6000mm
    if L < 20:   # values in meters → convert to mm
        scale = 1000.0
        verts = verts * scale
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        L, W, H = L * scale, W * scale, H * scale

    # ── Rear slant angle ───────────────────────────────────────────────
    # Use rear 25% of body, upper 50% of height (avoids bumper/wheel noise)
    rear_mask   = (verts[:, 0] > mn[0] + L * 0.75) & (verts[:, 2] > mn[2] + H * 0.5)
    rear_verts  = verts[rear_mask]
    rear_slant  = 0.0
    if len(rear_verts) > 50:
        X, Z   = rear_verts[:, 0], rear_verts[:, 2]
        if X.std() > 1.0:
            slope      = float(np.polyfit(X, Z, 1)[0])
            rear_slant = round(abs(float(np.degrees(np.arctan(slope)))), 1)

    # ── Windshield / A-pillar angle ────────────────────────────────────
    # Front 25% of body, upper 40% of height
    front_mask  = (verts[:, 0] < mn[0] + L * 0.25) & (verts[:, 2] > mn[2] + H * 0.6)
    front_verts = verts[front_mask]
    wind_angle  = 0.0
    if len(front_verts) > 50:
        X, Z   = front_verts[:, 0], front_verts[:, 2]
        if X.std() > 1.0:
            slope     = float(np.polyfit(X, Z, 1)[0])
            wind_angle = round(abs(float(np.degrees(np.arctan(slope)))), 1)

    # ── Roof height (mid-body peak) ────────────────────────────────────
    mid_mask   = (verts[:, 0] > mn[0] + L * 0.3) & (verts[:, 0] < mn[0] + L * 0.6)
    mid_verts  = verts[mid_mask]
    roof_h     = float(mid_verts[:, 2].max() - mn[2]) if len(mid_verts) else H

    # ── Cabin length fraction ──────────────────────────────────────────
    # Estimate cabin as region where roof is above 80% of roof height
    top_mask   = verts[:, 2] > mn[2] + roof_h * 0.80
    top_verts  = verts[top_mask]
    cabin_frac = 0.0
    if len(top_verts) > 0:
        cabin_len  = float(top_verts[:, 0].max() - top_verts[:, 0].min())
        cabin_frac = round(cabin_len / L, 3) if L > 0 else 0.0

    return {
        'design_id':        design_id,
        'length_mm':        round(L, 1),
        'width_mm':         round(W, 1),
        'height_mm':        round(H, 1),
        'frontal_area_mm2': round(W * H, 0),
        'roof_height_mm':   round(roof_h, 1),
        'rear_slant_deg':   rear_slant,
        'windshield_deg':   wind_angle,
        'cabin_length_frac':cabin_frac,
        'n_vertices':       len(verts),
    }


# ── Batch extraction ──────────────────────────────────────────────────────────

# Maps zip filename stem → CSV config prefix
_ZIP_TO_CONFIG = {
    'N_S_WW_WM':   'N_S_WW_WM',
    'N_S_WWC_WM':  'N_S_WWC_WM',
    'N_S_WWS_WM':  'N_S_WWS_WM',
    'F_S_WWC_WM':  'F_S_WWC_WM',
    'F_S_WWS_WM':  'F_S_WWS_WM',
    'E_S_WW_WM':   'E_S_WW_WM',
    'E_S_WWC_WM':  'E_S_WWC_WM',
    'F_D_WM_WW_1': 'F_D_WM_WW',
    'F_D_WM_WW_2': 'F_D_WM_WW',
}


def _design_id_from_stl_name(stl_name: str, zip_stem: str) -> str:
    """Convert STL filename to design ID matching the CSV format."""
    base = Path(stl_name).stem   # e.g. 'N_S_WW_WM_001'
    # F_D_WM_WW_1 zip has subfolder: '3DMeshesSTL/F_D_WM_WW_1/F_D_WM_WW_0001.stl'
    base = Path(stl_name).stem.replace('/', '_').split('_')
    # reconstruct: join until we get a numeric tail
    # STL names are like 'N_S_WW_WM_001' or 'F_D_WM_WW_0001'
    return Path(stl_name).stem.replace('/', '_')


def extract_all_features(
    mesh_dir:    str,
    cd_csv:      str,
    out_csv:     str,
    max_per_zip: int = 200,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Extract geometry features for all available mesh ZIPs and merge with Cd.

    Parameters
    ----------
    mesh_dir    : folder containing the mesh ZIP files
    cd_csv      : DrivAerNetPlusPlus_Cd_8k_Updated.csv path
    out_csv     : output path for the merged feature CSV
    max_per_zip : max designs to process per ZIP (for speed; None = all)
    verbose     : print progress

    Returns merged DataFrame with columns:
        design_id, cd, length_mm, width_mm, height_mm,
        rear_slant_deg, windshield_deg, cabin_length_frac, ...
    """
    # Load Cd reference
    cd_df = pd.read_csv(cd_csv)
    cd_df.columns = ['design_id', 'cd']
    cd_lookup = dict(zip(cd_df['design_id'], cd_df['cd']))

    mesh_dir = Path(mesh_dir)
    all_rows = []

    for zip_path in sorted(mesh_dir.glob('*.zip')):
        stem = zip_path.stem
        if stem not in _ZIP_TO_CONFIG:
            if verbose:
                print(f'  Skipping unknown zip: {zip_path.name}')
            continue

        if verbose:
            print(f'  Processing {zip_path.name} ...')

        count = 0
        for stl_name, stl_bytes in _stream_stl_files(str(zip_path), max_files=max_per_zip):
            # Derive design_id from STL filename
            design_id = Path(stl_name).stem   # e.g. 'N_S_WW_WM_001'
            # Pad 3-digit IDs to 4-digits if needed (N_S_WW_WM_001 → N_S_WW_WM_0001)
            parts = design_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 3:
                design_id = parts[0] + '_0' + parts[1]

            cd_val = cd_lookup.get(design_id)
            if cd_val is None:
                # try without padding
                cd_val = cd_lookup.get(Path(stl_name).stem)

            feats = extract_features(stl_bytes, design_id)
            if not feats:
                continue

            feats['cd']     = cd_val
            feats['config'] = _ZIP_TO_CONFIG[stem]
            all_rows.append(feats)
            count += 1

        if verbose:
            print(f'    → {count} designs extracted')

    if not all_rows:
        print('WARNING: no STL features extracted')
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[df['cd'].notna()].copy()
    df = df.sort_values('design_id').reset_index(drop=True)

    os.makedirs(Path(out_csv).parent, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f'\nSaved {len(df)} designs with geometry features → {out_csv}')

    return df
