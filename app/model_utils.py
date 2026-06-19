"""
app/model_utils.py

Cached model + dataset loading for the Streamlit app.
"""

from __future__ import annotations
import os
import sys

import numpy as np
import torch
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource(show_spinner=False)
def load_model_and_dataset():
    """Load unified checkpoint + dataset once; shared across all Streamlit reruns."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from data_pipeline.unified_loader import load_unified
    from koopman.model import ParaKoopModel

    dataset = load_unified(verbose=False)

    ckpt_path = os.path.join(_ROOT, 'checkpoints', 'unified', 'parakoop_unified_best.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ParaKoopModel(**ckpt['model_cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, dataset


def theta_from_sliders(
    style: str,
    rear_slant_deg: float,
    height_mm: float,
    width_height_ratio: float,
    cabin_frac: float,
    detail_flag: float,
    ref_length_mm: float = 4800.0,
) -> np.ndarray:
    """Build 8-dim theta from UI slider values."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from data_pipeline.unified_loader import THETA_MIN, THETA_MAX

    style_oh = {
        'fastback':   [1., 0., 0.],
        'notchback':  [0., 1., 0.],
        'estateback': [0., 0., 1.],
    }[style]

    h_l = height_mm / ref_length_mm
    theta = np.array(
        style_oh + [h_l, width_height_ratio, rear_slant_deg / 90.0,
                    cabin_frac, detail_flag],
        dtype=np.float32,
    )
    return np.clip(theta, THETA_MIN, THETA_MAX)


@torch.no_grad()
def predict(model, theta_np: np.ndarray) -> dict:
    """Forward pass: theta → {'Cd': float, 'Cl': float}."""
    t = torch.tensor(theta_np, dtype=torch.float32).unsqueeze(0)
    cd, cl, _ = model.forward_cd_only(t)
    return {'Cd': float(cd.item()), 'Cl': float(cl.item())}
