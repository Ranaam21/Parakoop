"""
koopman/geometry_predictor.py

Geometry-to-Cd regression model trained on extracted STL features.

Trains a gradient-boosted regressor on real DrivAerNet++ designs with
measured geometry (from STL meshes) and CFD-validated Cd values.

Provides:
  - predict(geom_dict)         → Cd prediction + confidence interval
  - sensitivity(geom_dict)     → ΔCd per unit change in each dimension
  - nearest_designs(geom_dict) → top-N real designs closest to given geometry
  - what_if(geom_dict, changes)→ Cd impact of specific dimensional changes

Input geometry parameters (all in mm, degrees, or fractions):
  length_mm          car overall length         (typical: 4500–5200)
  width_mm           car overall width          (typical: 900–1100)
  height_mm          car overall height         (typical: 1250–1450)
  rear_slant_deg     rear deck slope angle      (fastback~25, notchback~8, estate~5)
  windshield_deg     A-pillar slope from horiz  (typical: 30–55)
  cabin_length_frac  cabin as fraction of body  (typical: 0.40–0.60)
  roof_height_mm     peak roof height from floor(typical: 1100–1380)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble       import GradientBoostingRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import cross_val_score

# AhmedML CSV paths (relative to this file)
_AHMED_FM_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'ahmedml', 'force_mom_all.csv')
_AHMED_GEO_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'ahmedml', 'geo_parameters_all.csv')

FEATURE_COLS = [
    'length_mm', 'width_mm', 'height_mm',
    'frontal_area_mm2', 'roof_height_mm',
    'rear_slant_deg', 'windshield_deg', 'cabin_length_frac',
]

# Typical design ranges (for UI bounds and validation)
PARAM_RANGES = {
    'length_mm':          (4400,  5300, 'mm',   'Overall car length'),
    'width_mm':           (900,   1100, 'mm',   'Car width'),
    'height_mm':          (1200,  1450, 'mm',   'Car height (roof to ground)'),
    'frontal_area_mm2':   (1e6,   1.8e6,'mm²',  'Frontal area (W × H)'),
    'roof_height_mm':     (1050,  1380, 'mm',   'Peak roof height from floor'),
    'rear_slant_deg':     (2,     40,   '°',    'Rear deck/window slope angle'),
    'windshield_deg':     (25,    60,   '°',    'Windshield (A-pillar) slope'),
    'cabin_length_frac':  (0.35,  0.65, '',     'Cabin length / body length'),
}

# Typical values per body style (for "start from template" UX)
STYLE_DEFAULTS = {
    'fastback': {
        'length_mm': 4850, 'width_mm': 1020, 'height_mm': 1310,
        'frontal_area_mm2': 1335200, 'roof_height_mm': 1280,
        'rear_slant_deg': 26.0, 'windshield_deg': 42.0, 'cabin_length_frac': 0.52,
    },
    'notchback': {
        'length_mm': 4780, 'width_mm': 1022, 'height_mm': 1320,
        'frontal_area_mm2': 1350840, 'roof_height_mm': 1290,
        'rear_slant_deg': 8.5, 'windshield_deg': 38.0, 'cabin_length_frac': 0.47,
    },
    'estateback': {
        'length_mm': 4800, 'width_mm': 1020, 'height_mm': 1330,
        'frontal_area_mm2': 1356600, 'roof_height_mm': 1300,
        'rear_slant_deg': 5.0, 'windshield_deg': 35.0, 'cabin_length_frac': 0.45,
    },
}


# ── Ahmed-body Cl estimator ───────────────────────────────────────────────────

_AHMED_GEO_FEATS = [
    'slant-angle-degrees', 'body-length', 'body-height', 'body-width',
    'front-arc-diameter', 'slant-angle-length', 'slant-angle-height',
]
# Median Ahmed body dimensions (for filling missing features during inference)
_AHMED_DEFAULTS = {
    'body-length': 1000.0, 'body-height': 282.0, 'body-width': 400.0,
    'front-arc-diameter': 200.0, 'slant-angle-length': 160.0, 'slant-angle-height': 121.0,
}


class AhmedClEstimator:
    """
    Cl (lift coefficient) estimator trained on AhmedML data.

    The Ahmed body is a canonical CFD benchmark whose rear slant angle drives
    Cl in the same way as a car's fastback/notchback rear slope. This class
    provides an indicative Cl estimate for a given rear_slant_deg, labeled
    as an "Ahmed-body analogy" estimate with explicit caveats.

    Positive Cl = aerodynamic lift (upward force), negative Cl = downforce.
    """

    def __init__(self):
        self._model_cl = None
        self._model_cd = None
        self._cv_mae_cl = None
        self._trained = False

    @classmethod
    def from_csv(cls, fm_csv: str = _AHMED_FM_CSV,
                 geo_csv: str = _AHMED_GEO_CSV) -> 'AhmedClEstimator':
        est = cls()
        if not os.path.exists(fm_csv) or not os.path.exists(geo_csv):
            return est   # return untrained; caller checks _trained flag

        fm = pd.read_csv(fm_csv)
        fm.columns = [c.strip() for c in fm.columns]
        geo = pd.read_csv(geo_csv)
        df  = geo.merge(fm, on='run').dropna()

        X     = df[_AHMED_GEO_FEATS].values
        y_cl  = df['cl'].values
        y_cd  = df['cd'].values

        for target, attr in [(y_cl, '_model_cl'), (y_cd, '_model_cd')]:
            pipe = Pipeline([
                ('s', StandardScaler()),
                ('gbr', GradientBoostingRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )),
            ])
            pipe.fit(X, target)
            setattr(est, attr, pipe)

        est._cv_mae_cl = float(
            -cross_val_score(est._model_cl, X, y_cl, cv=5,
                             scoring='neg_mean_absolute_error').mean()
        )
        est._trained = True
        return est

    def _build_input(self, rear_slant_deg: float) -> np.ndarray:
        """Map rear_slant_deg → Ahmed body input vector using median body dims."""
        row = [rear_slant_deg] + [_AHMED_DEFAULTS[f] for f in _AHMED_GEO_FEATS[1:]]
        return np.array([row])

    def estimate(self, rear_slant_deg: float) -> dict:
        """
        Estimate Cl and Ahmed-analogy Cd for a given rear slant angle.

        Returns
        -------
        dict with:
          cl_estimate : predicted lift coefficient (Ahmed-body analogy)
          cd_ahmed    : Ahmed-body Cd at this slant (for reference only)
          ld_ratio    : Cl/Cd lift-to-drag (Ahmed-body basis)
          cv_mae_cl   : cross-validated MAE on AhmedML dataset
          note        : caveat string
        """
        if not self._trained:
            return {
                'cl_estimate': None, 'cd_ahmed': None, 'ld_ratio': None,
                'cv_mae_cl': None,
                'note': 'AhmedML data not found — Cl unavailable',
            }
        X    = self._build_input(rear_slant_deg)
        cl   = float(self._model_cl.predict(X)[0])
        cd_a = float(self._model_cd.predict(X)[0])
        ld   = cl / cd_a if cd_a != 0 else float('nan')
        return {
            'cl_estimate': round(cl, 4),
            'cd_ahmed':    round(cd_a, 4),
            'ld_ratio':    round(ld, 3),
            'cv_mae_cl':   round(self._cv_mae_cl, 4),
            'note': (
                'Ahmed-body analogy (slant angle proxy). '
                'Indicative trend only — scale and geometry differ from production cars.'
            ),
        }


# Shared instance loaded on demand
_ahmed_estimator: Optional[AhmedClEstimator] = None


def get_ahmed_estimator() -> AhmedClEstimator:
    global _ahmed_estimator
    if _ahmed_estimator is None:
        _ahmed_estimator = AhmedClEstimator.from_csv()
    return _ahmed_estimator


# ── Geometry → Cd predictor ───────────────────────────────────────────────────

class GeometryPredictor:
    """
    Geometry → Cd predictor trained on real DrivAerNet++ mesh + CFD data.

    Usage
    -----
    gp = GeometryPredictor.from_csv('data/drivaernet/geometry_features.csv')
    result = gp.predict({'length_mm': 4800, 'rear_slant_deg': 22, ...})
    print(result['cd_pred'], result['cd_range'])
    """

    def __init__(self):
        self.model     = None
        self.df        = None
        self._trained  = False

    @classmethod
    def from_csv(cls, features_csv: str) -> 'GeometryPredictor':
        gp = cls()
        gp.df = pd.read_csv(features_csv).dropna(subset=FEATURE_COLS + ['cd'])
        gp._train()
        return gp

    def _train(self):
        X = self.df[FEATURE_COLS].values
        y = self.df['cd'].values

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('gbr', GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])
        self.model.fit(X, y)

        cv_mae = -cross_val_score(
            self.model, X, y, cv=5, scoring='neg_mean_absolute_error'
        ).mean()
        self._cv_mae  = cv_mae
        self._y_std   = float(y.std())
        self._trained = True

    def _to_array(self, geom: dict) -> np.ndarray:
        """Fill missing features with dataset median, return feature array."""
        row = []
        for col in FEATURE_COLS:
            if col in geom:
                row.append(float(geom[col]))
            else:
                row.append(float(self.df[col].median()))
        return np.array([row])

    def predict(self, geom: dict) -> dict:
        """
        Predict Cd for given geometry.

        Returns
        -------
        dict with:
          cd_pred   : predicted drag coefficient
          cd_range  : (low, high) approximate 90% confidence interval
          cv_mae    : cross-validated MAE of the predictor
        """
        X   = self._to_array(geom)
        cd  = float(self.model.predict(X)[0])
        return {
            'cd_pred':  round(cd, 4),
            'cd_range': (round(cd - 1.65*self._cv_mae, 4),
                         round(cd + 1.65*self._cv_mae, 4)),
            'cv_mae':   round(self._cv_mae, 4),
        }

    def sensitivity(self, geom: dict, step_frac: float = 0.05) -> dict:
        """
        Compute ΔCd per unit change in each geometry parameter.

        Uses finite difference: perturb each param by ±5% of its range,
        compute ΔCd/Δparam. Returns sorted dict (most impactful first).
        """
        base_cd = self.predict(geom)['cd_pred']
        sensitivities = {}

        for col, (lo, hi, unit, desc) in PARAM_RANGES.items():
            step = (hi - lo) * step_frac
            geom_up   = {**geom, col: geom.get(col, (lo+hi)/2) + step}
            geom_down = {**geom, col: geom.get(col, (lo+hi)/2) - step}
            cd_up   = self.predict(geom_up)['cd_pred']
            cd_down = self.predict(geom_down)['cd_pred']
            dcd_dparam = (cd_up - cd_down) / (2 * step)
            sensitivities[col] = {
                'dCd_per_unit': round(float(dcd_dparam), 6),
                'unit':         unit,
                'description':  desc,
                'impact_per_10pct': round(float(dcd_dparam * (hi-lo) * 0.10), 4),
            }

        return dict(sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1]['dCd_per_unit']),
            reverse=True,
        ))

    def what_if(self, geom: dict, changes: dict) -> dict:
        """
        Predict Cd impact of specific dimensional changes.

        changes: {param: new_value} — partial overrides of geom

        Returns base Cd, new Cd, and delta for each change independently
        and combined.
        """
        base = self.predict(geom)
        results = {'base_cd': base['cd_pred'], 'changes': {}}

        # Individual impacts
        for param, new_val in changes.items():
            new_geom = {**geom, param: new_val}
            new_pred = self.predict(new_geom)
            delta    = round(new_pred['cd_pred'] - base['cd_pred'], 4)
            old_val  = geom.get(param, self.df[param].median())
            results['changes'][param] = {
                'old_value':  round(float(old_val), 2),
                'new_value':  round(float(new_val), 2),
                'unit':       PARAM_RANGES.get(param, ('','','',''))[2],
                'delta_cd':   delta,
                'new_cd':     new_pred['cd_pred'],
            }

        # Combined impact (all changes together)
        combined_geom = {**geom, **changes}
        combined_pred = self.predict(combined_geom)
        results['combined_cd']    = combined_pred['cd_pred']
        results['combined_delta'] = round(combined_pred['cd_pred'] - base['cd_pred'], 4)

        return results

    def nearest_designs(
        self,
        geom: dict,
        top_n: int = 5,
        config_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Find the top-N real DrivAerNet++ designs most geometrically similar
        to the given specification.

        Uses normalised Euclidean distance in feature space.
        config_filter: optional config string to restrict search
                       (e.g. 'F_D_WM_WW', 'N_S_WW_WM')
        """
        df = self.df.copy()
        if config_filter:
            df = df[df['config'].str.contains(config_filter, na=False)]
        if df.empty:
            df = self.df.copy()

        # Normalise by dataset std
        stds = self.df[FEATURE_COLS].std().replace(0, 1)
        query_arr = np.array([
            geom.get(c, float(self.df[c].median()))
            for c in FEATURE_COLS
        ])
        query_norm = query_arr / stds.values

        dists = []
        for _, row in df.iterrows():
            row_arr  = row[FEATURE_COLS].values.astype(float)
            row_norm = row_arr / stds.values
            dist = float(np.linalg.norm(query_norm - row_norm))
            dists.append(dist)

        df = df.copy()
        df['geom_distance'] = dists
        top = df.nsmallest(top_n, 'geom_distance').copy()
        top['cd_pred'] = self.model.predict(top[FEATURE_COLS].values)
        top = top[['design_id', 'config', 'cd',
                   'length_mm', 'width_mm', 'height_mm',
                   'rear_slant_deg', 'windshield_deg', 'cabin_length_frac',
                   'geom_distance', 'cd_pred']]
        return top.reset_index(drop=True)

    def model_quality(self) -> dict:
        return {
            'cv_mae':     round(self._cv_mae, 5),
            'n_training': len(self.df),
            'cd_range':   (round(float(self.df['cd'].min()), 4),
                           round(float(self.df['cd'].max()), 4)),
        }
