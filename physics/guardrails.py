"""
physics/guardrails.py  —  ParaKoop automotive edition

Guardrail set for external automotive aerodynamics.
Only 5 numbers — 3 flow-regime + 2 output-range — reflecting the lighter
set defined in the ParaKoop scope (Paper4_Automotive_Koopman_HHL_ParaKoop_Scope.txt §5.9).

Removed from Paper 1 (ALD-specific, not relevant here):
  Pr, Nu, Bi, Sc, Sh, Pe_h, Pe_m, Da  — chemical/thermal/species regime checks

Kept:
  Re  — Reynolds number       (turbulent external aero regime)
  Ma  — Mach number           (incompressibility assumption)
  Eu  — Euler number          (pressure-force / dynamic-pressure sanity)

Added (automotive output-range checks):
  Cd  — drag coefficient      [0.1, 0.5]
  LD  — lift-to-drag ratio    [0.0, 5.0]   (|Cl/Cd|, magnitude)

Usage
-----
    from physics.guardrails import GuardrailEngine, GuardrailBounds

    engine = GuardrailEngine()
    result = engine.check({"Re": 3e6, "Ma": 0.10, "Eu": 0.30, "Cd": 0.28, "LD": 0.5})
    print(result.passed)        # True
    print(result.summary())

    # Override a bound (e.g. tighter Cd window during search)
    engine.update_bounds(Cd=(0.15, 0.35))

Enforcement in the inverse-design loop:
    This engine is called as a HARD CHECK with reject-and-resample —
    not a soft training penalty. Any result with passed=False causes
    the candidate to be discarded and resampled.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple


# ── Default bounds ─────────────────────────────────────────────────────────
# Automotive external aerodynamics (passenger cars / SUVs / race cars).
# Derived from DrivAerNet++ and AhmedML operating conditions.

DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Flow-regime checks
    "Re": (1e5,  8e6),    # external aero: laminar boundary layer onset ~1e5,
                           # upper bound covers heavy trucks / high-speed test
    "Ma": (0.0,  0.3),    # incompressible assumption; >0.3 needs compressible model
    "Eu": (0.01, 5.0),    # pressure-coefficient sanity band for bluff bodies

    # Automotive output-range checks (as specified in scope §5.9)
    "Cd": (0.1,  0.5),    # 0.1: hyper-aerodynamic; 0.5: boxy / truck territory
    "LD": (0.0,  5.0),    # |Cl/Cd| magnitude; most cars 0-1, race cars up to ~3-4
}

# Severity: how much each violation decrements confidence [0–1]
SEVERITY: Dict[str, float] = {
    "Re":  0.15,   # turbulence regime shift — model may extrapolate
    "Ma":  0.40,   # compressibility is a hard model-switch; high penalty
    "Eu":  0.15,   # pressure sanity — moderate
    "Cd":  0.20,   # outside training distribution of AhmedML/DrivAerNet++
    "LD":  0.10,   # L/D is a derived check; softer penalty
}


# ══════════════════════════════════════════════════════════════════════════
# Data structures  (same interface as Paper 1 — drop-in compatible)
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class GuardrailBounds:
    """User-controllable bounds for each dimensionless number / output range."""
    Re: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Re"])
    Ma: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Ma"])
    Eu: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Eu"])
    Cd: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Cd"])
    LD: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["LD"])

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GuardrailBounds":
        return cls(**{k: tuple(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclasses.dataclass
class ViolationRecord:
    symbol:      str
    value:       float
    lo:          float
    hi:          float
    severity:    float
    reason_code: str    # machine-readable tag
    message:     str    # human-readable explanation

    @property
    def direction(self) -> str:
        return "above_max" if self.value > self.hi else "below_min"


@dataclasses.dataclass
class CheckResult:
    violations:      List[ViolationRecord]
    confidence:      float          # 1.0 = fully trusted; <0.5 = discard candidate
    special_flags:   List[str]
    recommendations: List[str]

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        lines = [f"Confidence: {self.confidence:.2f}  |  Violations: {len(self.violations)}"]
        for v in self.violations:
            lines.append(
                f"  [{v.reason_code}] {v.symbol}={v.value:.4g} "
                f"(allowed [{v.lo:.4g}, {v.hi:.4g}])  — {v.message}"
            )
        for flag in self.special_flags:
            lines.append(f"  ! {flag}")
        for rec in self.recommendations:
            lines.append(f"  -> {rec}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════

class GuardrailEngine:
    """
    Checks computed values against bounds and returns a CheckResult.

    Used as a HARD GATE in the inverse-design loop:
        result = engine.check(candidate_values)
        if not result.passed:
            resample()   # reject — do not refine further
    """

    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self, bounds: Optional[GuardrailBounds] = None):
        self.bounds = bounds or GuardrailBounds()

    def check(self, values: Dict[str, float]) -> CheckResult:
        """
        Parameters
        ----------
        values : dict  e.g. {"Re": 3e6, "Ma": 0.10, "Eu": 0.30, "Cd": 0.28, "LD": 0.5}
            Any subset of the 5 automotive guardrail keys.
            Unknown keys are silently ignored.

        Returns
        -------
        CheckResult
        """
        violations: List[ViolationRecord] = []
        special_flags: List[str] = []
        recommendations: List[str] = []

        bounds_dict = self.bounds.as_dict()

        for symbol, value in values.items():
            if symbol not in bounds_dict:
                continue
            lo, hi = bounds_dict[symbol]
            if value < lo or value > hi:
                violations.append(self._make_violation(symbol, value, lo, hi))

        # ── Special physics flags ──────────────────────────────────────────

        # Ma > 0.3: incompressible surrogate invalid — hard model switch needed
        if "Ma" in values and values["Ma"] > 0.3:
            special_flags.append("COMPRESSIBILITY_SWITCH")
            recommendations.append(
                f"Ma={values['Ma']:.3f} > 0.3: incompressible assumption breaks down. "
                f"Switch to compressible flow model (rhoPimpleFoam / compressible surrogate). "
                f"Outside DrivAerNet++ training regime — candidate should be discarded."
            )

        # Re > 5e6: deep turbulence regime — check if surrogate was trained here
        if "Re" in values and values["Re"] > 5e6:
            special_flags.append("HIGH_RE_EXTRAPOLATION")
            recommendations.append(
                f"Re={values['Re']:.2e} > 5e6: approaching upper edge of DrivAerNet++ "
                f"training distribution. Prediction confidence reduced; prefer OpenFOAM "
                f"validation for this candidate."
            )

        # Cd < 0.15: physically extreme — flag for extra validation
        if "Cd" in values and values["Cd"] < 0.15:
            special_flags.append("EXTREME_LOW_DRAG")
            recommendations.append(
                f"Cd={values['Cd']:.3f} < 0.15: exceptionally low drag — verify geometry "
                f"is physically realistic and not a surrogate artefact. "
                f"Schedule OpenFOAM ground-truth run."
            )

        # ── Confidence score ───────────────────────────────────────────────
        confidence = 1.0
        for v in violations:
            confidence -= v.severity
        confidence = max(0.0, confidence)

        if confidence < self.CONFIDENCE_THRESHOLD:
            recommendations.append(
                f"Confidence {confidence:.2f} < {self.CONFIDENCE_THRESHOLD}: "
                f"candidate fails guardrail check — discard and resample."
            )

        return CheckResult(
            violations=violations,
            confidence=round(confidence, 4),
            special_flags=special_flags,
            recommendations=recommendations,
        )

    def update_bounds(self, **kwargs):
        """Update individual bounds in-place. e.g. engine.update_bounds(Cd=(0.15, 0.35))"""
        for key, val in kwargs.items():
            if hasattr(self.bounds, key):
                setattr(self.bounds, key, tuple(val))
            else:
                raise KeyError(f"Unknown automotive guardrail: '{key}'. "
                               f"Valid keys: {list(self.bounds.as_dict().keys())}")

    @staticmethod
    def _make_violation(symbol: str, value: float, lo: float, hi: float) -> ViolationRecord:
        direction = "above_max" if value > hi else "below_min"
        severity = SEVERITY.get(symbol, 0.10)

        messages = {
            ("Re",  "above_max"): "Re exceeds upper training bound; deep turbulence regime, surrogate may extrapolate.",
            ("Re",  "below_min"): "Re below automotive range; check vehicle speed and characteristic length.",
            ("Ma",  "above_max"): "Compressibility non-negligible; incompressible surrogate invalid above Ma=0.3.",
            ("Ma",  "below_min"): "Ma < 0 is unphysical.",
            ("Eu",  "above_max"): "Euler number unusually high; check pressure reference or geometry.",
            ("Eu",  "below_min"): "Euler number near zero; check dynamic pressure calculation.",
            ("Cd",  "above_max"): "Cd above 0.5 — outside training distribution; truck/bus territory.",
            ("Cd",  "below_min"): "Cd below 0.1 — physically unrealistic for a road vehicle.",
            ("LD",  "above_max"): "|Cl/Cd| > 5 — extreme lift-to-drag; verify Cl prediction.",
            ("LD",  "below_min"): "|Cl/Cd| < 0 is unphysical for a magnitude check.",
        }
        msg = messages.get(
            (symbol, direction),
            f"{symbol}={value:.4g} outside allowed [{lo:.4g}, {hi:.4g}]."
        )
        code = f"{symbol}_{direction.upper()}"
        return ViolationRecord(
            symbol=symbol, value=value, lo=lo, hi=hi,
            severity=severity, reason_code=code, message=msg,
        )


# ══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    engine = GuardrailEngine()

    # All-pass: a typical DrivAerNet-range sedan
    ok = engine.check({"Re": 3e6, "Ma": 0.10, "Eu": 0.30, "Cd": 0.28, "LD": 0.5})
    assert ok.passed, f"Expected pass:\n{ok.summary()}"
    assert ok.confidence == 1.0, f"Expected confidence 1.0, got {ok.confidence}"

    # Ma compressibility flag
    r = engine.check({"Ma": 0.45})
    assert "COMPRESSIBILITY_SWITCH" in r.special_flags
    assert r.confidence < 1.0

    # Cd too high
    r = engine.check({"Cd": 0.7})
    assert any(v.symbol == "Cd" for v in r.violations)

    # Cd too low — extreme low drag flag
    r = engine.check({"Cd": 0.08})
    assert any(v.symbol == "Cd" for v in r.violations)
    assert "EXTREME_LOW_DRAG" in r.special_flags

    # Re extrapolation flag
    r = engine.check({"Re": 6e6})
    assert "HIGH_RE_EXTRAPOLATION" in r.special_flags

    # Low confidence triggers discard recommendation
    r = engine.check({"Re": 1e3, "Ma": 0.8, "Cd": 0.9})
    assert r.confidence < engine.CONFIDENCE_THRESHOLD
    assert any("discard" in rec for rec in r.recommendations)

    # Bounds override
    engine.update_bounds(Cd=(0.20, 0.35))
    r = engine.check({"Cd": 0.15})
    assert any(v.symbol == "Cd" for v in r.violations)

    print("All automotive guardrail tests passed.")


if __name__ == "__main__":
    _run_tests()
