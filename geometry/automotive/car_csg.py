"""
geometry/automotive/car_csg.py

CarGeometry dataclass — the structured parameter vector theta for ParaKoop.

theta is the SINGLE object that:
  (a) conditions the parametric Koopman operator  A(theta)
  (b) serves as the output representation of the inverse-design mechanism
      (every suggested design is a CarGeometry instance, guaranteed valid
       by grammar-constrained decoding)

build_car_csg(theta) assembles a full CSG tree from a CarGeometry instance,
mirroring the same pattern used by geometry/vices/variants.py for ALD shapes.

Discrete topology (Catalan part of theta):
    has_spoiler   : bool — rear spoiler present?
    has_diffuser  : bool — underbody diffuser present?
    body_style    : 'fastback' | 'notchback' | 'estateback'
                    (maps to rear_slant_angle range and hull proportions)

Continuous parameters (real-valued part of theta):
    See CarGeometry dataclass fields below — 17 floats covering the four
    SDF primitive groups (CarBody, Spoiler, Diffuser, WheelArch×2 axles).

Usage
-----
    from geometry.automotive.car_csg import CarGeometry, build_car_csg

    theta = CarGeometry()                      # default mid-size sedan
    tree  = build_car_csg(theta)

    # Evaluate SDF at arbitrary points
    import numpy as np
    pts = np.random.uniform(-3, 3, (1000, 3))
    sdf_vals = tree.evaluate(pts)              # negative = inside car body

    # Convert to parameter vector for A(theta) conditioning
    vec = theta.to_vector()                    # shape (17,) float32
    t2  = CarGeometry.from_vector(vec)         # round-trip

    # Describe the discrete topology (Catalan grammar node)
    print(theta.topology_str())
"""

from __future__ import annotations

import dataclasses
import numpy as np
from typing import Literal

from geometry.vices.csg        import CSGTree, CSGNode, CSGLeaf, Op, catalan
from geometry.automotive.primitives import CarBody, Spoiler, Diffuser, WheelArch


# ══════════════════════════════════════════════════════════════════════════
# Theta — structured parameter vector
# ══════════════════════════════════════════════════════════════════════════

BODY_STYLES = ('fastback', 'notchback', 'estateback')

# Default slant-angle ranges per body style (midpoint used as default)
STYLE_SLANT = {
    'fastback':    (20.0, 35.0),
    'notchback':   (60.0, 80.0),
    'estateback':  (10.0, 20.0),
}

# Parameter bounds — used by the inverse-design loop and guardrail checks
PARAM_BOUNDS = {
    'body_length':        (3.5,  5.5),
    'body_width':         (1.5,  2.1),
    'body_height':        (1.1,  1.8),
    'corner_radius':      (0.05, 0.25),
    'rear_slant_angle':   (5.0,  85.0),
    'ground_clearance':   (0.08, 0.25),
    'spoiler_span':       (0.8,  1.9),
    'spoiler_chord':      (0.05, 0.20),
    'spoiler_thickness':  (0.01, 0.05),
    'spoiler_z_offset':   (0.02, 0.15),
    'spoiler_angle':      (0.0,  20.0),
    'diffuser_length':    (0.20, 0.70),
    'diffuser_angle':     (5.0,  20.0),
    'wheel_radius':       (0.26, 0.40),
    'wheel_width':        (0.18, 0.28),
    'front_axle_frac':    (0.15, 0.30),  # fraction of body_length from front
    'rear_axle_frac':     (0.65, 0.85),  # fraction of body_length from front
}


@dataclasses.dataclass
class CarGeometry:
    """
    Structured geometry parameter vector theta for a single car design.

    Discrete (topology) fields — define the Catalan CSG tree structure:
        body_style  : fastback / notchback / estateback
        has_spoiler : rear spoiler present
        has_diffuser: underbody diffuser present

    Continuous fields (17 floats) — real-valued CSG dimensions:
        (ordered to match to_vector() / from_vector() round-trip)
    """

    # ── Discrete topology (Catalan grammar) ───────────────────────────────
    body_style:   Literal['fastback', 'notchback', 'estateback'] = 'fastback'
    has_spoiler:  bool = True
    has_diffuser: bool = True

    # ── CarBody ───────────────────────────────────────────────────────────
    body_length:       float = 4.50
    body_width:        float = 1.80
    body_height:       float = 1.40
    corner_radius:     float = 0.12
    rear_slant_angle:  float = 25.0   # degrees from horizontal
    ground_clearance:  float = 0.15   # metres

    # ── Spoiler ───────────────────────────────────────────────────────────
    spoiler_span:       float = 1.40
    spoiler_chord:      float = 0.10
    spoiler_thickness:  float = 0.025
    spoiler_z_offset:   float = 0.06  # metres above hull top surface
    spoiler_angle:      float = 8.0   # degrees of attack

    # ── Diffuser ──────────────────────────────────────────────────────────
    diffuser_length:  float = 0.40
    diffuser_angle:   float = 12.0   # degrees ramp angle from horizontal

    # ── Wheels (symmetric L/R; two axles) ────────────────────────────────
    wheel_radius:     float = 0.32
    wheel_width:      float = 0.22
    front_axle_frac:  float = 0.22   # fraction of body_length from front
    rear_axle_frac:   float = 0.75   # fraction of body_length from front

    # ── Parameter vector interface ────────────────────────────────────────

    _CONTINUOUS_FIELDS = (
        'body_length', 'body_width', 'body_height', 'corner_radius',
        'rear_slant_angle', 'ground_clearance',
        'spoiler_span', 'spoiler_chord', 'spoiler_thickness',
        'spoiler_z_offset', 'spoiler_angle',
        'diffuser_length', 'diffuser_angle',
        'wheel_radius', 'wheel_width',
        'front_axle_frac', 'rear_axle_frac',
    )  # 17 floats

    def to_vector(self) -> np.ndarray:
        """Return continuous parameters as a float32 vector of shape (17,)."""
        return np.array(
            [getattr(self, f) for f in self._CONTINUOUS_FIELDS],
            dtype=np.float32
        )

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        body_style: str  = 'fastback',
        has_spoiler: bool  = True,
        has_diffuser: bool = True,
    ) -> CarGeometry:
        """Reconstruct from a continuous parameter vector + discrete flags."""
        assert len(vec) == 17, f"Expected 17 params, got {len(vec)}"
        kwargs = {f: float(vec[i]) for i, f in enumerate(cls._CONTINUOUS_FIELDS)}
        return cls(
            body_style=body_style,
            has_spoiler=has_spoiler,
            has_diffuser=has_diffuser,
            **kwargs
        )

    def topology_str(self) -> str:
        """Human-readable description of the discrete CSG topology."""
        parts = [f'CarBody({self.body_style})']
        if self.has_spoiler:
            parts.append('Spoiler')
        if self.has_diffuser:
            parts.append('Diffuser')
        parts += ['WheelArch×4']
        ops = ['Union'] * (len(parts) - 1) + ['Subtract×4']
        return f"CSG[{' + '.join(parts)}]  |  Catalan n={len(parts)}: C={catalan(len(parts)-1)}"

    def is_valid(self) -> tuple[bool, list[str]]:
        """Check all continuous parameters are within PARAM_BOUNDS."""
        violations = []
        for field, (lo, hi) in PARAM_BOUNDS.items():
            val = getattr(self, field)
            if not (lo <= val <= hi):
                violations.append(f"{field}={val:.4g} outside [{lo}, {hi}]")
        # Spoiler span must not exceed car width
        if self.spoiler_span > self.body_width:
            violations.append(
                f"spoiler_span={self.spoiler_span:.3f} > body_width={self.body_width:.3f}"
            )
        return len(violations) == 0, violations


# ══════════════════════════════════════════════════════════════════════════
# CSG assembly
# ══════════════════════════════════════════════════════════════════════════

def build_car_csg(theta: CarGeometry) -> CSGTree:
    """
    Assemble a full CSG tree from a CarGeometry parameter vector.

    Tree structure (depends on discrete flags):
        base = CarBody
        if has_spoiler:  base = Union(base, Spoiler)
        if has_diffuser: base = Union(base, Diffuser)
        base = Subtract(base, WheelArch_FL)
        base = Subtract(base, WheelArch_FR)
        base = Subtract(base, WheelArch_RL)
        base = Subtract(base, WheelArch_RR)

    The four WheelArch subtracts are always present (a car always has wheels).
    """
    L  = theta.body_length
    W  = theta.body_width
    H  = theta.body_height
    gc = theta.ground_clearance
    hull_top_z = gc + H          # z coordinate of hull top surface

    # ── CarBody ────────────────────────────────────────────────────────────
    body = CSGLeaf(CarBody(
        length=L,
        width=W,
        height=H,
        corner_radius=theta.corner_radius,
        rear_slant_angle=theta.rear_slant_angle,
        ground_clearance=gc,
    ))
    root = body

    # ── Spoiler ────────────────────────────────────────────────────────────
    if theta.has_spoiler:
        spoiler_x = L / 2.0 - theta.spoiler_chord / 2.0
        spoiler_z = hull_top_z + theta.spoiler_z_offset + theta.spoiler_thickness / 2.0
        spoiler = CSGLeaf(Spoiler(
            span=theta.spoiler_span,
            chord=theta.spoiler_chord,
            thickness=theta.spoiler_thickness,
            x_pos=spoiler_x,
            z_pos=spoiler_z,
            angle=theta.spoiler_angle,
        ))
        root = CSGNode(Op.UNION, root, spoiler)

    # ── Diffuser ───────────────────────────────────────────────────────────
    if theta.has_diffuser:
        diffuser_max_h = 0.10  # fixed max height; angle controls ramp shape
        diffuser_z     = gc / 2.0
        diffuser = CSGLeaf(Diffuser(
            length=theta.diffuser_length,
            width=W * 0.85,
            max_height=diffuser_max_h,
            x_pos=L / 2.0,
            z_pos=diffuser_z,
            angle=theta.diffuser_angle,
        ))
        root = CSGNode(Op.UNION, root, diffuser)

    # ── WheelArch cutouts (4 wheels, symmetric L/R) ────────────────────────
    axle_z   = gc + theta.wheel_radius       # axle height from z=0
    front_x  = -L / 2.0 + theta.front_axle_frac * L
    rear_x   =  -L / 2.0 + theta.rear_axle_frac * L

    for ax_x in (front_x, rear_x):           # front and rear axles
        arch = CSGLeaf(WheelArch(
            x_pos=ax_x,
            z_pos=axle_z,
            radius=theta.wheel_radius,
            width=theta.wheel_width,
        ))
        root = CSGNode(Op.SUBTRACT, root, arch)

    return CSGTree(root)


# ══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    import numpy as np

    # Default fastback sedan
    theta = CarGeometry()
    tree  = build_car_csg(theta)

    # Sample points in and around a ~4.5 × 1.8 × 1.4 m car
    rng = np.random.default_rng(42)
    pts = rng.uniform([-3, -1.5, -0.1], [3, 1.5, 2.0], size=(2000, 3))
    sdf = tree.evaluate(pts)
    assert sdf.shape == (2000,), "SDF shape mismatch"

    # Centre of the hull should be inside (negative SDF)
    centre = np.array([[0.0, 0.0, 0.85]])   # 0.85 ≈ midheight with gc=0.15
    assert tree.evaluate(centre)[0] < 0, "Hull centre should be inside (sdf < 0)"

    # Far-away point should be outside
    far = np.array([[20.0, 0.0, 0.0]])
    assert tree.evaluate(far)[0] > 0, "Distant point should be outside (sdf > 0)"

    # Parameter vector round-trip
    vec = theta.to_vector()
    assert vec.shape == (17,), f"Expected 17 params, got {vec.shape}"
    theta2 = CarGeometry.from_vector(vec, theta.body_style,
                                      theta.has_spoiler, theta.has_diffuser)
    np.testing.assert_allclose(theta2.to_vector(), vec, rtol=1e-5)

    # Validity check passes for default
    ok, msgs = theta.is_valid()
    assert ok, f"Default theta should be valid: {msgs}"

    # Topology string
    desc = theta.topology_str()
    assert 'CarBody' in desc and 'Catalan' in desc

    # Notchback (no spoiler, no diffuser)
    t_nb = CarGeometry(body_style='notchback', rear_slant_angle=70.0,
                       has_spoiler=False, has_diffuser=False)
    tree_nb = build_car_csg(t_nb)
    assert tree_nb.evaluate(centre)[0] < 0, "Notchback hull centre inside"

    print("All automotive VICES primitive tests passed.")
    print(f"  Default topology: {theta.topology_str()}")
    print(f"  Param vector (17): {vec.round(3)}")


if __name__ == '__main__':
    _run_tests()
