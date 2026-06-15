"""
geometry/automotive/primitives.py

SDF primitives for automotive body geometry.
Follows the same SDFPrimitive interface as geometry/vices/primitives.py —
each class implements sdf(pts) -> np.ndarray[N], negative inside, positive outside.

Coordinate convention (car centred at bounding-box origin):
    x  : car-length axis  (front = -x, rear = +x)
    y  : lateral          (left  = -y, right = +y)
    z  : vertical         (bottom = -z, top  = +z)

Real-world scale: metres.  A typical passenger car sits in ~4.5 × 1.8 × 1.4 m.

Primitives:
    RoundedBox    — smoothly-cornered box (helper, also used for Spoiler/Diffuser)
    HalfSpace     — infinite half-space defined by a plane (helper for slant cuts)
    CarBody       — main hull: rounded box with Ahmed-body rear slant
    Spoiler       — rear wing / lip: angled thin plate
    Diffuser      — underbody rear ramp: wedge via box + half-space
    WheelArch     — wheel-housing cutout: lateral cylinder
"""

import numpy as np
from geometry.vices.primitives import SDFPrimitive


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

class RoundedBox(SDFPrimitive):
    """
    Axis-aligned box with uniformly-rounded edges and corners.
    Standard "SDF rounded box" formulation.

    Parameters
    ----------
    center      : (cx, cy, cz)
    size        : (sx, sy, sz)  full extents
    radius      : corner / edge rounding radius (< min(size)/2)
    """

    def __init__(self, center, size, radius: float = 0.05):
        self.cx, self.cy, self.cz = center
        self.sx, self.sy, self.sz = size
        self.r = max(float(radius), 1e-6)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        r = self.r
        dx = np.abs(pts[:, 0] - self.cx) - (self.sx / 2 - r)
        dy = np.abs(pts[:, 1] - self.cy) - (self.sy / 2 - r)
        dz = np.abs(pts[:, 2] - self.cz) - (self.sz / 2 - r)
        outside = np.sqrt(
            np.maximum(dx, 0) ** 2 +
            np.maximum(dy, 0) ** 2 +
            np.maximum(dz, 0) ** 2
        )
        inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
        return outside + inside - r


class HalfSpace(SDFPrimitive):
    """
    Infinite half-space: sdf < 0 on the 'inside' (opposite to outward normal).

    Parameters
    ----------
    point_on_plane   : any point lying on the dividing plane
    outward_normal   : vector pointing away from the 'inside' region
                       (will be normalised automatically)
    """

    def __init__(self, point_on_plane, outward_normal):
        self.p0 = np.array(point_on_plane, dtype=float)
        n = np.array(outward_normal, dtype=float)
        self.n  = n / np.linalg.norm(n)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.p0) @ self.n


# ══════════════════════════════════════════════════════════════════════════
# Automotive primitives
# ══════════════════════════════════════════════════════════════════════════

class CarBody(SDFPrimitive):
    """
    Main car hull: a RoundedBox intersected with an Ahmed-body-style rear slant.

    The slant is defined by a half-space whose plane passes through the
    top-rear corner of the hull and is tilted at `rear_slant_angle` from
    horizontal.  This parameterises the key aerodynamic topology distinction
    between fastback (~25-35°), notchback (no effective cut ≈ 0° or 90°)
    and estateback.

    Parameters
    ----------
    length            : car body length (x)
    width             : car body width  (y)
    height            : car body height (z)
    corner_radius     : hull edge rounding
    rear_slant_angle  : Ahmed slant angle in degrees from horizontal
                        0°  → no slant effect (flat roof to rear face)
                        25° → classic Ahmed body / fastback
                        90° → vertical rear face (no slant effect)
    ground_clearance  : z-offset so the hull bottom sits above z = 0
                        (the hull centre is shifted up by height/2 + gc)
    """

    def __init__(
        self,
        length: float          = 4.50,
        width: float           = 1.80,
        height: float          = 1.40,
        corner_radius: float   = 0.12,
        rear_slant_angle: float = 25.0,
        ground_clearance: float = 0.15,
    ):
        self.length          = length
        self.width           = width
        self.height          = height
        self.corner_radius   = corner_radius
        self.rear_slant_angle = rear_slant_angle
        self.ground_clearance = ground_clearance

        # Hull centre (x=0 lateral/longitudinal midpoint, z lifted by gc)
        cz = ground_clearance + height / 2.0
        self._box = RoundedBox(
            center=(0.0, 0.0, cz),
            size=(length, width, height),
            radius=corner_radius,
        )

        # Slant half-space:
        #   normal direction (outward, away from car body):
        #     n = (sin α, 0, cos α)  where α = rear_slant_angle
        #   plane passes through top-rear corner of the hull:
        #     p0 = (L/2, 0, cz + H/2)
        alpha = np.radians(rear_slant_angle)
        p0 = np.array([length / 2.0, 0.0, cz + height / 2.0])
        n_out = np.array([np.sin(alpha), 0.0, np.cos(alpha)])
        self._slant = HalfSpace(p0, n_out)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        # Intersection = max(sdf_box, sdf_slant):
        #   keeps only points inside BOTH the rounded box AND below the slant plane
        return np.maximum(self._box.sdf(pts), self._slant.sdf(pts))


class Spoiler(SDFPrimitive):
    """
    Rear wing / lip spoiler: a thin rounded plate, optionally angled.

    Modelled as a RoundedBox rotated about the y-axis by `angle` degrees
    (positive angle tilts the leading edge downward — increasing downforce).

    Parameters
    ----------
    span         : wing span (y)
    chord        : wing chord length (x)
    thickness    : wing cross-section thickness (z)
    x_pos        : x position of wing centre (usually near rear: ~L/2 - chord/2)
    z_pos        : z position of wing centre (above trunk lid)
    angle        : angle of attack in degrees (0 = flat, +ve = nose-down)
    corner_radius: edge rounding
    """

    def __init__(
        self,
        span: float           = 1.40,
        chord: float          = 0.10,
        thickness: float      = 0.025,
        x_pos: float          = 2.05,   # ~rear of a 4.5m car
        z_pos: float          = 1.55,   # above trunk (~hull top + small offset)
        angle: float          = 8.0,
        corner_radius: float  = 0.008,
    ):
        self.span          = span
        self.chord         = chord
        self.thickness     = thickness
        self.x_pos         = x_pos
        self.z_pos         = z_pos
        self.angle         = angle
        self.corner_radius = corner_radius
        self._alpha        = np.radians(angle)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        # Rotate points about y-axis (inverse rotation to bring points into
        # the spoiler's local frame where it is axis-aligned)
        a = -self._alpha   # inverse rotation
        cos_a, sin_a = np.cos(a), np.sin(a)

        dx = pts[:, 0] - self.x_pos
        dz = pts[:, 2] - self.z_pos

        # Rotate x and z, keep y unchanged
        lx = cos_a * dx - sin_a * dz
        ly = pts[:, 1]
        lz = sin_a * dx + cos_a * dz

        local = np.stack([lx, ly, lz], axis=1)

        # Evaluate axis-aligned rounded box in local frame (centred at origin)
        box = RoundedBox(
            center=(0.0, 0.0, 0.0),
            size=(self.chord, self.span, self.thickness),
            radius=self.corner_radius,
        )
        return box.sdf(local)


class Diffuser(SDFPrimitive):
    """
    Rear underbody diffuser: a wedge-shaped ramp that accelerates underbody
    flow and reduces base pressure drag.

    Implemented as a RoundedBox (the diffuser volume) intersected with a
    HalfSpace (the upward-angled ramp face), producing a wedge.

    Parameters
    ----------
    length           : diffuser length (x)
    width            : diffuser width  (y, typically close to car width)
    max_height       : maximum height of diffuser at its exit (z extent)
    x_pos            : x position of diffuser rear face centre (~car rear)
    z_pos            : z position of diffuser bottom face centre
    angle            : ramp angle from horizontal in degrees
                       (larger = steeper ramp = more aggressive diffuser)
    corner_radius    : edge rounding
    """

    def __init__(
        self,
        length: float          = 0.40,
        width: float           = 1.50,
        max_height: float      = 0.12,
        x_pos: float           = 2.05,
        z_pos: float           = 0.06,
        angle: float           = 12.0,
        corner_radius: float   = 0.015,
    ):
        self.length        = length
        self.width         = width
        self.max_height    = max_height
        self.x_pos         = x_pos
        self.z_pos         = z_pos
        self.angle         = angle
        self.corner_radius = corner_radius

        self._box = RoundedBox(
            center=(x_pos - length / 2.0, 0.0, z_pos + max_height / 2.0),
            size=(length, width, max_height),
            radius=corner_radius,
        )

        # Ramp plane: passes through front-bottom corner of diffuser box,
        # tilted at `angle` from horizontal. Outward normal points upward-forward.
        alpha = np.radians(angle)
        p0 = np.array([x_pos - length, 0.0, z_pos])
        n_out = np.array([-np.sin(alpha), 0.0, -np.cos(alpha)])
        self._ramp = HalfSpace(p0, n_out)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        # Keep only what is inside the box AND above the ramp (inside the wedge)
        return np.maximum(self._box.sdf(pts), self._ramp.sdf(pts))


class WheelArch(SDFPrimitive):
    """
    Wheel-arch cutout: a cylinder oriented along the y-axis (lateral),
    representing the wheel housing volume.

    Typically used as a SUBTRACT node in the CSG tree to cut wheel arches
    into the car body.

    Parameters
    ----------
    x_pos    : longitudinal centre of the wheel (front or rear axle x)
    z_pos    : vertical centre of the wheel (axle height from ground)
    radius   : outer wheel/tyre radius
    width    : tyre width (y extent of the cylinder)
    """

    def __init__(
        self,
        x_pos: float  = 1.05,
        z_pos: float  = 0.32,
        radius: float = 0.32,
        width: float  = 0.22,
    ):
        self.x_pos  = x_pos
        self.z_pos  = z_pos
        self.radius = radius
        self.width  = width

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        # Cylinder along y-axis: radial distance in xz-plane, bounded by width in y
        dx = pts[:, 0] - self.x_pos
        dz = pts[:, 2] - self.z_pos
        r_dist = np.sqrt(dx ** 2 + dz ** 2) - self.radius
        y_dist = np.abs(pts[:, 1]) - self.width / 2.0
        return np.maximum(r_dist, y_dist)
