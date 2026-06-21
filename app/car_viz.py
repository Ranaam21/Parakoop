"""
app/car_viz.py

Parametric 2D car side-profile visualisation using Plotly.
All dimensions in mm; blueprint dark aesthetic (#0d1117 background).
"""

from __future__ import annotations
import base64
import os as _os
import numpy as np
import plotly.graph_objects as go

# ── GT-R reference image (embedded once at import time) ───────────────────────
_IMG_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                           'data', 'Car Sample Image.jpg')
with open(_IMG_PATH, 'rb') as _f:
    _CAR_IMG_B64 = 'data:image/jpeg;base64,' + base64.b64encode(_f.read()).decode()

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = '#0d1117'
BLUE   = '#1E88E5'
AMBER  = '#FF8F00'
GREEN  = '#43A047'
RED    = '#E53935'
GREY   = '#6e7681'
WHITE  = '#e6edf3'
PANEL  = '#21262d'


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40):
    """Return (x, y) arrays for an arc from a0 to a1 degrees."""
    angles = np.linspace(np.radians(a0), np.radians(a1), n)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def _circle(cx: float, cy: float, r: float, n: int = 60):
    a = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def side_profile_coords(params: dict) -> tuple[list, list]:
    """
    Build (x_mm, y_mm) polygon for car side profile — GT-R sports-coupe style.

    No flat sill at front or rear: bumpers taper to near-ground so the car
    sits flush with the ground line and no "platform extension" appears.

    Uses keys: ref_length_mm, height_mm, rear_slant_deg, cabin_frac, style.
    """
    L     = float(params['ref_length_mm'])
    H     = float(params['height_mm'])
    slant = float(params['rear_slant_deg'])
    cabin = float(params['cabin_frac'])
    style = str(params['style'])

    gc      = max(70.0, 0.055 * H)   # sports-car ground clearance
    gc_lip  = max(8.0,  0.008 * H)   # near-ground bumper/tail height
    hood_h  = gc + 0.22 * H          # hood height at windscreen base

    # Key x positions — GT-R proportions
    x_ws    = 0.30 * L               # windshield base (end of long hood)
    x_a     = 0.42 * L               # A-pillar top / roof start
    x_c     = min(x_a + cabin * 0.52 * L, 0.81 * L)  # C-pillar
    x_rear  = 0.91 * L               # end of rear slope
    x_end   = L

    rear_drop = np.tan(np.radians(max(slant, 0.1))) * (x_rear - x_c)
    rear_h    = float(np.clip(H - rear_drop, gc + 0.15 * H, H))

    # Front face — swept-back nose (5 points, slopes backward like GT-R bumper)
    front_xs = [0,       0.03*L,  0.07*L,  0.10*L,  0.14*L ]
    front_ys = [gc_lip,  gc_lip,  0.14*H,  0.20*H,  hood_h*0.94]

    # Hood — gentle rise from bumper top to windscreen base (nearly flat, sporty)
    hood_xs  = [0.20*L,  0.25*L,  x_ws  ]
    hood_ys  = [hood_h*0.96, hood_h*0.98, hood_h]

    if style == 'notchback':
        trunk_h    = gc + 0.42 * H
        slant_norm = float(np.clip(slant / 21.4, 0.0, 1.0))
        window_run = (0.04 + (1.0 - slant_norm) * 0.11) * L
        x_tc       = x_c + window_run            # top of rear window / trunk start
        x_te       = min(x_tc + 0.10 * L, 0.90 * L)  # trunk lid end / rear face top
        xs = (front_xs + hood_xs +
              [x_a, x_c, x_tc, x_te, x_te, x_end, 0])
        ys = (front_ys + hood_ys +
              [H, H, trunk_h, trunk_h, gc_lip, gc_lip, gc_lip])
    else:
        # Fastback / estateback — rear slope + small trunk deck + steep rear face
        x_trunk = x_rear - 0.04 * L          # trunk deck leading edge
        trunk_top = rear_h + 0.02 * H        # slight spoiler uptick
        xs = (front_xs + hood_xs +
              [x_a, x_c, x_trunk, x_rear, x_end, 0])
        ys = (front_ys + hood_ys +
              [H, H, trunk_top, rear_h, gc_lip, gc_lip])

    return xs, ys


# ── Core draw functions ───────────────────────────────────────────────────────

def draw_car_side(
    params: dict,
    color: str = BLUE,
    fill_opacity: float = 0.18,
    name: str = '',
    show_dims: bool = True,
) -> go.Figure:
    """
    Plotly figure: car side profile with dimension annotations.
    Blueprint dark aesthetic.
    """
    L  = float(params['ref_length_mm'])
    H  = float(params['height_mm'])
    gc = max(70.0, 0.055 * H)   # must match side_profile_coords

    r, g, b  = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fill_col = f'rgba({r},{g},{b},{fill_opacity})'

    xs, ys = side_profile_coords(params)

    wheel_r = 0.115 * H
    arch_r  = wheel_r * 1.18
    fw_x    = 0.22 * L
    rw_x    = 0.77 * L
    w_cy    = gc

    fig = go.Figure()

    # ── Car body ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill='toself', fillcolor=fill_col,
        line=dict(color=color, width=2.5),
        mode='lines', name=name or 'Body', hoverinfo='skip',
    ))

    # ── Wheel arch cutouts (bg-coloured fill over body) ───────────────────
    for wx in (fw_x, rw_x):
        ax, ay = _arc(wx, w_cy, arch_r, 0, 180)
        ax = np.append(ax, [wx + arch_r, wx - arch_r])
        ay = np.append(ay, [w_cy, w_cy])
        fig.add_trace(go.Scatter(
            x=ax, y=ay,
            fill='toself', fillcolor=BG,
            line=dict(color=color, width=1.5),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ── Wheels ───────────────────────────────────────────────────────────────
    for wx in (fw_x, rw_x):
        cx, cy = _circle(wx, w_cy, wheel_r)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, fill='toself', fillcolor='#2d333b',
            line=dict(color=GREY, width=1.5),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))
        hx, hy = _circle(wx, w_cy, wheel_r * 0.30)
        fig.add_trace(go.Scatter(
            x=hx, y=hy, fill='toself', fillcolor=GREY,
            line=dict(color=WHITE, width=0.8),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ── Ground shadow ─────────────────────────────────────────────────────────
    fig.add_hrect(y0=-wheel_r * 0.5, y1=0,
                  fillcolor='rgba(110,118,129,0.08)', line_width=0)
    fig.add_hline(y=0, line=dict(color=GREY, width=0.8, dash='dot'))

    # ── Dimension annotations ─────────────────────────────────────────────────
    if show_dims:
        slant   = params['rear_slant_deg']
        cabin   = params['cabin_frac']
        x_c     = 0.30 * L + cabin * 0.62 * L
        x_c     = min(x_c, 0.83 * L)
        mid_slant_x = (x_c + 0.93 * L) / 2
        mid_slant_y = H * 0.84

        # Height double-arrow
        fig.add_annotation(
            x=L * 1.07, y=H, ax=L * 1.07, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            text=f"H = {H:.0f} mm",
            showarrow=True, arrowhead=2,
            arrowcolor=AMBER, arrowwidth=1.5, arrowsize=1,
            font=dict(color=AMBER, size=11),
        )
        # Slant angle label
        fig.add_annotation(
            x=mid_slant_x, y=mid_slant_y,
            text=f"α = {slant:.1f}°",
            showarrow=False,
            font=dict(color=AMBER, size=11),
            bgcolor='rgba(13,17,23,0.7)',
        )
        # Length
        fig.add_annotation(
            x=L / 2, y=-wheel_r * 0.9,
            text=f"L = {L:.0f} mm",
            showarrow=False,
            font=dict(color=GREY, size=10),
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-0.06 * L, L * 1.18]),
        yaxis=dict(
            visible=False, scaleanchor='x', scaleratio=1,
            range=[-wheel_r * 1.4, H * 1.30],
        ),
        margin=dict(l=5, r=5, t=5, b=5),
        height=220,
        showlegend=False,
    )
    return fig


_BEFORE_CLR = '#26C6DA'   # cyan — used for "before" ghost in all comparison views


def draw_car_comparison(
    params_before: dict,
    params_after: dict,
    delta: dict,
    height: int = 300,
) -> go.Figure:
    """
    Side-profile overlay: cyan dashed ghost (before) + blue solid (after).
    Annotates each changed dimension with Δmm / Δ° arrows.
    """
    L       = float(params_after['ref_length_mm'])
    H_b     = float(params_before['height_mm'])
    H_a     = float(params_after['height_mm'])
    H_max   = max(H_b, H_a)
    gc_a    = max(70.0, 0.055 * H_a)
    wheel_r = 0.115 * H_max

    xs_b, ys_b = side_profile_coords(params_before)
    xs_a, ys_a = side_profile_coords(params_after)

    fig = go.Figure()

    # Ground / road
    fig.add_hrect(y0=-wheel_r * 0.5, y1=0,
                  fillcolor='rgba(110,118,129,0.06)', line_width=0)
    fig.add_hline(y=0, line=dict(color=GREY, width=0.8, dash='dot'))

    # Ghost before — cyan dashed
    fig.add_trace(go.Scatter(
        x=xs_b, y=ys_b,
        fill='toself', fillcolor='rgba(38,198,218,0.07)',
        line=dict(color=_BEFORE_CLR, width=2.2, dash='dash'),
        mode='lines', name='Before', hoverinfo='skip',
    ))
    # Solid after — blue
    fig.add_trace(go.Scatter(
        x=xs_a, y=ys_a,
        fill='toself', fillcolor='rgba(30,136,229,0.18)',
        line=dict(color=BLUE, width=2.5),
        mode='lines', name='After (suggested)', hoverinfo='skip',
    ))

    # Wheels (after)
    for wx in (0.22 * L, 0.77 * L):
        cx, cy = _circle(wx, gc_a, wheel_r)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, fill='toself', fillcolor='#2d333b',
            line=dict(color=GREY, width=1.2),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ── Delta annotations ─────────────────────────────────────────────────────
    def _arrow(x, y, ax, ay, col, txt):
        fig.add_annotation(
            x=x, y=y, ax=ax, ay=ay,
            xref='x', yref='y', axref='x', ayref='y',
            text=txt, showarrow=True,
            arrowhead=3, arrowcolor=col, arrowwidth=2.5, arrowsize=0.9,
            font=dict(color=col, size=11), bgcolor='rgba(13,17,23,0.88)',
        )

    def _label(x, y, txt, col, angle=0):
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, textangle=angle,
                           font=dict(color=col, size=11),
                           bgcolor='rgba(13,17,23,0.88)')

    if 'height_mm' in delta:
        _, _, d = delta['height_mm']
        col, sign = (GREEN, '') if d < 0 else (AMBER, '+')
        x_arr = L * 1.04
        # dashed old-H line
        fig.add_shape(type='line', x0=x_arr, y0=0, x1=x_arr, y1=H_b,
                      line=dict(color=col, width=2, dash='dashdot'))
        fig.add_shape(type='line', x0=x_arr - L*0.007, y0=H_b,
                      x1=x_arr + L*0.007, y1=H_b,
                      line=dict(color=col, width=2))
        _arrow(x_arr, H_a, x_arr, H_b, col, '')
        _label(x_arr + L*0.045, (H_a + H_b) / 2,
               f"ΔH {sign}{d:.0f} mm", col, -90)

    if 'rear_slant_deg' in delta:
        _, _, d = delta['rear_slant_deg']
        col, sign = (GREEN, '') if d < 0 else (AMBER, '+')
        cabin = params_after['cabin_frac']
        x_c   = min(0.30 * L + cabin * 0.52 * L, 0.81 * L)
        _label((x_c + 0.92 * L) / 2, H_max * 0.78,
               f"Δα {sign}{d:.1f}°", col)

    if 'width_mm' in delta:
        _, _, d = delta['width_mm']
        col, sign = (GREEN, '') if d < 0 else (AMBER, '+')
        _label(L * 0.5, H_max * 1.10, f"ΔW {sign}{d:.0f} mm", col)

    if 'cabin_frac' in delta:
        _, _, d = delta['cabin_frac']
        col, sign = (GREEN, '') if d < 0 else (AMBER, '+')
        _label(L * 0.62, H_max * 0.92, f"Δcabin {sign}{d:.3f}", col)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-0.05 * L, L * 1.16]),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1,
                   range=[-wheel_r * 1.2, H_max * 1.30]),
        margin=dict(l=5, r=5, t=5, b=5),
        height=height,
        showlegend=False,
    )
    return fig


def draw_front_comparison(
    params_before: dict,
    params_after: dict,
    delta: dict,
    height: int = 300,
) -> go.Figure:
    """
    Front-face view: cyan dashed (before) vs blue solid (after).
    Shows W and H changes clearly on a flat 2D face.
    """
    W_a = float(params_after['width_mm'])
    W_b = float(params_before['width_mm'])
    H_a = float(params_after['height_mm'])
    H_b = float(params_before['height_mm'])
    W_max = max(W_a, W_b)
    H_max = max(H_a, H_b)
    wheel_r = 0.115 * H_a
    gc      = max(70.0, 0.055 * H_a)

    fig = go.Figure()

    # Road
    fig.add_hrect(y0=-wheel_r * 0.4, y1=0,
                  fillcolor='rgba(110,118,129,0.06)', line_width=0)
    fig.add_hline(y=0, line=dict(color=GREY, width=0.8, dash='dot'))

    # Before rectangle — cyan dashed
    fig.add_shape(type='rect', x0=-W_b/2, y0=0, x1=W_b/2, y1=H_b,
                  line=dict(color=_BEFORE_CLR, width=2.2, dash='dash'),
                  fillcolor='rgba(38,198,218,0.06)')

    # After rectangle — blue solid
    fig.add_shape(type='rect', x0=-W_a/2, y0=0, x1=W_a/2, y1=H_a,
                  line=dict(color=BLUE, width=2.5),
                  fillcolor='rgba(30,136,229,0.12)')

    # Windshield area (after) — darker inset
    ws_w = W_a * 0.72;  ws_h = H_a * 0.28;  ws_y = H_a * 0.56
    fig.add_shape(type='rect', x0=-ws_w/2, y0=ws_y, x1=ws_w/2, y1=ws_y + ws_h,
                  line=dict(color='#1565C0', width=1),
                  fillcolor='rgba(13,17,23,0.55)')

    # Wheels (after) — two circles at the bottom
    for wx in (-W_a * 0.28, W_a * 0.28):
        cx, cy = _circle(wx, gc, wheel_r)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, fill='toself', fillcolor='#2d333b',
            line=dict(color=GREY, width=1.2),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ΔW: horizontal bracket at the top
    if 'width_mm' in delta:
        _, _, d = delta['width_mm']
        col, sign = (GREEN, '') if d < 0 else (AMBER, '+')
        y_br = H_max * 1.06
        # show old W as dashed horizontal line
        fig.add_shape(type='line', x0=-W_b/2, y0=y_br, x1=W_b/2, y1=y_br,
                      line=dict(color=col, width=2, dash='dashdot'))
        # arrow to new W edge
        fig.add_annotation(
            x=W_a/2, y=y_br, ax=W_b/2, ay=y_br,
            xref='x', yref='y', axref='x', ayref='y',
            text='', showarrow=True,
            arrowhead=3, arrowcolor=col, arrowwidth=2.5, arrowsize=0.9,
        )
        fig.add_annotation(x=0, y=y_br + H_max * 0.06,
                           text=f"ΔW {sign}{d:.0f} mm", showarrow=False,
                           font=dict(color=col, size=11),
                           bgcolor='rgba(13,17,23,0.88)')

    # ΔH: vertical bracket on the right
    if 'height_mm' in delta:
        _, _, d = delta['height_mm']
        col, sign = (GREEN, '') if d < 0 else (AMBER, '+')
        x_br = W_max * 0.58
        # old H dashed line
        fig.add_shape(type='line', x0=x_br, y0=0, x1=x_br, y1=H_b,
                      line=dict(color=col, width=2, dash='dashdot'))
        fig.add_shape(type='line', x0=x_br - W_max*0.025, y0=H_b,
                      x1=x_br + W_max*0.025, y1=H_b,
                      line=dict(color=col, width=2))
        fig.add_annotation(
            x=x_br, y=H_a, ax=x_br, ay=H_b,
            xref='x', yref='y', axref='x', ayref='y',
            text='', showarrow=True,
            arrowhead=3, arrowcolor=col, arrowwidth=2.5, arrowsize=0.9,
        )
        fig.add_annotation(x=x_br + W_max * 0.07, y=(H_a + H_b) / 2,
                           text=f"ΔH {sign}{d:.0f} mm", showarrow=False, textangle=-90,
                           font=dict(color=col, size=11),
                           bgcolor='rgba(13,17,23,0.88)')

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-W_max * 0.72, W_max * 0.80]),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1,
                   range=[-wheel_r * 1.2, H_max * 1.28]),
        margin=dict(l=5, r=5, t=5, b=5),
        height=height, showlegend=False,
    )
    return fig


def draw_car_iso(params: dict) -> go.Figure:
    """
    3-quarter isometric view: near-side body + top face + far-side ghost + both wheel pairs.
    Width (W) is shown as a diagonal dimension line across the roof.
    """
    L     = float(params['ref_length_mm'])
    H     = float(params['height_mm'])
    W     = float(params['width_mm'])
    slant = float(params.get('rear_slant_deg', 15))
    cabin = float(params.get('cabin_frac', 0.5))
    style = str(params.get('style', 'fastback'))

    gc      = max(70.0, 0.055 * H)
    gc_lip  = max(8.0,  0.008 * H)
    hood_h  = gc + 0.22 * H
    wheel_r = 0.115 * H
    arch_r  = wheel_r * 1.18
    fw_x    = 0.22 * L
    rw_x    = 0.77 * L
    w_cy    = gc

    x_ws   = 0.30 * L
    x_a    = 0.42 * L
    x_c    = min(x_a + cabin * 0.52 * L, 0.81 * L)
    x_rear = 0.91 * L
    x_end  = L

    rear_drop = np.tan(np.radians(max(slant, 0.1))) * (x_rear - x_c)
    rear_h    = float(np.clip(H - rear_drop, gc + 0.15 * H, H))

    if style == 'notchback':
        trunk_h    = gc + 0.42 * H
        slant_norm = float(np.clip(slant / 21.4, 0.0, 1.0))
        window_run = (0.04 + (1.0 - slant_norm) * 0.11) * L
        x_tc = x_c + window_run
        x_te = min(x_tc + 0.10 * L, 0.90 * L)
        top_nx = [0.14*L, 0.20*L, 0.25*L, x_ws, x_a, x_c, x_tc, x_te, x_end]
        top_ny = [hood_h*0.94, hood_h*0.96, hood_h*0.98,
                  hood_h, H, H, trunk_h, trunk_h, gc_lip]
    else:
        x_trunk   = x_rear - 0.04 * L
        trunk_top = rear_h + 0.02 * H
        top_nx = [0.14*L, 0.20*L, 0.25*L, x_ws, x_a, x_c, x_trunk, x_rear, x_end]
        top_ny = [hood_h*0.94, hood_h*0.96, hood_h*0.98,
                  hood_h, H, H, trunk_top, rear_h, gc_lip]

    xs, ys = side_profile_coords(params)

    # Isometric depth: width W projects upper-right at 30° (cabinet projection, 0.28 scale)
    scale = 0.28
    dx = W * scale * np.cos(np.radians(30))
    dy = W * scale * np.sin(np.radians(30))

    xs_f = [x + dx for x in xs]
    ys_f = [y + dy for y in ys]

    fig = go.Figure()

    # ── Road band (wider to cover both near and far footprints) ───────────────
    fig.add_shape(type='rect',
                  x0=-L*0.04, y0=-H*0.07, x1=x_end + dx + L*0.04, y1=0,
                  fillcolor='#161b22', line=dict(width=0))
    fig.add_shape(type='line',
                  x0=-L*0.04, y0=0, x1=x_end + dx + L*0.04, y1=0,
                  line=dict(color=GREY, width=1.0, dash='dot'))

    # ── Far-side body (ghost, drawn first so near side covers it) ─────────────
    fig.add_trace(go.Scatter(
        x=xs_f, y=ys_f,
        fill='toself', fillcolor='rgba(13,17,23,0.55)',
        line=dict(color='#3a4a60', width=1.4),
        mode='lines', hoverinfo='skip', showlegend=False,
    ))

    # ── Top surface (upper face of the car body — reveals width) ──────────────
    top_fx = [x + dx for x in reversed(top_nx)]
    top_fy = [y + dy for y in reversed(top_ny)]
    fig.add_trace(go.Scatter(
        x=top_nx + top_fx,
        y=top_ny + top_fy,
        fill='toself', fillcolor='rgba(21,101,192,0.32)',
        line=dict(color='#1565C0', width=1.3),
        mode='lines', hoverinfo='skip', showlegend=False,
    ))

    # ── Near-side body (main face) ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill='toself', fillcolor='rgba(30,136,229,0.18)',
        line=dict(color=BLUE, width=2.5),
        mode='lines', name='Body', hoverinfo='skip',
    ))

    # ── Near-side wheel arch cutouts ──────────────────────────────────────────
    for wx in (fw_x, rw_x):
        ax, ay = _arc(wx, w_cy, arch_r, 0, 180)
        ax = np.append(ax, [wx + arch_r, wx - arch_r])
        ay = np.append(ay, [w_cy, w_cy])
        fig.add_trace(go.Scatter(
            x=ax, y=ay, fill='toself', fillcolor=BG,
            line=dict(color=BLUE, width=1.5),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ── Near-side wheels ──────────────────────────────────────────────────────
    for wx in (fw_x, rw_x):
        cx, cy = _circle(wx, w_cy, wheel_r)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, fill='toself', fillcolor='#2d333b',
            line=dict(color=GREY, width=1.5),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))
        hx, hy = _circle(wx, w_cy, wheel_r * 0.30)
        fig.add_trace(go.Scatter(
            x=hx, y=hy, fill='toself', fillcolor=GREY,
            line=dict(color=WHITE, width=0.8),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ── Far-side wheels (simplified, behind body) ─────────────────────────────
    for wx in (fw_x, rw_x):
        cx, cy = _circle(wx + dx, w_cy + dy, wheel_r)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, fill='toself', fillcolor='rgba(30,35,44,0.7)',
            line=dict(color='#3a4a60', width=1.1),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # ── Structural depth edges (A-pillar, C-pillar, rear corner) ─────────────
    for px, py in [(x_a, H), (x_c, H), (x_end, gc_lip)]:
        fig.add_shape(type='line',
                      x0=px, y0=py, x1=px + dx, y1=py + dy,
                      line=dict(color='#1565C0', width=1.0, dash='dot'))

    # ── L dimension (horizontal below car, near-side only) ────────────────────
    y_arr = -H * 0.06
    fig.add_shape(type='line', x0=0, y0=y_arr, x1=L, y1=y_arr,
                  line=dict(color=AMBER, width=1.2))
    for tx in (0, L):
        fig.add_shape(type='line',
                      x0=tx, y0=y_arr - H*0.015, x1=tx, y1=y_arr + H*0.015,
                      line=dict(color=AMBER, width=1.2))
    fig.add_annotation(x=L/2, y=y_arr - H*0.045,
                       text=f"L = {L:.0f} mm", showarrow=False,
                       font=dict(color=AMBER, size=10),
                       bgcolor='rgba(13,17,23,0.85)')

    # ── H dimension (vertical, just right of near-side body) ──────────────────
    x_arr = L + L * 0.025
    fig.add_shape(type='line', x0=x_arr, y0=0, x1=x_arr, y1=H,
                  line=dict(color=AMBER, width=1.2))
    for ty in (0, H):
        fig.add_shape(type='line',
                      x0=x_arr - L*0.008, y0=ty, x1=x_arr + L*0.008, y1=ty,
                      line=dict(color=AMBER, width=1.2))
    fig.add_annotation(x=x_arr + L*0.034, y=H/2,
                       text=f"H = {H:.0f} mm", showarrow=False, textangle=-90,
                       font=dict(color=AMBER, size=10),
                       bgcolor='rgba(13,17,23,0.85)')

    # ── W dimension (diagonal across the roof, near A-pillar → far A-pillar) ──
    fig.add_shape(type='line',
                  x0=x_a, y0=H, x1=x_a + dx, y1=H + dy,
                  line=dict(color=AMBER, width=1.2))
    perp = np.radians(120)
    tk   = H * 0.015
    for ex, ey in [(x_a, H), (x_a + dx, H + dy)]:
        fig.add_shape(type='line',
                      x0=ex + tk*np.cos(perp), y0=ey + tk*np.sin(perp),
                      x1=ex - tk*np.cos(perp), y1=ey - tk*np.sin(perp),
                      line=dict(color=AMBER, width=1.2))
    fig.add_annotation(x=x_a + dx*0.5, y=H + dy*0.5 + H*0.065,
                       text=f"W = {W:.0f} mm", showarrow=False,
                       font=dict(color=AMBER, size=10),
                       bgcolor='rgba(13,17,23,0.85)')

    # ── Invisible hover markers — define dimension terms on cursor hover ──────
    for hx, hy, htip in [
        (L / 2,
         y_arr - H*0.048,
         "L — Reference length (mm)<br>Front bumper to rear tail of the car"),
        (x_arr + L*0.040,
         H / 2,
         "H — Overall height (mm)<br>Ground plane to rooftop"),
        (x_a + dx*0.5,
         H + dy*0.5 + H*0.070,
         "W — Body width (mm)<br>Maximum cross-section width of the car<br>"
         "Shown as the diagonal dimension across the roof"),
    ]:
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy], mode='markers',
            marker=dict(size=26, color='rgba(0,0,0,0)'),
            hovertemplate=htip + '<extra></extra>',
            showlegend=False, name='',
        ))

    # Both axes fixed to the full slider envelope so neither L, H, nor W changes
    # make the car appear to shrink/grow in the other dimension.
    # x covers: L_max=5500 + H-annotation text (~5500*0.06) + dx_max(W=1.8*H=1753)≈765 + margin
    # y covers: H_max=1753 + dy_max≈266 + W-label + L-label below zero
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-300, 7200]),
        yaxis=dict(visible=False, range=[-380, 2750]),
        margin=dict(l=5, r=10, t=5, b=5),
        height=250,
        showlegend=False,
    )
    return fig


def draw_car_iso_comparison(
    before: dict,
    after: dict,
    delta: dict,
    height: int = 300,
) -> go.Figure:
    """
    3D isometric view: solid blue = after design, grey ghost = before.
    Changed dimensions get a bright dashed old-value line + arrow + Δ label.
    Axes are tight around the actual car so the car fills the figure.
    """
    L   = float(after['ref_length_mm'])
    H_a = float(after['height_mm'])
    H_b = float(before['height_mm'])
    W_a = float(after['width_mm'])
    W_b = float(before['width_mm'])
    x_a = 0.42 * L
    scale = 0.28
    cos30 = np.cos(np.radians(30))
    sin30 = np.sin(np.radians(30))
    dx_a  = W_a * scale * cos30
    dy_a  = W_a * scale * sin30
    dx_b  = W_b * scale * cos30
    dy_b  = W_b * scale * sin30

    H_max  = max(H_a, H_b)
    dy_max = max(dy_a, dy_b)

    # ── Ghost (before) — cyan dashed, renders behind the after-car ───────────
    _BEFORE_CLR = '#26C6DA'   # cyan — clearly distinct from after-car blue
    xs_b, ys_b = side_profile_coords(before)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_b, y=ys_b,
        fill='toself', fillcolor='rgba(38,198,218,0.07)',
        line=dict(color=_BEFORE_CLR, width=2.2, dash='dash'),
        mode='lines', name='Before', hoverinfo='skip',
    ))

    # ── After car traces + base layout (fixed range overridden below) ─────────
    after_fig = draw_car_iso(after)
    for trace in after_fig.data:
        fig.add_trace(trace)
    fig.update_layout(after_fig.layout)

    # ── ΔH: bold dashed old-H line + arrow showing change direction ──────────
    if 'height_mm' in delta:
        _, _, d = delta['height_mm']
        col  = GREEN if d < 0 else AMBER
        sign = '+' if d > 0 else ''
        x_arr = L + L * 0.030

        # dashed line at old H — thick, coloured
        fig.add_shape(type='line', x0=x_arr, y0=0, x1=x_arr, y1=H_b,
                      line=dict(color=col, width=2.5, dash='dashdot'))
        # end-tick at old H
        tk = L * 0.009
        fig.add_shape(type='line',
                      x0=x_arr - tk, y0=H_b, x1=x_arr + tk, y1=H_b,
                      line=dict(color=col, width=2.5))
        # arrow from old H level to new H level
        fig.add_annotation(
            x=x_arr, y=H_a,
            ax=x_arr, ay=H_b,
            xref='x', yref='y', axref='x', ayref='y',
            text='', showarrow=True,
            arrowhead=3, arrowcolor=col, arrowwidth=2.5, arrowsize=1.0,
        )
        # Δ label beside the arrow
        fig.add_annotation(
            x=x_arr + L * 0.045, y=(H_a + H_b) / 2,
            text=f"ΔH {sign}{d:.0f} mm", showarrow=False, textangle=-90,
            font=dict(color=col, size=11, family='monospace'),
            bgcolor='rgba(13,17,23,0.90)',
        )

    # ── ΔW: bold dashed old-W diagonal + arrow ───────────────────────────────
    if 'width_mm' in delta:
        _, _, d = delta['width_mm']
        col  = GREEN if d < 0 else AMBER
        sign = '+' if d > 0 else ''

        # dashed line at old W
        fig.add_shape(type='line',
                      x0=x_a, y0=H_a, x1=x_a + dx_b, y1=H_a + dy_b,
                      line=dict(color=col, width=2.5, dash='dashdot'))
        # arrow from old W tip to new W tip
        fig.add_annotation(
            x=x_a + dx_a, y=H_a + dy_a,
            ax=x_a + dx_b, ay=H_a + dy_b,
            xref='x', yref='y', axref='x', ayref='y',
            text='', showarrow=True,
            arrowhead=3, arrowcolor=col, arrowwidth=2.5, arrowsize=1.0,
        )
        # Δ label above midpoint
        mx = x_a + (dx_a + dx_b) * 0.5 / 2
        my = H_a + (dy_a + dy_b) * 0.5 / 2 + H_a * 0.09
        fig.add_annotation(
            x=mx, y=my,
            text=f"ΔW {sign}{d:.0f} mm", showarrow=False,
            font=dict(color=col, size=11, family='monospace'),
            bgcolor='rgba(13,17,23,0.90)',
        )

    # ── Tight axis range — car fills the figure ───────────────────────────────
    x_arr_pos = L + L * 0.030
    x_label_end = x_arr_pos + L * 0.055   # room for H label + Δ label
    x_right = x_label_end + max(dx_a, dx_b) + L * 0.02
    y_top   = H_max + dy_max + H_max * 0.18
    y_bot   = -H_max * 0.12

    fig.update_layout(
        xaxis=dict(visible=False, range=[-L * 0.03, x_right]),
        yaxis=dict(visible=False, range=[y_bot, y_top]),
        height=height,
    )
    return fig


def _cd_zone_color(cd: float) -> str:
    if cd < 0.25:
        return GREEN
    if cd < 0.35:
        return AMBER
    return RED


def cd_gauge(
    cd_value: float,
    target_cd: float = None,
    cd_min: float = 0.15,
    cd_max: float = 0.50,
    height: int = 185,
) -> go.Figure:
    """
    Semi-circular Plotly Indicator gauge for Cd.
    When target_cd is given, the amber threshold line shows target; blue bar shows achieved.
    """
    cd_value = round(cd_value, 4)
    threshold_val = target_cd if target_cd is not None else cd_value
    threshold_color = AMBER if target_cd is not None else BLUE
    val_color = _cd_zone_color(cd_value)

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=cd_value,
        domain=dict(x=[0.05, 0.95], y=[0, 1]),
        number=dict(
            prefix='Cd: ',
            valueformat='.4f',
            font=dict(size=20, color=val_color),
        ),
        gauge=dict(
            axis=dict(
                range=[cd_min, cd_max],
                tickwidth=1,
                tickcolor=GREY,
                tickfont=dict(color=GREY, size=9),
                tickvals=[0.15, 0.25, 0.35, 0.50],
                ticktext=['0.15', '0.25', '0.35', '0.50'],
            ),
            bar=dict(color=val_color, thickness=0.22),
            bgcolor=PANEL,
            borderwidth=0,
            steps=[
                dict(range=[cd_min, 0.25], color='rgba(67,160,71,0.22)'),
                dict(range=[0.25, 0.35],   color='rgba(255,143,0,0.18)'),
                dict(range=[0.35, cd_max], color='rgba(229,57,53,0.18)'),
            ],
            threshold=dict(
                line=dict(color=threshold_color, width=3),
                thickness=0.82,
                value=threshold_val,
            ),
        ),
        title=dict(
            text='Drag Coefficient',
            font=dict(color=GREY, size=10),
        ),
    ))

    # Legend strip when both values are shown
    if target_cd is not None:
        fig.add_annotation(
            x=0.5, y=-0.10, xref='paper', yref='paper',
            text=(f'<span style="color:{BLUE}">▬ Achieved {cd_value:.4f}</span>'
                  f'   <span style="color:{AMBER}">| Target {target_cd:.3f}</span>'),
            showarrow=False,
            font=dict(size=9),
            align='center',
        )

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=height,
        margin=dict(l=30, r=30, t=35, b=35 if target_cd else 8),
    )
    return fig


def _cl_zone_color(cl: float) -> str:
    if cl <= 0.05:
        return GREEN
    if cl <= 0.20:
        return AMBER
    return RED


def cl_gauge(
    cl_value: float,
    cl_min: float = -0.30,
    cl_max: float = 0.50,
    height: int = 185,
) -> go.Figure:
    """Small semi-circular gauge for lift coefficient Cl."""
    cl_value = round(cl_value, 4)
    val_color = _cl_zone_color(cl_value)
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=cl_value,
        domain=dict(x=[0.05, 0.95], y=[0, 1]),
        number=dict(
            prefix='Cl: ',
            valueformat='.4f',
            font=dict(size=18, color=val_color),
        ),
        gauge=dict(
            axis=dict(
                range=[cl_min, cl_max],
                tickwidth=1,
                tickcolor=GREY,
                tickfont=dict(color=GREY, size=8),
                tickvals=[-0.30, 0.0, 0.05, 0.20, 0.50],
                ticktext=['-0.30', '0.0', '0.05', '0.20', '0.50'],
            ),
            bar=dict(color=val_color, thickness=0.22),
            bgcolor=PANEL,
            borderwidth=0,
            steps=[
                dict(range=[cl_min, 0.05], color='rgba(67,160,71,0.20)'),
                dict(range=[0.05, 0.20],   color='rgba(255,143,0,0.15)'),
                dict(range=[0.20, cl_max], color='rgba(229,57,53,0.15)'),
            ],
            threshold=dict(
                line=dict(color=WHITE, width=2),
                thickness=0.70,
                value=0,
            ),
        ),
        title=dict(
            text='Lift Coefficient  ' + ('↓ Downforce' if cl_value <= 0.05 else '↑ Lift'),
            font=dict(color=val_color, size=10),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=height,
        margin=dict(l=30, r=30, t=25, b=5),
    )
    return fig


def cd_scatter_with_highlight(
    geo_df,
    current_cd: float,
    current_style: str,
) -> go.Figure:
    """Scatter of 1163 designs + highlighted current design."""
    style_map = {'F': 'Fastback', 'N': 'Notchback', 'E': 'Estateback'}
    colors    = {'Fastback': '#1565C0', 'Notchback': '#E65100', 'Estateback': '#1B5E20'}
    hl_colors = {'Fastback': BLUE, 'Notchback': AMBER, 'Estateback': GREEN}

    geo_df = geo_df.copy()
    geo_df['style_label'] = geo_df['design_id'].apply(
        lambda x: style_map.get(x[0], 'Fastback')
    )

    fig = go.Figure()
    for sty, grp in geo_df.groupby('style_label'):
        fig.add_trace(go.Scatter(
            x=grp['rear_slant_deg'], y=grp['cd'],
            mode='markers',
            marker=dict(color=colors[sty], size=7, opacity=0.82),
            name=sty,
            hovertemplate='<b>%{customdata}</b><br>slant: %{x:.1f}°  Cd: %{y:.4f}',
            customdata=grp['design_id'],
        ))

    hl_style = current_style.capitalize()
    if current_style == 'fastback':   hl_style = 'Fastback'
    elif current_style == 'notchback': hl_style = 'Notchback'
    else:                              hl_style = 'Estateback'
    style_mean_slant = geo_df[geo_df['style_label'] == hl_style]['rear_slant_deg'].mean()

    fig.add_trace(go.Scatter(
        x=[style_mean_slant], y=[current_cd],
        mode='markers',
        marker=dict(
            symbol='star', size=16,
            color=hl_colors.get(hl_style, BLUE),
            line=dict(color=WHITE, width=1.5),
        ),
        name='Your design',
        hovertemplate=f'Your design<br>Cd = {current_cd:.4f}',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis_title='Rear slant angle (°)',
        yaxis_title='Cd',
        margin=dict(l=45, r=112, t=12, b=40),
        height=215,
        legend=dict(
            orientation='v',
            xanchor='left',  x=1.02,
            yanchor='top',   y=0.84,
            bgcolor='rgba(13,17,23,0.88)',
            bordercolor='#30363d', borderwidth=1,
            font=dict(size=10, color=WHITE),
            itemclick='toggleothers',
            itemdoubleclick='toggle',
        ),
    )
    return fig
