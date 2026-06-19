"""
app/car_viz.py

Parametric 2D car side-profile visualisation using Plotly.
All dimensions in mm; blueprint dark aesthetic (#0d1117 background).
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

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
    Build (x_mm, y_mm) polygon for car side profile.

    Uses keys: ref_length_mm, height_mm, rear_slant_deg, cabin_frac, style.

    Coordinate origin: front-bottom of car.
    """
    L     = float(params['ref_length_mm'])
    H     = float(params['height_mm'])
    slant = float(params['rear_slant_deg'])
    cabin = float(params['cabin_frac'])
    style = str(params['style'])

    gc       = max(100.0, 0.08 * H)      # ground clearance
    hood_h   = gc + 0.24 * H             # hood surface height

    # Key x positions
    x_hood   = 0.08 * L                  # front of hood (after bumper)
    x_ws     = 0.22 * L                  # windshield base
    x_a      = 0.30 * L                  # A-pillar / roof start
    x_c      = x_a + cabin * 0.62 * L   # C-pillar / roof end
    x_c      = min(x_c, 0.83 * L)
    x_rear   = 0.93 * L                  # rear bumper top
    x_end    = L                         # rear-most point

    # Rear height from slant
    rear_drop = np.tan(np.radians(slant)) * (x_rear - x_c)
    rear_h    = H - rear_drop
    rear_h    = float(np.clip(rear_h, gc + 0.15 * H, H))

    if style == 'notchback':
        trunk_h    = gc + 0.44 * H       # trunk lid height
        # rear_slant_deg controls rear-window steepness:
        # low slant → gentle/long window run; high slant → steep/short run
        slant_norm  = float(np.clip(slant / 21.4, 0.0, 1.0))
        window_run  = (0.04 + (1.0 - slant_norm) * 0.11) * L
        x_tc        = x_c + window_run
        x_te        = min(x_tc + 0.12 * L, 0.91 * L)
        xs = [
            0,          0.04*L,   x_hood,   x_ws,     x_ws,     x_a,
            x_a,        x_c,      x_tc,     x_te,     x_te,     x_rear,
            x_rear,     x_end,    x_end,    0,
        ]
        ys = [
            gc,         gc,       hood_h,   hood_h,   hood_h,   H,
            H,          H,        trunk_h,  trunk_h,  gc,       gc,
            gc,         gc,       gc,       gc,
        ]
    else:
        # Fastback / estateback: continuous rear slope
        xs = [
            0,      0.04*L,  x_hood,  x_ws,    x_ws,   x_a,
            x_a,    x_c,     x_rear,  x_rear,  x_end,  x_end,  0,
        ]
        ys = [
            gc,     gc,      hood_h,  hood_h,  hood_h, H,
            H,      H,       rear_h,  gc,      gc,     gc,     gc,
        ]

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
    gc = max(100.0, 0.08 * H)

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


def draw_car_comparison(
    params_before: dict,
    params_after: dict,
    delta: dict,
) -> go.Figure:
    """
    Side-profile overlay: grey ghost (before) + blue solid (after).
    Annotates each changed dimension with Δmm / Δ°.
    """
    L      = float(params_after['ref_length_mm'])
    H_b    = float(params_before['height_mm'])
    H_a    = float(params_after['height_mm'])
    H_max  = max(H_b, H_a)
    gc_a   = max(100.0, 0.08 * H_a)
    wheel_r = 0.115 * H_max

    xs_b, ys_b = side_profile_coords(params_before)
    xs_a, ys_a = side_profile_coords(params_after)

    fig = go.Figure()

    # Ghost before
    fig.add_trace(go.Scatter(
        x=xs_b, y=ys_b,
        fill='toself', fillcolor='rgba(110,118,129,0.08)',
        line=dict(color='#6e7681', width=2, dash='dash'),
        mode='lines', name='Before', hoverinfo='skip',
    ))
    # Solid after
    fig.add_trace(go.Scatter(
        x=xs_a, y=ys_a,
        fill='toself', fillcolor='rgba(30,136,229,0.18)',
        line=dict(color=BLUE, width=2.5),
        mode='lines', name='After', hoverinfo='skip',
    ))

    # Wheels (after state)
    for wx in (0.22 * L, 0.77 * L):
        cx, cy = _circle(wx, gc_a, wheel_r * 0.88)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, fill='toself', fillcolor='#2d333b',
            line=dict(color=GREY, width=1),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

    # Delta annotations
    ann_params = []
    if 'height_mm' in delta:
        v0, v1, d = delta['height_mm']
        col = GREEN if d < 0 else AMBER
        sign = '+' if d > 0 else ''
        ann_params.append(dict(
            x=L * 1.08, y=H_a / 2,
            text=f"ΔH {sign}{d:.0f} mm",
            color=col,
            arrow_x=L * 1.06, arrow_y=H_a / 2,
        ))
    if 'rear_slant_deg' in delta:
        v0, v1, d = delta['rear_slant_deg']
        col = GREEN if d < 0 else AMBER
        sign = '+' if d > 0 else ''
        cabin = params_after['cabin_frac']
        x_c   = min(0.30 * L + cabin * 0.62 * L, 0.83 * L)
        ann_params.append(dict(
            x=(x_c + 0.93 * L) / 2, y=H_max * 0.75,
            text=f"Δα {sign}{d:.1f}°",
            color=col,
            arrow_x=None, arrow_y=None,
        ))
    if 'width_mm' in delta:
        v0, v1, d = delta['width_mm']
        col = GREEN if d < 0 else AMBER
        sign = '+' if d > 0 else ''
        ann_params.append(dict(
            x=L * 0.5, y=H_max * 1.12,
            text=f"ΔW {sign}{d:.0f} mm",
            color=col,
            arrow_x=None, arrow_y=None,
        ))

    for ap in ann_params:
        if ap['arrow_x'] is not None:
            fig.add_annotation(
                x=ap['x'], y=ap['y'],
                ax=ap['arrow_x'], ay=ap['arrow_y'],
                xref='x', yref='y', axref='x', ayref='y',
                text=ap['text'], showarrow=True,
                arrowhead=2, arrowcolor=ap['color'], arrowwidth=1.5,
                font=dict(color=ap['color'], size=12),
                bgcolor='rgba(13,17,23,0.75)',
            )
        else:
            fig.add_annotation(
                x=ap['x'], y=ap['y'],
                text=ap['text'], showarrow=False,
                font=dict(color=ap['color'], size=12),
                bgcolor='rgba(13,17,23,0.75)',
            )

    fig.add_hline(y=0, line=dict(color=GREY, width=0.8, dash='dot'))
    fig.add_hrect(y0=-wheel_r * 0.5, y1=0,
                  fillcolor='rgba(110,118,129,0.06)', line_width=0)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-0.06 * L, L * 1.22]),
        yaxis=dict(
            visible=False, scaleanchor='x', scaleratio=1,
            range=[-wheel_r * 1.4, H_max * 1.38],
        ),
        margin=dict(l=5, r=5, t=30, b=5),
        height=280,
        legend=dict(
            x=0.02, y=0.97,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color=WHITE, size=11),
        ),
    )
    return fig


def cd_gauge(cd_value: float, cd_min: float = 0.15, cd_max: float = 0.50) -> go.Figure:
    """Semi-circular Plotly Indicator gauge for Cd."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=cd_value,
        number=dict(
            suffix='',
            valueformat='.4f',
            font=dict(size=26, color=WHITE),
        ),
        gauge=dict(
            axis=dict(
                range=[cd_min, cd_max],
                tickwidth=1,
                tickcolor=GREY,
                tickfont=dict(color=GREY, size=9),
                nticks=6,
            ),
            bar=dict(color=BLUE, thickness=0.22),
            bgcolor=PANEL,
            borderwidth=0,
            steps=[
                dict(range=[cd_min, 0.25], color='rgba(67,160,71,0.22)'),
                dict(range=[0.25, 0.35],   color='rgba(255,143,0,0.18)'),
                dict(range=[0.35, cd_max], color='rgba(229,57,53,0.18)'),
            ],
            threshold=dict(
                line=dict(color=AMBER, width=3),
                thickness=0.75,
                value=cd_value,
            ),
        ),
        title=dict(
            text='C<sub>d</sub>',
            font=dict(color=GREY, size=13),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=185,
        margin=dict(l=10, r=10, t=30, b=5),
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
        margin=dict(l=45, r=15, t=12, b=40),
        height=215,
        legend=dict(
            orientation='v',
            xanchor='left', x=0.01,
            yanchor='top',  y=0.99,
            bgcolor='rgba(13,17,23,0.72)',
            bordercolor='#30363d', borderwidth=1,
            font=dict(size=10, color=WHITE),
        ),
    )
    return fig
