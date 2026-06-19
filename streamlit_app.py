"""
streamlit_app.py — ParaKoop: Parametric Koopman Aerodynamics

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.car_viz import (
    BG, BLUE, AMBER, GREEN, RED, GREY, WHITE, PANEL,
    cd_gauge, cd_scatter_with_highlight, draw_car_side, draw_car_comparison,
)
from app.model_utils import load_model_and_dataset, theta_from_sliders, predict

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ParaKoop | Car Aero Design",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG}; }}
  [data-testid="stAppViewContainer"] {{ background-color: {BG}; }}
  [data-testid="stHeader"] {{ background-color: {BG}; border-bottom: none; height: 0 !important; }}
  [data-testid="stToolbar"] {{ display: none !important; }}
  .block-container {{ padding-top: 1.8rem !important; max-width: 100% !important; padding-bottom: 0.5rem !important; }}

  .pk-header {{
    display: flex; align-items: baseline; gap: 14px;
    padding: 4px 0 10px 0; border-bottom: 1px solid #21262d; margin-bottom: 10px;
  }}
  .pk-brand  {{ font-size: 26px; font-weight: 700; color: {WHITE}; letter-spacing: -0.5px; }}
  .pk-sub    {{ font-size: 13px; color: {GREY}; }}
  .pk-badge  {{
    font-size: 11px; color: {BLUE}; background: rgba(30,136,229,0.12);
    border: 1px solid rgba(30,136,229,0.3); border-radius: 20px; padding: 2px 10px;
  }}

  .pk-metric {{
    background: {PANEL}; border-radius: 8px; padding: 12px 16px;
    border-left: 3px solid {BLUE}; margin-bottom: 10px;
  }}
  .pk-metric-label {{ font-size: 11px; color: {GREY}; text-transform: uppercase; letter-spacing: 0.6px; }}
  .pk-metric-value {{ font-size: 28px; font-weight: 700; color: {WHITE}; line-height: 1.1; }}
  .pk-metric-sub   {{ font-size: 12px; color: {GREY}; margin-top: 2px; }}

  .pk-gr-pass {{
    background: rgba(67,160,71,0.15); border: 1px solid rgba(67,160,71,0.4);
    border-radius: 6px; padding: 8px 12px; text-align: center;
    color: {GREEN}; font-weight: 600; font-size: 13px; margin-bottom: 4px;
  }}
  .pk-gr-fail {{
    background: rgba(229,57,53,0.15); border: 1px solid rgba(229,57,53,0.4);
    border-radius: 6px; padding: 8px 12px; text-align: center;
    color: {RED}; font-weight: 600; font-size: 13px; margin-bottom: 4px;
  }}
  .pk-gr-note {{ font-size: 11px; color: {GREY}; line-height: 1.4; margin-bottom: 12px; }}

  .pk-section {{
    font-size: 11px; font-weight: 600; color: {GREY};
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 8px 0 5px 0; padding-bottom: 3px; border-bottom: 1px solid #21262d;
  }}

  .stButton > button {{
    background: {BLUE}; color: {WHITE}; border: none; border-radius: 6px;
    font-weight: 600; letter-spacing: 0.3px; transition: background 0.2s;
  }}
  .stButton > button:hover {{ background: #1565C0 !important; }}

  /* ── Slider: bright label + value ── */
  [data-testid="stSlider"] label,
  [data-testid="stSlider"] label p {{
    color: {WHITE} !important; font-size: 13px !important;
  }}
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
    color: #8b949e !important;
  }}
  [data-testid="stSlider"] div[data-baseweb="tooltip"] span {{
    color: {WHITE} !important; background: {PANEL} !important;
  }}

  /* ── Checkbox: bright label ── */
  [data-testid="stCheckbox"] label p {{ color: {WHITE} !important; }}
  [data-testid="stToggle"]   label p {{ color: {WHITE} !important; }}

  /* ── Caption / helper text ── */
  [data-testid="stCaptionContainer"] p {{ color: #8b949e !important; }}

  /* ── Radio: selected = white bold, unselected = muted ── */
  [data-testid="stRadio"] label {{
    color: #6e7681 !important; font-size: 13px !important;
    transition: color 0.15s;
  }}
  [data-testid="stRadio"] label:has(input:checked) {{
    color: {WHITE} !important; font-weight: 600 !important;
  }}
  [data-testid="stRadio"] label:has(input:checked) p {{
    color: {WHITE} !important; font-weight: 600 !important;
  }}
  [data-testid="stRadio"] label p {{ color: inherit !important; }}

  /* ── General body text ── */
  p, .stMarkdown p {{ color: #c9d1d9 !important; }}
  label {{ color: {WHITE} !important; }}

  /* ── Help icon: ? → italic i  (only inside widget labels, not dataframe headers) ── */
  label [data-testid="stTooltipHoverTarget"] svg {{ display: none !important; }}
  label [data-testid="stTooltipHoverTarget"] {{
    font-style: italic !important; font-weight: 700 !important;
    font-size: 11px !important; color: #8b949e !important;
    background: rgba(110,118,129,0.18) !important;
    border-radius: 50% !important;
    width: 16px !important; height: 16px !important;
    display: inline-flex !important;
    align-items: center !important; justify-content: center !important;
    cursor: help !important; vertical-align: middle !important;
    margin-left: 4px !important;
  }}
  label [data-testid="stTooltipHoverTarget"]::after {{
    content: "i" !important;
  }}

  /* ── Tooltip popup: white text on dark bg ── */
  div[role="tooltip"], [data-testid="stTooltipContent"],
  div[data-radix-popper-content-wrapper] {{
    background: {PANEL} !important;
    color: {WHITE} !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
  }}
  div[role="tooltip"] *, [data-testid="stTooltipContent"] *,
  div[data-radix-popper-content-wrapper] * {{
    color: {WHITE} !important;
  }}

  /* ── Custom CSS tooltip for HTML elements ── */
  .pk-tip {{ position: relative; display: block; }}
  .pk-tip-inner {{ cursor: help; }}
  .pk-tip::after {{
    content: attr(data-tip);
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: {PANEL};
    color: {WHITE};
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    white-space: pre-wrap;
    min-width: 230px;
    max-width: 340px;
    z-index: 9999;
    pointer-events: none;
    line-height: 1.55;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  }}
  .pk-tip:hover::after {{ display: block; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pk-header">
  <span class="pk-brand">🏎️ ParaKoop</span>
  <span class="pk-sub">Parametric Koopman Aerodynamics</span>
  <span class="pk-badge">DrivAerNet · AhmedML · 9,620 designs</span>
</div>
""", unsafe_allow_html=True)

# ── Shared resources (cached) ─────────────────────────────────────────────────

with st.spinner("Loading Koopman model…"):
    model, dataset = load_model_and_dataset()

_GEO_CSV = os.path.join(_ROOT, 'data', 'drivaernet', 'geometry_features.csv')
geo_df   = pd.read_csv(_GEO_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions (must be defined before tab rendering)
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_TIPS = {
    "Target C_d":   "Drag Coefficient target. Cd = F_drag / (½ρv²A). Typical cars: 0.25–0.35. Sports cars: 0.18–0.25.",
    "Achieved C_d": ("Cd the optimiser reached after gradient descent.\n"
                     "Err = Achieved − Target. |Err| < 0.005 is a good match."),
    "Achieved C_l": ("Lift Coefficient. Cl = F_lift / (½ρv²A).\n"
                     "Negative = downforce (pushes car down, improves grip).\n"
                     "Positive = aerodynamic lift (reduces tyre contact force)."),
    "Physics":      "Summary of Re / Ma / Eu guardrail checks. All must pass for the design to be physically valid.",
    "Lift C_l":     ("Lift Coefficient. Cl = F_lift / (½ρv²A).\n"
                     "Negative = downforce. Positive = lift.\n"
                     "Highway stability prefers Cl ≤ 0."),
}

def _metric(label: str, value: str, sub: str = '', color: str = BLUE) -> str:
    tip = _METRIC_TIPS.get(label, '').replace('"', '&quot;').replace('\n', '&#10;')
    tip_attr = f'class="pk-tip" data-tip="{tip}"' if tip else ''
    return f"""
    <div {tip_attr}>
    <div class="pk-metric" style="border-color:{color}">
      <div class="pk-metric-label">{label}{' <em style="font-size:10px;opacity:.6;font-weight:400">i</em>' if tip else ''}</div>
      <div class="pk-metric-value" style="color:{color};font-size:22px">{value}</div>
      <div class="pk-metric-sub">{sub}</div>
    </div>
    </div>"""


_GR_TIPS = {
    'Re': ("Reynolds Number  Re = U∞ × L / ν\n"
           "Measures inertial vs viscous forces.\n"
           "Valid range: 3×10⁶ – 3×10⁷ (full-scale car at highway speed).\n"
           "U∞ = 40 m/s, ν = 1.516×10⁻⁵ m²/s"),
    'Ma': ("Mach Number  Ma = U∞ / c\n"
           "Ratio of flow speed to speed of sound.\n"
           "Must be < 0.30 for incompressible RANS to be valid.\n"
           "U∞ = 40 m/s → Ma ≈ 0.117"),
    'Eu': ("Euler Number proxy  Eu ≈ Cd\n"
           "Ratio of pressure difference to dynamic pressure.\n"
           "Plausible automotive range: 0.15 – 0.60.\n"
           "Below 0.15 = physically implausible; above 0.60 = bluff body"),
}

def _render_guardrails(guardrails: dict) -> None:
    gr_cols = st.columns(3)
    for i, (name, g) in enumerate(guardrails.items()):
        with gr_cols[i]:
            cls = 'pk-gr-pass' if g['passed'] else 'pk-gr-fail'
            sym = '✓' if g['passed'] else '✗'
            tip = _GR_TIPS.get(name, '').replace('"', '&quot;').replace('\n', '&#10;')
            st.markdown(
                f'<div class="pk-tip" data-tip="{tip}">'
                f'  <div class="{cls}">{sym} {name} <em style="font-size:10px;opacity:.7">i</em></div>'
                f'  <div class="pk-gr-note">{g["note"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_design_result(result: dict, all_results: list) -> None:
    """Full-width inverse design result panel."""
    err = result['achieved_cd'] - result['target_cd']
    ok  = result['guardrails_ok']

    # ── Metric strip ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric("Target C_d", f"{result['target_cd']:.3f}"),
                    unsafe_allow_html=True)
    with c2:
        ec = GREEN if abs(err) < 0.005 else AMBER
        st.markdown(_metric("Achieved C_d", f"{result['achieved_cd']:.4f}",
                             f"err {err:+.4f}", ec), unsafe_allow_html=True)
    with c3:
        cc = GREEN if result['achieved_cl'] <= 0 else AMBER
        lbl = "Downforce ↓" if result['achieved_cl'] <= 0 else "Lift ↑"
        st.markdown(_metric("Achieved C_l", f"{result['achieved_cl']:+.4f}", lbl, cc),
                    unsafe_allow_html=True)
    with c4:
        pc = GREEN if ok else RED
        st.markdown(_metric("Physics", "✓ All pass" if ok else "✗ Violations", '', pc),
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Car comparison + batch cards side by side ─────────────────────────────
    active_cd = result['achieved_cd']

    if len(all_results) > 1:
        cards_col, car_col = st.columns([1, 2.6], gap="large")
        with cards_col:
            st.markdown('<div class="pk-section">All styles — click to switch</div>',
                        unsafe_allow_html=True)
            for i, res in enumerate(all_results):
                is_active = abs(res['achieved_cd'] - active_cd) < 1e-6
                col   = BLUE if is_active else (GREEN if i == 0 else GREY)
                label = "★ Best" if i == 0 else f"#{i + 1}"
                err_i = res['achieved_cd'] - res['target_cd']
                border = f"outline: 2px solid {BLUE};" if is_active else ""
                st.markdown(f"""
                <div class="pk-metric" style="border-color:{col};{border};margin-bottom:6px">
                  <div class="pk-metric-label">{label} · {res['start_params']['style'].title()}</div>
                  <div class="pk-metric-value" style="color:{col};font-size:18px">
                    Cd {res['achieved_cd']:.4f}
                  </div>
                  <div class="pk-metric-sub">err {err_i:+.4f}</div>
                </div>""", unsafe_allow_html=True)
                if not is_active:
                    if st.button("View this design →", key=f"batch_btn_{i}",
                                 use_container_width=True):
                        st.session_state['design_result'] = res
                        st.rerun()
                else:
                    st.caption("▲ Currently shown above")
                st.markdown("")
        with car_col:
            st.markdown('<div class="pk-section">Before → After</div>',
                        unsafe_allow_html=True)
            fig_cmp = draw_car_comparison(
                result['start_params'], result['end_params'], result['delta'],
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.markdown('<div class="pk-section">Before → After</div>', unsafe_allow_html=True)
        fig_cmp = draw_car_comparison(
            result['start_params'], result['end_params'], result['delta'],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Delta table + guardrails ──────────────────────────────────────────────
    col_tbl, col_gr = st.columns([3, 2], gap="large")

    with col_tbl:
        st.markdown('<div class="pk-section">Geometry Changes</div>',
                    unsafe_allow_html=True)
        if result['delta']:
            rows = []
            for param, (v0, v1, d) in sorted(
                result['delta'].items(), key=lambda x: abs(x[1][2]), reverse=True
            ):
                unit = '°' if 'deg' in param else ('mm' if '_mm' in param else '')
                rows.append({
                    'Parameter': param.replace('_', ' '),
                    'Before': f"{v0:.1f}{unit}",
                    'After':  f"{v1:.1f}{unit}",
                    'Δ':      f"{d:+.1f}{unit}",
                    'Effect': "▼ less drag" if d < 0 else "▲ more drag",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
        else:
            st.info("Model already near target — no significant changes needed.")

    with col_gr:
        st.markdown('<div class="pk-section">Physics Guardrails</div>',
                    unsafe_allow_html=True)
        _render_guardrails(result['guardrails'])


def _render_explore(gdf: pd.DataFrame) -> None:
    """Explore tab content — single row of 3 charts + stat strip."""
    style_map  = {'F': 'Fastback', 'N': 'Notchback', 'E': 'Estateback'}
    color_disc = {'Fastback': '#1E88E5', 'Notchback': '#FF8F00', 'Estateback': '#43A047'}

    gdf = gdf.copy()
    gdf['Style'] = gdf['design_id'].apply(lambda x: style_map.get(x[0], 'Fastback'))
    gdf['height_mm'] = gdf['height_mm'].fillna(gdf['height_mm'].median())

    c_scatter, c_box, c_val = st.columns([1.6, 1, 1], gap="medium")

    with c_scatter:
        fig1 = px.scatter(
            gdf, x='rear_slant_deg', y='cd',
            color='Style', size='height_mm', size_max=9,
            hover_data=['design_id', 'height_mm', 'width_mm', 'cabin_length_frac'],
            labels={'rear_slant_deg': 'Rear Slant (°)', 'cd': 'Cd'},
            title="Cd vs Rear Slant  ·  1,163 designs",
            color_discrete_map=color_disc,
            template='plotly_dark', opacity=0.75,
        )
        fig1.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG, height=268,
            title_font_color=WHITE,
            margin=dict(l=45, r=108, t=38, b=40),
            legend=dict(
                orientation='v',
                xanchor='left', x=1.02,
                yanchor='top',  y=1.0,
                bgcolor='rgba(13,17,23,0.88)', bordercolor='#30363d', borderwidth=1,
                font=dict(size=9, color=WHITE),
                itemclick='toggleothers', itemdoubleclick='toggle',
            ),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c_box:
        fig2 = px.box(
            gdf, x='Style', y='cd', color='Style',
            color_discrete_map=color_disc,
            labels={'cd': 'Cd'},
            title="Cd Distribution",
            template='plotly_dark', points='outliers',
        )
        fig2.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
            height=268, margin=dict(l=40, r=10, t=38, b=40),
            title_font_color=WHITE,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c_val:
        val_path = os.path.join(_ROOT, 'results', 'ahmed_holdout_val_v3.csv')
        if os.path.exists(val_path):
            vdf = pd.read_csv(val_path)
            mae = (vdf['cd_pred'] - vdf['cd_cfd']).abs().mean()
            lo  = min(vdf['cd_cfd'].min(), vdf['cd_pred'].min()) - 0.01
            hi  = max(vdf['cd_cfd'].max(), vdf['cd_pred'].max()) + 0.01

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=vdf['cd_cfd'], y=vdf['cd_pred'], mode='markers',
                marker=dict(color=BLUE, size=7, opacity=0.88,
                            line=dict(color=WHITE, width=0.5)),
                name='ParaKoop',
                hovertemplate='CFD: %{x:.4f}<br>Pred: %{y:.4f}<extra></extra>',
            ))
            fig3.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode='lines',
                line=dict(color=GREY, dash='dot', width=1.5),
                name='Perfect', hoverinfo='skip',
            ))
            fig3.update_layout(
                title=dict(
                    text=f'Pred vs CFD  (MAE = {mae:.4f})',
                    font=dict(color=WHITE),
                ),
                xaxis_title='CFD Cd', yaxis_title='Predicted Cd',
                template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
                height=268, margin=dict(l=45, r=10, t=38, b=45),
                legend=dict(
                    x=0.02, y=0.97,
                    bgcolor='rgba(110,118,129,0.22)',
                    bordercolor='#6e7681', borderwidth=1,
                    font=dict(color=WHITE, size=10),
                ),
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="pk-section">Validation Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(_metric("DrivAerNet MAE", "0.0087", "vs high-fidelity CFD GT", GREEN),
                    unsafe_allow_html=True)
    with s2:
        st.markdown(_metric("AhmedML MAE", "0.0404", "74-sample holdout", BLUE),
                    unsafe_allow_html=True)
    with s3:
        st.markdown(_metric("Training samples", "9,620", "DrivAerNet + AhmedML", GREY),
                    unsafe_allow_html=True)
    with s4:
        st.markdown(_metric("Koopman dim K", "256", "rank-32 low-rank A(θ)", GREY),
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT (Forward flow)
# ══════════════════════════════════════════════════════════════════════════════

tab_predict, tab_design, tab_explore = st.tabs([
    "⚡  Predict", "🎯  Design", "📊  Explore",
])

with tab_predict:
    left, car_col, right_col = st.columns([1, 2.3, 0.9], gap="medium")

    with left:
        st.markdown('<div class="pk-section">Body Style</div>', unsafe_allow_html=True)
        style = st.radio(
            "style", ['fastback', 'notchback', 'estateback'],
            format_func=lambda s: s.title(),
            horizontal=True, label_visibility='collapsed',
            help=("Rear body shape:\n"
                  "• Fastback — continuous sloped roof flowing into the tail\n"
                  "• Notchback — separate boot/trunk with vertical rear glass\n"
                  "• Estateback — wagon/estate tail, almost vertical rear"),
        )
        mean_theta  = dataset.style_mean_theta(style)
        mean_params = dataset.theta_to_named_params(mean_theta)

        st.markdown('<div class="pk-section">Geometry</div>', unsafe_allow_html=True)
        slant = st.slider(
            "Rear slant angle (°)", 0.0, 21.4,
            float(np.clip(mean_params['rear_slant_deg'], 0.0, 21.4)), 0.1,
            help=("Angle of the rear body slope from horizontal.\n"
                  "Higher = more aggressive taper (fastback); lower = more upright.\n"
                  "DrivAerNet dataset range: 0° – 21.4°"),
        )
        height = st.slider(
            "Height (mm)", 1210, 1753,
            int(np.clip(mean_params['height_mm'], 1210, 1753)), 5,
            help=("Overall car height from ground to roof in mm.\n"
                  "Taller cars have greater frontal area → generally higher Cd.\n"
                  "DrivAerNet range: 1,210 – 1,753 mm"),
        )
        w_h = st.slider(
            "Width / Height ratio", 0.50, 1.80,
            float(np.clip(mean_params['width_height_ratio'], 0.50, 1.80)), 0.01,
            help=("Car width divided by car height (dimensionless).\n"
                  "Higher ratio = wider, lower car (sports-car proportions).\n"
                  "DrivAerNet range: 0.50 – 1.80"),
        )
        cabin = st.slider(
            "Cabin fraction", 0.35, 0.70,
            float(np.clip(mean_params['cabin_frac'], 0.35, 0.70)), 0.01,
            help=("Passenger cabin length as a fraction of total body length.\n"
                  "Higher = longer cabin, shorter rear overhang.\n"
                  "Affects where the C-pillar (rear of roofline) sits."),
        )
        detailed = st.checkbox(
            "Detailed (mirrors + wheels)", value=True,
            help=("Detailed geometry flag (θ₇).\n"
                  "Checked = full detail including side mirrors and wheel geometry.\n"
                  "Unchecked = simplified smooth body (lower Cd in some configs)."),
        )

    theta  = theta_from_sliders(style, slant, height, w_h, cabin,
                                 1.0 if detailed else 0.0)
    perf   = predict(model, theta)
    params = dataset.theta_to_named_params(theta)

    from koopman.inverse_design import check_physics_guardrails
    guardrails = check_physics_guardrails(params, perf['Cd'])

    with car_col:
        # Car figure with guardrail badges overlaid top-right
        fig_car = draw_car_side(params, show_dims=True)
        for j, (gr_name, g) in enumerate(guardrails.items()):
            sym  = '✓' if g['passed'] else '✗'
            clr  = GREEN if g['passed'] else RED
            fig_car.add_annotation(
                x=0.995, y=0.99 - j * 0.16,
                xref='paper', yref='paper',
                text=f" {sym} {gr_name} ",
                showarrow=False,
                font=dict(color=clr, size=11, family='monospace'),
                bgcolor='rgba(33,38,45,0.88)',
                bordercolor=clr, borderwidth=1, borderpad=4,
                xanchor='right', yanchor='top',
            )
        st.plotly_chart(fig_car, use_container_width=True)

        # Live geometry table directly below car
        st.markdown('<div class="pk-section">Current Geometry</div>',
                    unsafe_allow_html=True)
        live_rows = [
            ('Style',        params['style'].title()),
            ('Rear slant',   f"{params['rear_slant_deg']:.1f} °"),
            ('Height',       f"{params['height_mm']:.0f} mm"),
            ('Width',        f"{params['width_mm']:.0f} mm"),
            ('Cabin frac',   f"{params['cabin_frac']:.3f}"),
            ('Cd',           f"{perf['Cd']:.4f}"),
            ('Cl',           f"{perf['Cl']:+.4f}"),
        ]
        st.dataframe(
            pd.DataFrame(live_rows, columns=['Parameter', 'Value']),
            hide_index=True, use_container_width=True,
        )

    with right_col:
        st.plotly_chart(cd_gauge(perf['Cd']), use_container_width=True)
        cl_col = GREEN if perf['Cl'] <= 0 else AMBER
        cl_lbl = "Downforce ↓" if perf['Cl'] <= 0 else "Lift ↑"
        tip_cl = ("Lift Coefficient Cl = F_lift / (½ρv²A).&#10;"
                  "Negative = downforce (improves grip).&#10;"
                  "Highway stability prefers Cl ≤ 0.")
        st.markdown(f"""
        <div class="pk-tip" data-tip="{tip_cl}">
        <div class="pk-metric" style="border-color:{cl_col};margin-top:-6px">
          <div class="pk-metric-label">Lift C<sub>l</sub> <em style="font-size:10px;opacity:.6">i</em></div>
          <div class="pk-metric-value" style="color:{cl_col};font-size:22px">
            {perf['Cl']:+.4f}
          </div>
          <div class="pk-metric-sub">{cl_lbl}</div>
        </div></div>""", unsafe_allow_html=True)

    if 'show_scatter' not in st.session_state:
        st.session_state['show_scatter'] = False
    btn_lbl = ("▲ Hide Position in Design Space"
               if st.session_state['show_scatter']
               else "📍 Show Position in Design Space ▼")
    just_opened = False
    if st.button(btn_lbl, key='scatter_btn', use_container_width=True):
        st.session_state['show_scatter'] = not st.session_state['show_scatter']
        just_opened = st.session_state['show_scatter']

    if st.session_state['show_scatter']:
        st.markdown('<div id="pk-scatter-anchor"></div>', unsafe_allow_html=True)
        st.plotly_chart(cd_scatter_with_highlight(geo_df, perf['Cd'], style),
                        use_container_width=True)
        if just_opened:
            import streamlit.components.v1 as _sc
            _sc.html("""<script>
setTimeout(function(){
  var el = window.parent.document.getElementById('pk-scatter-anchor');
  if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
}, 300);
</script>""", height=0)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DESIGN (Inverse flow)
# ══════════════════════════════════════════════════════════════════════════════

with tab_design:
    d_left, d_right = st.columns([1, 2.3], gap="large")

    with d_left:
        st.markdown('<div class="pk-section">Starting Style</div>', unsafe_allow_html=True)
        start_style = st.radio(
            "start_style", ['fastback', 'notchback', 'estateback'],
            format_func=lambda s: s.title(),
            horizontal=True, label_visibility='collapsed',
            help=("Starting geometry for the optimiser.\n"
                  "The optimiser begins at this style's mean geometry and moves\n"
                  "toward your target Cd/Cl via gradient descent."),
        )

        st.markdown('<div class="pk-section">Targets</div>', unsafe_allow_html=True)
        target_cd = st.slider(
            "Target C_d", 0.18, 0.40, 0.23, 0.005, format="%.3f",
            help=("Desired drag coefficient.\n"
                  "Cd = F_drag / (½ρv²A)  — lower is better for fuel economy.\n"
                  "Typical cars: 0.25–0.35 · Sports cars: 0.18–0.25 · SUVs: 0.35–0.45"),
        )

        use_cl    = st.toggle(
            "Constrain lift (C_l)",
            help=("Enable to also optimise toward a target lift coefficient.\n"
                  "Cl = F_lift / (½ρv²A). Negative = downforce (improves grip)."),
        )
        target_cl = None
        if use_cl:
            target_cl = st.slider(
                "Target C_l", -0.20, 0.30, 0.05, 0.01, format="%.3f",
                help=("Desired lift coefficient.\n"
                      "Negative = downforce (pushes car onto road).\n"
                      "Positive = aerodynamic lift (reduces tyre contact force).\n"
                      "Highway stability prefers Cl ≤ 0."),
            )

        st.markdown('<div class="pk-section">Design Freedom (λ_prox)</div>',
                    unsafe_allow_html=True)
        lambda_prox = st.slider(
            "λ_prox", 0.5, 10.0, 2.0, 0.5, label_visibility='collapsed',
            help=("Proximity regularisation weight λ.\n"
                  "Loss term: λ‖θ − θ₀‖²  penalises drift from starting geometry.\n"
                  "• Low (0.5–1): large changes, may extrapolate beyond training data\n"
                  "• Balanced (2.0): default — plausible changes within training range\n"
                  "• High (5–10): tiny nudges, very conservative"),
        )
        st.caption("← Aggressive · Balanced (2.0) · Conservative →")

        all_styles = st.checkbox(
            "Try all 3 starting styles",
            help="Runs the optimiser from Fastback, Notchback, and Estateback starting points, then shows all three results so you can pick the best.",
        )
        st.markdown("")
        run_btn = st.button("🎯  Suggest Geometry", type="primary",
                            use_container_width=True)

    # Show starting geometry preview on the right before run
    with d_right:
        preview_theta  = dataset.style_mean_theta(start_style)
        preview_params = dataset.theta_to_named_params(preview_theta)
        st.markdown('<div class="pk-section">Starting Geometry Preview</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            draw_car_side(preview_params, color=GREY, fill_opacity=0.10,
                          name=start_style.title(), show_dims=True),
            use_container_width=True,
        )

    if run_btn:
        from koopman.inverse_design import suggest_geometry, batch_suggest
        with st.spinner("Optimising geometry…"):
            if all_styles:
                results = batch_suggest(
                    model, dataset, target_cd, target_cl,
                    n_restarts=3, lambda_prox=lambda_prox, device='cpu',
                )
                st.session_state['design_result']  = results[0]
                st.session_state['design_results'] = results
            else:
                r = suggest_geometry(
                    model, dataset, target_cd, target_cl,
                    start_style=start_style,
                    lambda_prox=lambda_prox,
                    device='cpu', verbose=False,
                )
                st.session_state['design_result']  = r
                st.session_state['design_results'] = [r]
        st.session_state['design_just_ran'] = True

    if 'design_result' in st.session_state:
        st.divider()
        st.markdown('<div id="pk-results-anchor"></div>', unsafe_allow_html=True)
        _render_design_result(
            st.session_state['design_result'],
            st.session_state.get('design_results', []),
        )
        if st.session_state.pop('design_just_ran', False):
            import streamlit.components.v1 as _components
            _components.html("""<script>
setTimeout(function(){
  var el = window.parent.document.getElementById('pk-results-anchor');
  if (el) {
    el.scrollIntoView({behavior: 'smooth', block: 'start'});
  } else {
    var m = window.parent.document.querySelector('[data-testid="stMain"]')
         || window.parent.document.querySelector('section.main');
    if (m) m.scrollBy({top: 480, behavior: 'smooth'});
  }
}, 400);
</script>""", height=0)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════

with tab_explore:
    _render_explore(geo_df)
