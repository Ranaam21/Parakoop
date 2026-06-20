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
    cd_gauge, cl_gauge, cd_scatter_with_highlight, draw_car_side, draw_car_comparison,
    draw_car_iso, draw_car_iso_comparison, draw_front_comparison,
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
  .block-container {{ padding-top: 0.8rem !important; max-width: 100% !important; padding-bottom: 0.1rem !important; }}

  /* Hide scrollbar (visual cleanliness) while keeping content accessible */
  [data-testid="stMain"] {{ scrollbar-width: none; -ms-overflow-style: none; }}
  [data-testid="stMain"]::-webkit-scrollbar {{ display: none; }}

  /* Tighten vertical gaps between Streamlit elements */
  [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stVerticalBlock"] > div {{ gap: 0 !important; }}
  div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {{ gap: 0 !important; }}
  .element-container {{ margin-bottom: 0 !important; }}
  [data-testid="stSlider"] {{ padding-top: 3px !important; padding-bottom: 3px !important; }}
  [data-testid="stRadio"] {{ padding-top: 1px !important; padding-bottom: 1px !important; }}
  [data-testid="stCheckbox"] {{ padding-top: 1px !important; padding-bottom: 1px !important; }}
  [data-testid="stToggle"] {{ padding-top: 1px !important; padding-bottom: 1px !important; }}

  .pk-header {{
    display: flex; align-items: baseline; gap: 14px;
    padding: 2px 0 5px 0; border-bottom: 1px solid #21262d; margin-bottom: 5px;
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
    font-size: 10px; font-weight: 600; color: {GREY};
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 4px 0 2px 0; padding-bottom: 2px; border-bottom: 1px solid #21262d;
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

  /* ── Radio: selected = white bold, unselected = visible grey ── */
  [data-testid="stRadio"] label {{
    color: #adbac7 !important; font-size: 13px !important;
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

  /* ── Help icon: ? → italic i  (widget labels + metric labels, NOT dataframe headers) ── */
  label [data-testid="stTooltipHoverTarget"] svg,
  [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] svg {{ display: none !important; }}
  label [data-testid="stTooltipHoverTarget"],
  [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] {{
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
  label [data-testid="stTooltipHoverTarget"]::after,
  [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"]::after {{
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

  /* .pk-tip base */
  .pk-tip {{ display: block; }}

  /* ── Guardrail badges: equal fixed size at rest, expand on hover ── */
  .gr-badge {{
    max-height: 62px !important;
    overflow: hidden !important;
    transition: max-height 0.25s ease;
  }}
  .gr-badge:hover {{
    max-height: 320px !important;
  }}
  .gr-info {{ display: none; }}
  .gr-badge:hover .gr-info {{ display: block !important; }}

  /* ── Compact geometry metrics ── */
  [data-testid="stMetric"] {{
    padding: 3px 8px !important;
    background: {PANEL} !important;
    border-radius: 5px !important;
    margin-bottom: 2px !important;
  }}
  [data-testid="stMetricLabel"] p {{
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: #adbac7 !important;
    font-weight: 600 !important;
  }}
  [data-testid="stMetricValue"] > div {{
    font-size: 12px !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    color: {WHITE} !important;
  }}
  [data-testid="stMetricDelta"] span {{
    font-size: 10px !important;
    color: #8b949e !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── JS floating tooltip — event delegation, survives DOM re-renders ──
import streamlit.components.v1 as _v1
_v1.html("""<script>
(function(){
  var doc = window.parent.document;

  // One shared floating tip appended to parent body
  var tip = doc.getElementById('pk-float-tip');
  if (!tip) {
    tip = doc.createElement('div');
    tip.id = 'pk-float-tip';
    tip.style.cssText = [
      'display:none','position:fixed','background:#21262d','color:#e6edf3',
      'border:1px solid #30363d','border-radius:8px','padding:10px 14px',
      'font-size:12px','white-space:pre-wrap','min-width:220px','max-width:340px',
      'z-index:2147483647','pointer-events:none','line-height:1.6',
      'box-shadow:0 4px 18px rgba(0,0,0,0.6)'
    ].join(';');
    doc.body.appendChild(tip);
  }

  function decode(s) {
    return s.replace(/&#10;/g,'\n').replace(/&quot;/g,'"')
            .replace(/&amp;/g,'&').replace(/&#39;/g,"'")
            .replace(/&lt;/g,'<').replace(/&gt;/g,'>');
  }

  function position(e) {
    var tw = tip.offsetWidth || 280, th = tip.offsetHeight || 120;
    var x = e.clientX + 14;
    var y = e.clientY - th - 10;
    if (y < 4) y = e.clientY + 14;
    if (x + tw > window.parent.innerWidth - 8) x = e.clientX - tw - 14;
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }

  // Walk up from target to find the nearest ancestor with data-tip
  function findTipEl(target) {
    var el = target;
    while (el && el !== doc.body) {
      if (el.hasAttribute && el.hasAttribute('data-tip')) return el;
      el = el.parentElement;
    }
    return null;
  }

  var current = null;

  // Capture-phase mouseover: fires before child elements, works with delegated structure
  doc.body.addEventListener('mouseover', function(e) {
    var el = findTipEl(e.target);
    if (el) {
      current = el;
      tip.textContent = decode(el.getAttribute('data-tip') || '');
      tip.style.display = 'block';
      position(e);
    } else {
      current = null;
      tip.style.display = 'none';
    }
  }, true);

  doc.body.addEventListener('mousemove', function(e) {
    if (current) position(e);
  }, true);

  doc.body.addEventListener('mouseout', function(e) {
    if (!current) return;
    var rel = e.relatedTarget;
    if (!rel || !current.contains(rel)) {
      current = null;
      tip.style.display = 'none';
    }
  }, true);
})();
</script>""", height=1, scrolling=False)

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
    "Koopman dim K": ("Koopman observable space dimension K = 256.\n"
                      "The model lifts the 8-dim geometry vector θ into a\n"
                      "K-dimensional space where aerodynamic evolution is\n"
                      "approximately linear: ψ(t+1) = A(θ) ψ(t).\n"
                      "A(θ) is parametrised as a rank-32 low-rank matrix\n"
                      "A(θ) = U Σ Vᵀ, keeping only 32 singular values for\n"
                      "efficiency while capturing the dominant flow modes."),
    "DrivAerNet MAE": ("Mean Absolute Error on the DrivAerNet dataset.\n"
                       "MAE = mean |Cd_pred − Cd_CFD| over 1,163 designs.\n"
                       "0.0087 means predictions are within ~0.009 Cd units\n"
                       "of high-fidelity CFD on average — well within\n"
                       "engineering design tolerance (±0.02)."),
    "AhmedML MAE":   ("Mean Absolute Error on 74 held-out AhmedML samples.\n"
                      "Ahmed body is a simplified bluff body used as a CFD\n"
                      "benchmark; slant angle drives large Cd variation.\n"
                      "MAE = 0.0404 Cd units on this out-of-distribution set."),
    "Training samples": ("Total CFD designs used to train the Koopman model:\n"
                         "• 9,163 DrivAerNet parametric car simulations\n"
                         "• 457 AhmedML Ahmed body configurations\n"
                         "= 9,620 samples spanning fastback / notchback /\n"
                         "  estateback styles across a wide geometry range."),
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
           "Automotive design target range: 0.18 – 0.35.\n"
           "Below 0.18 = suspiciously low (model extrapolation risk).\n"
           "Above 0.35 = above typical production-car target (SUV / high-drag).\n"
           "Physics allows up to ~0.60 for a car-like shape."),
}

_GEOM_TIPS = {
    'Style':      ("Body style — determines the rear-end architecture:\n"
                   "• Fastback    — continuous sloped roofline flowing into the tail\n"
                   "• Notchback   — separate boot/trunk lid with upright rear glass\n"
                   "• Estateback  — wagon/estate tail (nearly vertical rear panel)"),
    'Rear slant': ("α — rear body slope angle measured from horizontal.\n"
                   "Higher angle = more aggressive fastback taper.\n"
                   "DrivAerNet dataset range: 0° – 21.4°.\n"
                   "Strongly influences Cd: higher slant often reduces drag in fastbacks."),
    'Height':     ("H — overall car height, ground plane to roof peak (mm).\n"
                   "Taller cars have a larger frontal area → tend toward higher Cd.\n"
                   "DrivAerNet range: 1,210 – 1,753 mm."),
    'Width':      ("W — maximum body width at the widest cross-section (mm).\n"
                   "Shown as the diagonal W dimension line across the roof in the isometric view.\n"
                   "Contributes to frontal area and side-wind sensitivity.\n"
                   "DrivAerNet range: ~1,400 – 2,200 mm."),
    'Cabin frac': ("Cabin fraction — length of passenger compartment as a fraction\n"
                   "of total body length. Controls where the C-pillar (rear of roofline) sits.\n"
                   "Higher = longer cabin, shorter rear overhang."),
    'Length':     ("L — Reference body length (mm). Overall car length from front bumper to tail.\n"
                   "DrivAerNet style defaults: Fastback ~4850 mm · Notchback ~4780 mm · Estateback ~4800 mm.\n"
                   "Affects the height/length ratio (h/L) fed to the Koopman model — longer car → lower h/L → lower predicted Cd."),
    'Cd':         ("Drag Coefficient   Cd = F_drag / (½ ρ U∞² A)\n"
                   "Dimensionless aerodynamic drag. Lower is better for fuel economy.\n"
                   "U∞ = freestream wind speed  |  A = frontal projected area.\n"
                   "Sports car: 0.18–0.25  ·  Sedan: 0.25–0.35  ·  SUV: 0.35–0.45"),
    'Cl':         ("Lift Coefficient   Cl = F_lift / (½ ρ U∞² A)\n"
                   "Dimensionless aerodynamic lift/downforce.\n"
                   "Negative = downforce (pushes car into road → better tyre grip).\n"
                   "Positive = aerodynamic lift (reduces tyre contact force).\n"
                   "Highway stability prefers Cl ≤ 0."),
}


def _render_guardrails(guardrails: dict) -> None:
    gr_cols = st.columns(3)
    for col, (name, g) in zip(gr_cols, guardrails.items()):
        with col:
            color  = GREEN if g['passed'] else RED
            sym    = '✓' if g['passed'] else '✗'
            status = 'Pass' if g['passed'] else 'Fail'
            note   = g.get('note', '')
            tip    = _GR_TIPS.get(name, '').replace('\n', '<br>')
            st.markdown(f'''
<div class="gr-badge" style="
    background:#21262d; border-radius:6px; padding:5px 9px;
    border-top:3px solid {color}; margin-bottom:4px; min-height:58px">
  <div style="font-size:10px; color:#adbac7; text-transform:uppercase;
              letter-spacing:0.6px; font-weight:600; margin-bottom:3px">
    {name}
    <span style="font-size:8px; color:#6e7681; border:1px solid #6e7681;
                 border-radius:50%; padding:0 2.5px; font-style:italic;
                 margin-left:3px; vertical-align:middle">i</span>
  </div>
  <div style="font-size:13px; font-weight:700; color:{color}; line-height:1.3">
    {sym}&nbsp;{status}
  </div>
  <div style="font-size:10px; color:#8b949e; margin-top:2px">{note}</div>
  <div class="gr-info" style="
      font-size:10px; color:#c9d1d9; background:rgba(13,17,23,0.85);
      border-radius:4px; padding:5px 7px; margin-top:6px;
      border-left:2px solid {color}; line-height:1.5">
    {tip}
  </div>
</div>''', unsafe_allow_html=True)


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

    # ── Physics Guardrails — full-width row so tooltips aren't clipped ──────────
    st.markdown('<div class="pk-section">Physics Guardrails</div>',
                unsafe_allow_html=True)
    _render_guardrails(result['guardrails'])

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

    # ── Delta table — full width ──────────────────────────────────────────────
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
        _gr_warn = st.empty()   # filled below if any guardrail fails
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

        _STYLE_L_DEFAULT = {'fastback': 4850, 'notchback': 4780, 'estateback': 4800}

        st.markdown('<div class="pk-section">Geometry</div>', unsafe_allow_html=True)
        length = st.slider(
            "Length (mm)", 4200, 5500,
            _STYLE_L_DEFAULT.get(style, 4800), 10,
            help=("Overall car length from front bumper to tail in mm.\n"
                  "DrivAerNet style defaults: Fastback 4850 · Notchback 4780 · Estateback 4800.\n"
                  "Longer car → lower height/length ratio → generally lower Cd."),
        )
        height = st.slider(
            "Height (mm)", 1210, 1753,
            int(np.clip(mean_params['height_mm'], 1210, 1753)), 5,
            help=("Overall car height from ground to roof in mm.\n"
                  "Taller cars have greater frontal area → generally higher Cd.\n"
                  "DrivAerNet range: 1,210 – 1,753 mm"),
        )
        _w_default = int(np.clip(
            mean_params['width_height_ratio'] * mean_params['height_mm'], 1400, 2200
        ))
        width = st.slider(
            "Width (mm)", 1400, 2200, _w_default, 10,
            help=("Maximum car width at the widest cross-section (mm).\n"
                  "Shown as the diagonal W dimension line across the roof.\n"
                  "DrivAerNet range: ~1,400 – 2,200 mm."),
        )
        w_h = float(width) / float(height)
        slant = st.slider(
            "Rear slant angle (°)", 0.0, 21.4,
            float(np.clip(mean_params['rear_slant_deg'], 0.0, 21.4)), 0.1,
            help=("Angle of the rear body slope from horizontal.\n"
                  "Higher = more aggressive taper (fastback); lower = more upright.\n"
                  "DrivAerNet dataset range: 0° – 21.4°"),
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
                                 1.0 if detailed else 0.0,
                                 ref_length_mm=float(length))
    perf   = predict(model, theta)
    params = dataset.theta_to_named_params(theta)
    params['ref_length_mm'] = float(length)   # override with slider value
    params['width_mm']      = float(width)    # override: slider is direct mm, not ratio

    from koopman.inverse_design import check_physics_guardrails
    guardrails = check_physics_guardrails(params, perf['Cd'])

    # ── Guardrail warnings in the left slider panel ───────────────────────────
    _failed = [(n, g) for n, g in guardrails.items() if not g['passed']]
    if _failed:
        with _gr_warn.container():
            for name, g in _failed:
                if name == 'Eu':
                    st.warning(
                        f"**Eu / Cd = {g['value']:.4f}** is outside the valid range "
                        f"[{g['min']:.2f} – {g['max']:.2f}]. "
                        f"Predicted drag is {'too low' if g['value'] < g['min'] else 'too high'} "
                        f"for a production car — adjust rear slant or height."
                    )
                elif name == 'Re':
                    st.warning(
                        f"**Re = {g['value']:.2e}** is out of range "
                        f"[{g['min']:.0e} – {g['max']:.0e}]. "
                        f"Adjust car length via the Height slider."
                    )
                elif name == 'Ma':
                    st.warning(
                        f"**Ma = {g['value']:.3f}** exceeds the incompressible limit "
                        f"(< {g['max']:.2f}). Flow compressibility effects not modelled."
                    )

    with car_col:
        # Physics guardrails — compact badges ABOVE the car
        st.markdown('<div class="pk-section">Physics Guardrails</div>',
                    unsafe_allow_html=True)
        _render_guardrails(guardrails)
        # 3D isometric car (shows width, height, style, slant all in one shape)
        st.plotly_chart(draw_car_iso(params), use_container_width=True)
        # Compact geometry strip — Cd/Cl shown in right panel; Length is read-only
        st.markdown('<div class="pk-section">Current Geometry</div>',
                    unsafe_allow_html=True)
        live_rows = [
            ('Style',      params['style'].title()),
            ('Rear slant', f"{params['rear_slant_deg']:.1f} °"),
            ('Height',     f"{params['height_mm']:.0f} mm"),
            ('Width',      f"{params['width_mm']:.0f} mm"),
            ('Length',     f"{params['ref_length_mm']:.0f} mm"),
            ('Cabin frac', f"{params['cabin_frac']:.3f}"),
        ]
        _mc = st.columns(3)
        for i, (lbl, val) in enumerate(live_rows):
            with _mc[i % 3]:
                st.metric(label=lbl, value=val,
                          help=_GEOM_TIPS.get(lbl) or None)

    with right_col:
        st.plotly_chart(cd_gauge(perf['Cd']), use_container_width=True)
        st.plotly_chart(cl_gauge(perf['Cl']), use_container_width=True)

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
    d_left, d_right = st.columns([1, 2.5], gap="large")

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

        use_cl = st.toggle(
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
                      "Positive = aerodynamic lift.\n"
                      "Highway stability prefers Cl ≤ 0."),
            )

        st.markdown('<div class="pk-section">Design Freedom (λ_prox)</div>',
                    unsafe_allow_html=True)
        lambda_prox = st.slider(
            "λ_prox", 0.5, 10.0, 2.0, 0.5, label_visibility='collapsed',
            help=("Proximity regularisation weight λ.\n"
                  "Loss: λ‖θ − θ₀‖² penalises drift from starting geometry.\n"
                  "• Low (0.5–1): aggressive changes\n"
                  "• Balanced (2.0): default\n"
                  "• High (5–10): conservative nudges"),
        )
        st.caption("← Aggressive · Balanced · Conservative →")

        all_styles = st.checkbox(
            "Try all 3 starting styles",
            help="Runs from Fastback, Notchback, and Estateback — compare results.",
        )
        run_btn = st.button("🎯  Suggest Geometry", type="primary",
                            use_container_width=True)

        if 'design_result' in st.session_state:
            if st.button("✕  Clear result", use_container_width=True):
                del st.session_state['design_result']
                st.session_state.pop('design_results', None)
                st.rerun()

    # Run optimiser BEFORE d_right so session state is ready when it renders
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

    with d_right:
        if 'design_result' not in st.session_state:
            # ── Clean empty state ─────────────────────────────────────────────
            st.markdown("""
<div style="height:460px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:12px;">
  <div style="font-size:52px;opacity:0.12;line-height:1">🎯</div>
  <div style="font-size:13px;color:#6e7681;letter-spacing:0.3px">
    Set your target Cd and click <strong style="color:#adbac7">Suggest Geometry</strong>
  </div>
  <div style="font-size:11px;color:#484f58">
    The Koopman optimiser will suggest new car dimensions to reach your target.
  </div>
</div>""", unsafe_allow_html=True)

        else:
            result      = st.session_state['design_result']
            all_results = st.session_state.get('design_results', [])
            err = result['achieved_cd'] - result['target_cd']
            ok  = result['guardrails_ok']

            # ── Row 1: Style chips (batch) ────────────────────────────────────
            if len(all_results) > 1:
                chip_cols = st.columns(len(all_results))
                for i, (col, res) in enumerate(zip(chip_cols, all_results)):
                    is_active = (res is result)
                    sty   = res['start_params']['style'].title()
                    cd_i  = res['achieved_cd']
                    err_i = cd_i - res['target_cd']
                    bdr = f"2px solid {BLUE}" if is_active else "1px solid #30363d"
                    bg  = "rgba(30,136,229,0.10)" if is_active else PANEL
                    lc  = BLUE if is_active else WHITE
                    with col:
                        st.markdown(f"""<div style="background:{bg};border:{bdr};
border-radius:6px;padding:5px 8px;text-align:center;margin-bottom:2px">
  <div style="font-size:9px;color:#adbac7;text-transform:uppercase;
              letter-spacing:0.5px;font-weight:600">{sty}</div>
  <div style="font-size:13px;font-weight:700;color:{lc}">Cd {cd_i:.4f}</div>
  <div style="font-size:9px;color:#6e7681">err {err_i:+.4f}</div>
</div>""", unsafe_allow_html=True)
                        if not is_active:
                            if st.button("Select", key=f"dsb_{i}",
                                         use_container_width=True):
                                st.session_state['design_result'] = res
                                st.rerun()

            # ── Row 2: Physics guardrails — reuse same function as Predict tab ──
            _render_guardrails(result['guardrails'])

            # ── Row 3: Car (left) + stacked gauges (right) ───────────────────
            car_c, gauge_c = st.columns([3.2, 1.1], gap="small")

            with gauge_c:
                st.plotly_chart(
                    cd_gauge(result['achieved_cd'], target_cd=result['target_cd'], height=175),
                    use_container_width=True,
                )
                st.plotly_chart(
                    cl_gauge(result['achieved_cl'], height=155),
                    use_container_width=True,
                )

            with car_c:
                # View selector + legend on the same row
                leg_col, radio_col = st.columns([1, 1])
                with leg_col:
                    st.markdown("""
<div style="display:flex;gap:18px;align-items:center;
            font-size:10px;color:#adbac7;padding-top:4px">
  <span style="display:flex;align-items:center;gap:5px">
    <svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5"
      stroke="#26C6DA" stroke-width="2.2" stroke-dasharray="5,3"/></svg>
    Before
  </span>
  <span style="display:flex;align-items:center;gap:5px">
    <svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5"
      stroke="#1E88E5" stroke-width="2.5"/></svg>
    After
  </span>
</div>""", unsafe_allow_html=True)
                with radio_col:
                    view_mode = st.radio(
                        "view",
                        ["↔ Side", "⬛ Front", "◈ 3D"],
                        horizontal=True,
                        index=0,
                        label_visibility="collapsed",
                        key="design_view_mode",
                    )

                if view_mode == "↔ Side":
                    fig_cmp = draw_car_comparison(
                        result['start_params'], result['end_params'], result['delta'],
                        height=240,
                    )
                elif view_mode == "⬛ Front":
                    fig_cmp = draw_front_comparison(
                        result['start_params'], result['end_params'], result['delta'],
                        height=240,
                    )
                else:
                    fig_cmp = draw_car_iso_comparison(
                        result['start_params'], result['end_params'], result['delta'],
                        height=240,
                    )
                st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Row 4: Geometry Changes table (below car column only) ─────────
            if result['delta']:
                rows = []
                for param, (v0, v1, d) in sorted(
                    result['delta'].items(), key=lambda x: abs(x[1][2]), reverse=True
                ):
                    unit = '°' if 'deg' in param else ('mm' if '_mm' in param else '')
                    rows.append((param.replace('_', ' ').title(),
                                 f"{v0:.1f}{unit}", f"{v1:.1f}{unit}", d, f"{d:+.1f}{unit}"))

                body = ''
                for i, (label, bef, aft, d_raw, d_str) in enumerate(rows):
                    bg  = '#21262d' if i % 2 == 0 else '#1c2128'
                    dc  = GREEN if d_raw < 0 else (AMBER if d_raw > 0 else GREY)
                    body += (
                        f'<tr style="background:{bg}">'
                        f'<td style="padding:2px 8px;color:#c9d1d9">{label}</td>'
                        f'<td style="padding:2px 8px;color:#8b949e;text-align:right">{bef}</td>'
                        f'<td style="padding:2px 8px;color:{WHITE};font-weight:600;text-align:right">{aft}</td>'
                        f'<td style="padding:2px 8px;color:{dc};font-weight:700;text-align:right">{d_str}</td>'
                        f'</tr>'
                    )
                th = ('padding:2px 8px;font-size:9px;color:#8b949e;text-transform:uppercase;'
                      'letter-spacing:0.5px;font-weight:600;border-bottom:1px solid #30363d')
                st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:11px;
              font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin-top:2px">
  <thead style="background:#161b22">
    <tr>
      <th style="{th};text-align:left">Parameter</th>
      <th style="{th};text-align:right">Before</th>
      <th style="{th};text-align:right">After</th>
      <th style="{th};text-align:right">Δ</th>
    </tr>
  </thead>
  <tbody>{body}</tbody>
</table>""", unsafe_allow_html=True)
            else:
                st.caption("Model already near target — no significant changes.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════

with tab_explore:
    _render_explore(geo_df)
