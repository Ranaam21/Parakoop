# Section 4: Datasets

<!-- Draft status: complete. Numbers verified against data_pipeline/unified_loader.py -->
<!-- Writing order per PAPER_PLAN: this section is written first (no OpenFOAM dependency) -->

---

## 4. Datasets

ParaKoop is trained on two publicly available automotive aerodynamics datasets that differ in
vehicle scale, geometric complexity, and available quantities of interest. A shared
scale-invariant geometry representation (Section 3.1) allows joint training across both without
domain-specific preprocessing.

### 4.1 DrivAerNet++

DrivAerNet++ [Elrefaie et al., 2024] is the largest open computational automotive aerodynamics
dataset, providing steady-state RANS drag coefficients for a parametric family of full-scale
passenger car geometries. Simulations were conducted at U∞ = 38.89 m/s (Re ≈ 5×10⁶) using
OpenFOAM with a standard k-ω SST turbulence closure.

**Geometry.** Three body archetypes are represented — fastback (F), notchback (N), and
estateback (E) — each further varied across simplified/detailed underbody (S/D), with/without
side mirrors (WM), and closed/open wheel wells (WW/WWC). This produces eight configuration
families. Reference vehicle length is approximately 4700–5100 mm (sedan scale).

**Data modalities.** The dataset ships in two complementary forms:

- *Parametric CSV subset (~8,000 designs):* drag coefficients paired with discrete configuration
  flags (body style, mesh detail level, mirror, wheel). Raw geometry dimensions (height, width,
  rear slant angle) are not provided — only the config label.

- *STL subset (1,163 designs):* a curated subset for which surface meshes are available.
  We extract five continuous geometry parameters per design — overall length, height, width,
  rear slant angle (degrees), and cabin length fraction — using a custom geometry feature
  extractor (`data_pipeline/geometry_features.csv`).

When a design ID appears in both modalities, the STL-derived geometry takes priority
(deduplication: ~214 designs promoted from CSV to STL).

**Cd range.** 0.19–0.44 across all configurations.

### 4.2 AhmedML

AhmedML [Ashton et al., 2024] provides transient RANS-LES simulations of the Ahmed bluff body,
the canonical simple-geometry surrogate for production car aerodynamics research. Unlike
DrivAerNet++, AhmedML supplies both drag and lift coefficients, making it essential for training
the Cl prediction head.

**Geometry.** 499 unique shape variants spanning: rear slant angle 6.3°–70.4°, body height
200–388 mm, body width 389–489 mm, with vehicle length held near 1,044 mm. At highway speed
(U∞ = 40 m/s), the Ahmed body Reynolds number is Re ≈ 2.7×10⁶ — one order of magnitude below
DrivAerNet++ but within the turbulent automotive regime.

**Quantities of interest.** Cd: 0.183–0.537; Cl: −0.219–0.709. Both are available for all 499
cases. Note the bimodal Cd distribution near the slant critical angle (~30°), a known feature
of Ahmed body aerodynamics.

**VTU flow fields.** 76 cases provide full volumetric OpenFOAM output (velocity, pressure, and
turbulence fields) as VTU files. These are used exclusively for phi-supervised grounding of the
Koopman eigenfunctions (Section 3.3); the remaining 423 cases contribute only (θ, Cd, Cl) tuples
to the joint loss.

**Domain validation split.** 74 cases (∼15% of AhmedML) are held out entirely before training
begins. These constitute the domain validation set reported in Section 5.1 — the model never
sees their geometry or coefficients during training or model selection.

### 4.3 Unified Geometry Representation

Merging datasets of different vehicle scales requires a representation that is invariant to
absolute size. We use an 8-dimensional unified theta vector:

```
θ = [style_fastback, style_notchback, style_estateback,   ← one-hot (3 dims)
     height/length,                                         ← aspect ratio
     width/height,                                          ← cross-section ratio
     rear_slant_deg / 90,                                   ← normalised slant
     cabin_frac,                                            ← cabin/body proportion
     detail_flag]                                           ← 0=simplified, 0.5=Ahmed, 1=detailed
```

Ratio features (h/L, w/h) capture shape independently of scale: a DrivAerNet fastback at
4850 mm and an Ahmed body at 1044 mm occupy overlapping regions of the h/L dimension
(≈ 0.27 and ≈ 0.26, respectively). Raw millimetre values would conflate scale with shape and
prevent cross-dataset transfer.

**AhmedML mapping.** Ahmed bodies are treated as the fastback archetype (forward-swept slant,
no notch or estate geometry) with detail_flag = 0.5, distinct from simplified DrivAerNet
configurations (0.0) and detailed STL meshes (1.0).

**Physical bounds.** Theta is bounded element-wise: ratios within physically observed ranges
(h/L ∈ [0.20, 0.45], w/h ∈ [0.50, 1.80]), slant in [0°, 49.5°], cabin fraction in [0.25, 0.70].
These bounds are enforced during inverse design gradient clamping.

**Dataset statistics.** Figure 5 shows the training sample composition by source and the
overlap in Cd distributions between DrivAerNet++ and AhmedML.

| Source | Designs | Cd (range) | Cl available | Geometry source |
|---|---|---|---|---|
| DrivAerNet++ STL | 1,163 | 0.19–0.44 | No | Real extracted (5 dims) |
| DrivAerNet++ CSV | ~8,000 | 0.19–0.44 | No | Config-conditional imputed |
| AhmedML (training) | 425 | 0.183–0.537 | Yes | CSV params (5 dims) |
| **Total training** | **~9,588** | 0.183–0.537 | 425/~9,588 | — |
| AhmedML (held-out) | 74 | 0.183–0.537 | Yes | Domain val only |

> **Note on DrivAerNet++ training count:** the exact training count from the CSV subset
> varies based on available design IDs after deduplication; the model checkpoint's training
> summary reports a combined DrivAerNet training count of 9,163. Exact per-source breakdown
> is logged in `data_pipeline/unified_loader.py` (`UnifiedDataset.summary()`).

---

