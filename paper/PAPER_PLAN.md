# ParaKoop — Paper Writing Plan & Instructions

**Title:** From Performance to Geometry: A Parametric-Koopman Inverse-Design Engine
         for Quantum-Ready Automotive Aerodynamics
**Author:** Amit Rana
**Target:** arXiv preprint → conference / journal submission (AIAA / NeurIPS ML4Eng track)
**This file:** Single source of truth for all paper writing in this folder.
             Read this before writing any section. Update status as sections complete.

---

## 1. Files in This Folder

| File | Purpose |
|---|---|
| `PAPER_PLAN.md` | This file — structure, rules, status tracking |
| `00_abstract.md` | Abstract (≤ 250 words) |
| `01_introduction.md` | Introduction + paper roadmap |
| `02_related_work.md` | Related work (Koopman surrogates, inverse design, quantum ML) |
| `03_methodology.md` | Full methodology — the ParaKoop architecture |
| `04_datasets.md` | Dataset descriptions (DrivAerNet++, AhmedML) |
| `05_experiments.md` | All experiments, tables, figures |
| `06_discussion.md` | Discussion, limitations, future work |
| `07_conclusion.md` | Conclusion (≤ 200 words) |
| `08_references.bib` | BibTeX references |
| `figures/` | All figure source files (plots, diagrams) |

**Source documents to draw from (do NOT edit these, read-only reference):**
- `/Users/amit21/Desktop/Car_CFD/Paper4_Automotive_Koopman_HHL_ParaKoop_Scope.txt`
  → Architecture, core argument, section-by-section scope (the blueprint)
- `/Users/amit21/Desktop/Car_CFD/Paper4_ParaKoop_Design_Rationale_QA.md`
  → 15 Q&A design decisions → lift directly into Methods and Discussion
- `/Users/amit21/Desktop/Car_CFD/parakoop/` → code, results, model checkpoints

---

## 2. Paper Structure & Section Targets

### Section 0: Abstract `00_abstract.md` — STATUS: [ ] pending
- ≤ 250 words. One sentence each: problem / gap / method / key result / implication.
- Must contain: "parametric Koopman", "inverse design", "HHL", "κ", specific MAE numbers.
- Key numbers to include:
  - Cd MAE 0.01378 (vs GBR 0.01549, −11%)
  - AhmedML domain val MAE 0.0387 (74 held-out runs)
  - κ = 1.08–1.24, HHL speedup 14.7–16.9×, 15 qubits

### Section 1: Introduction `01_introduction.md` — STATUS: [ ] pending
- ~800 words. Hook → problem → gap → contribution list → roadmap.
- **Hook:** Each full-car CFD evaluation takes 6–48 hours. Industry runs 10–50 per cycle.
  The entire design loop is forward: given geometry, get drag. We run it backward.
- **Contributions** (bulleted, numbered, precise):
  1. ParaKoop: a geometry-conditioned ("parametric") Koopman operator A(θ) jointly
     trained on AhmedML (500 transient, multi-geometry Ahmed-body cases) and
     DrivAerNet++ (8,000 steady car shapes).
  2. All reported quantities — Cd, Cl, modal structure, HHL input system, and inverse
     design gradient — are structural readouts of one learned object A(θ). Nothing forks.
  3. Inverse design via gradient descent on the differentiable θ → A(θ) → performance
     chain: given target Cd/Cl, the system outputs real geometry changes (mm, degrees).
  4. HHL condition-number study across shape families: κ = 1.08–1.24 (with phi grounding),
     theoretical speedup 14.7–16.9×, 15 qubits.
  5. Physics guardrails (Re, Ma, Eu) baked into the inverse design loop.
  6. Open-source Streamlit demo and code.

### Section 2: Related Work `02_related_work.md` — STATUS: [ ] pending
- ~600 words. Three subsections:
  **2.1 Koopman operator surrogates for fluid mechanics**
    - Brunton et al. 2021 (data-driven Koopman review)
    - Lusch et al. 2018 (deep Koopman)
    - What ParaKoop adds: geometry-CONDITIONING (A is a function of θ, not a single operator)
  **2.2 Surrogate-based inverse / aerodynamic design**
    - DrivAerNet (Elrefaie et al. 2024) — largest car surrogate dataset (we use it)
    - AhmedML (Ashton et al. 2024) — transient Ahmed body (we use it)
    - GBR / GNN surrogates: forward only, no inverse mechanism
  **2.3 Quantum linear solvers in engineering**
    - HHL (Harrow, Hassidim, Lloyd 2009)
    - Limitations (state prep, readout) — acknowledge honestly
    - What we add: first empirical κ study across a large automotive shape family

### Section 3: Methodology `03_methodology.md` — STATUS: [ ] pending
- ~2000 words. **Draw heavily from Q&A doc — it is already Methods-ready.**
- Subsections:

  **3.1 Structured Geometry Representation θ** (Scope §5.1, Q&A Q4, Q5)
    - 8-dim unified theta; ratios (h/L, w/h) for scale invariance across datasets
    - Why ratios, not raw mm: Q&A Q5 reasoning
    - One-hot style encoding + continuous dims

  **3.2 Parametric Koopman Operator A(θ)** (Scope §5.2–5.3, Q&A Q1, Q2, Q6)
    - z_net: θ → z* (lifted state), perf_head → (Cd, Cl)
    - A(θ) as a function of geometry (not temporal): Q&A Q1 "why not temporal"
    - Joint training loss: Cd MSE (all 9,620) + masked Cl MSE (499 AhmedML) + phi grounding
    - Proximity regularisation λ‖θ−θ₀‖² in inverse design: Q&A Q13

  **3.3 Performance Readout** (Scope §5.4, Q&A Q7, Q8)
    - Fixed-point: A(θ)·z* ≈ z* → Cd, Cl
    - Modal: eigenvalues of A(θ) → Strouhal-adjacent frequency modes
    - Wake recirculation: new metric from field structure
    - κ(A(θ)): triple duty (flow regime / HHL quantum-readiness / gradient trust)

  **3.4 HHL Quantum Interface** (Scope §5.5, Q&A Q9, Q10)
    - A_direct formulation: A(θ)·z = b(θ)
    - κ computation across all shapes
    - Phi grounding → κ reduction from 1.72–2.97 → 1.08–1.24
    - Theoretical speedup O(κ² log N); 15 qubits; Qiskit simulation

  **3.5 Inverse Design** (Scope §5.6, Q&A Q11, Q12, Q13)
    - Continuous relaxation (no Gumbel/softmax): Q&A Q11
    - AdamW on θ (300 steps, lr=5e-3)
    - Loss = (Cd_pred − Cd_target)² + λ_Cl(Cl_pred − Cl_target)² + λ_prox‖θ−θ₀‖²
    - Output: real mm/degree suggestions, not flag lookups
    - Batch mode: 3 starting styles, pick by |Cd error|

  **3.6 Physics Guardrails** (Scope §5.9, Q&A Q5)
    - Re = U∞·L/ν ∈ [3×10⁶, 3×10⁷]
    - Ma = U∞/c < 0.30 (incompressible RANS validity)
    - Eu ≈ Cd ∈ [0.18, 0.35] (automotive design range)
    - Reject-and-resample at post-optimisation check

### Section 4: Datasets `04_datasets.md` — STATUS: [ ] pending
- ~400 words.
  **4.1 DrivAerNet++**
    - 8,121 designs (1,162 STL + 6,959 parametric CSV), Cd only, steady RANS
    - 8 variant classes (F/N/E × S/D × WW/WWC × WM/no-mirrors)
    - Split: 9,163 training, 1,163 STL-derived ground-truth test

  **4.2 AhmedML**
    - 499 transient RANS-LES cases, 500 geometric variants, Cd + Cl
    - 76 cases with full VTU flow fields (used for phi grounding)
    - Slant: 6.3°–70.4°, Cd: 0.183–0.537, Cl: −0.219–0.709
    - 74-case held-out set (domain val, never seen in training)

  **4.3 Unified Loading**
    - Deduplication: 214 designs in both STL and CSV → STL wins
    - Scale invariance: ratios h/L, w/h work across DrivAerNet (4700mm) and AhmedML (1000mm)

### Section 5: Experiments & Results `05_experiments.md` — STATUS: [ ] pending
- ~1500 words. All numbers already computed — just write them up.

  **5.1 Forward Prediction (Cd, Cl)**
  | Model | Val Cd MAE | AhmedML Domain Val MAE | Cl MAE |
  |---|---|---|---|
  | GBR baseline | 0.01549 ± 0.00649 (CV) | — | — |
  | ParaKoop (ours) | **0.01378** (mixed val) | 0.0387 (74 held-out) | — |
  - GBR trained on 1,163 STL designs only; ParaKoop on full 9,620
  - ParaKoop −11% MAE vs GBR on common ground

  **5.2 Eigenspectrum & Strouhal Analysis**
    - Before phi grounding: κ = 1.72–2.97
    - After phi grounding: κ = 1.08–1.24 (−63% mean reduction)
    - Strouhal eigenvalues: near-zero (correct — A(θ) is geometric, not temporal;
      geometric conditioning ≠ time-stepping; cite Q&A §14 reasoning)
    - Phi grounding validates via κ reduction, not Strouhal frequencies

  **5.3 HHL Condition Number Study**
    - κ = 1.08–1.24 → theoretical speedup 14.7–16.9× vs classical
    - 15 qubits for full system
    - κ distribution across fastback / notchback / estateback families (figure)
    - A(θ).z = b simulation result: matches forward Cd to within 0.3%

  **5.4 Inverse Design Quality**
    - Show 3 example suggestions (fastback, notchback, estateback starts)
    - Target Cd = 0.23; achieved 0.241 ± 0.008 across styles
    - Typical geometry changes: rear_slant −8 to −12°, height −60 to −90mm
    - All 3 pass Re/Ma/Eu guardrails
    - Proximity regularisation effect: λ_prox=2 vs λ_prox=0 comparison

  **5.5 CFD Domain Validation (AhmedML Hold-out)** — STATUS: [x] COMPLETE (no new CFD needed)
    - AhmedML data IS OpenFOAM k-ω SST RANS — the held-out 74 cases are already CFD ground truth
    - Model predictions vs AhmedML CFD → Cd MAE = 0.0387 on 74 held-out runs
    - Script: `scripts/cfd_validate.py --mode ahmed` (reproduces seed=42 train/val split)
    - Cross-scale inverse design check: `--mode cross` finds nearest AhmedML run in θ-space
      for each inverse design output; reports |ΔCd| between prediction and CFD at nearest run
    - FRAMING: report as "CFD domain validation using AhmedML held-out set (OpenFOAM RANS)"
      NOT "we ran new OpenFOAM". AhmedML is already published, peer-reviewed CFD data.
    - No new OpenFOAM runs needed; the dataset itself provides the validation layer.

  **5.6 Streamlit Demo**
    - Figure: screenshot of predict + design tabs
    - Cite GitHub repo

### Section 6: Discussion `06_discussion.md` — STATUS: [ ] pending
- ~600 words. Draw from Q&A doc — decisions are pre-written.
  - Why one operator, not two tracks (Q1)
  - Why phi grounding validates via κ, not Strouhal (Q14)
  - Why proximity regularisation matters for inverse design safety (Q13)
  - HHL limitations: state prep, readout overhead; we report κ only (Q9/Q10)
  - What "quantum-ready" means: κ is the precondition, not a claim of quantum advantage now

### Section 7: Conclusion `07_conclusion.md` — STATUS: [ ] pending
- ≤ 200 words. Restate core claim + 3 key numbers + one open problem.

---

## 2.5 Instructions from `CFD Paper Instructions.docx` (mandatory)

These override defaults where they conflict:

1. **Language:** Simple, accessible to non-experts. Provide intuitions and real-world analogies. Avoid jargon without explanation.
2. **Acronyms:** Always give full form on first use — e.g. "Gradient-Boosted Regressor (GBR)".
3. **Equations:** Every variable in every equation must be defined inline the first time it appears. No dangling symbols.
4. **Figures:** 500+ DPI; no overlapping text; proper colour contrast (text must not merge with background); consistent colour schema across all charts.
5. **Architecture diagram:** Include a full pipeline figure showing both forward (θ→A(θ)→Cd/Cl) and inverse (Cd*→θ) flows.
6. **Citations:** Inline numbered [1], [2]… Elsevier/IEEE/Springer standard. Every number in the paper must cite its source.
7. **Novelty:** Explicitly call out what is new in a dedicated "Novelty" subsection or table. Do not assume the reader will infer it.
8. **Dimensionless quantities:** Re, Ma, Eu — explain each one: what it is, how it is computed here, and WHY it is the right guardrail for this problem.
9. **Industry use cases:** Section or box covering aerospace, automotive, semiconductor, space applications.
10. **IP protection:** Describe the architecture at the level of published methods. Do not expose layer sizes, training hyperparameters, or checkpoint details that would let someone reproduce without re-training.
11. **Formatting:** Sub-headings bold. Any text before a colon ":" in a bullet point is bold; rest is normal weight. Page numbers in footer.
12. **Author details:** Copy from QuantumRAG paper at `/Users/amit21/Desktop/Super Intelligence/QuantumAI/paper/QuantumRAG_Paper_v5.docx`.
13. **Public data:** Explicitly name and describe DrivAerNet++, AhmedML, and OpenFOAM as public datasets.
14. **Tone:** State-of-the-art and impressive, but never intimidating. A smart non-specialist should be able to follow.

## 3. Writing Rules (follow strictly)

1. **No hedging in the abstract or intro.** State what was done, not "we try to" or "we hope to."
2. **Every number has a reference.** If a number appears in text, cite the experiment/table it comes from.
3. **Q&A doc is pre-written methods material.** Q1–Q15 in `Paper4_ParaKoop_Design_Rationale_QA.md`
   map directly to paper sections — rephrase from Q&A voice to paper voice, do not reargue from scratch.
4. **Distinguish forward vs inverse clearly.** Forward = given θ, predict Cd/Cl. Inverse = given target Cd/Cl, output θ.
5. **Never call HHL an "advantage" without the κ caveat.** Always say "theoretical speedup conditioned on
   efficient state preparation" and cite the state-prep overhead honestly.
6. **κ is the load-bearing connector.** Every time κ appears, name all three roles: (1) flow regime indicator,
   (2) HHL quantum-readiness gate, (3) gradient-refinement trust metric. Don't split these.
7. **Geometry outputs are specific.** Inverse design results must quote real numbers (mm, degrees), not
   "improved geometry" or "better aerodynamic shape."
8. **Write sections in this order:** Abstract last. Order for drafting:
   04 → 03 → 05 → 01 → 02 → 06 → 07 → 00
   (Datasets first — they're factual. Methods next. Experiments next.
    Intro, related work, discussion after you know what the results say. Abstract last.)

---

## 4. Key Numbers Reference Card

| Quantity | Value | Source |
|---|---|---|
| Training samples | 9,620 (9,163 DrivAerNet + 457 AhmedML) | `data_pipeline/unified_loader.py` |
| ParaKoop val Cd MAE | **0.01401** | Checkpoint `best_val_loss` (run 2026-06-20) |
| GBR baseline CV-MAE | 0.01549 ± 0.00649 | `koopman/geometry_predictor.py` |
| AhmedML domain val Cd MAE | 0.0378 (mean), 0.0364 (median), 0.1233 (max) | `scripts/cfd_validate.py --mode ahmed` |
| AhmedML domain val Cl MAE | 0.1045 (mean), 0.1019 (median), 0.3520 (max) | same |
| κ fastback | 1.121 | `scripts/analyze_hhl.py` |
| κ notchback | 1.028 | same |
| κ estateback | 1.234 | same |
| κ range (all styles) | 1.03–1.23, mean 1.13 | same |
| HHL speedup | 14.8× (estateback) – 17.8× (notchback) | same |
| Strouhal St_max | 0.0002 (all styles) — near-zero, confirming geometric (not temporal) operator | same |
| Qubits required | 15 | `koopman/hhl_interface.py` |
| VTU runs used (phi) | 75 (of 76) | `data_pipeline/unified_loader.py` |
| Theta dimensionality | 8 | `data_pipeline/unified_loader.py` |
| Koopman lifting dim K | 128 (rank-16 A) — trained checkpoint | `koopman/hhl_interface.py`, `scripts/analyze_hhl.py` |
| Inverse design steps | 300 (AdamW, lr=5e-3) | `koopman/inverse_design.py` |
| Proximity reg default | λ_prox = 2.0 | `koopman/inverse_design.py` |
| Guardrails | Re∈[3e6,3e7], Ma<0.30, Eu∈[0.18,0.35] | `koopman/inverse_design.py` |
| CFD domain val | AhmedML 74 hold-out = OpenFOAM RANS ground truth | `scripts/cfd_validate.py --mode ahmed` |

---

## 5. Section Status Tracker

| Section | File | Status | Blocker |
|---|---|---|---|
| Datasets | `04_datasets.md` | [x] complete | — |
| Methodology | `03_methodology.md` | [x] complete | Note: §3.7 HHL speedup formula needs verify vs analyze_hhl.py |
| Experiments | `05_experiments.md` | [x] complete | All gaps filled from script runs 2026-06-20 |
| Introduction | `01_introduction.md` | [x] complete | — |
| Related Work | `02_related_work.md` | [x] complete | — |
| Discussion | `06_discussion.md` | [x] complete | — |
| Conclusion | `07_conclusion.md` | [x] complete | — |
| Abstract | `00_abstract.md` | [x] complete | — |

**All sections complete. Next actions:** (1) Read QuantumRAG paper for author details → fill §01 GitHub URL and author block. (2) Compile 08_references.bib from all [n] placeholders. (3) Assemble final PDF — figures, architecture diagram, formatting pass per CFD Paper Instructions.docx.
