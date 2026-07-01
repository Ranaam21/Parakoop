# Section 6: Discussion

<!-- Draft status: complete. Draws from Q&A doc Q1, Q3, Q6, Q9, Q13.
     Corrected Strouhal framing consistent with §5.3 actual results (near-zero eigenvalues).
     Follows CFD Paper Instructions.docx: plain language, limitations honest, future work clear. -->

---

## 6. Discussion

### 6.1 One Operator, All Readouts — Why This Design Choice Matters

The central architectural decision in ParaKoop is that a single learned object — the
parametric Koopman operator A(θ) — produces every quantity the paper reports: Cd
prediction, Cl prediction, eigenspectrum, the HHL linear system, and the inverse design
gradient. This is not merely an aesthetic preference; it has two concrete consequences.

**First, consistency.** When Cd prediction and the HHL linear system come from the same
A(θ), any claim about the condition number κ(A(θ)) directly characterises the same model
that predicts drag. If they were separate objects — a GNN surrogate for Cd, a separately
fitted operator for HHL — the condition number of the HHL system would say nothing about
the reliability of the drag prediction, and vice versa. The triple-duty role of κ (§3.5)
only holds because there is one operator.

**Second, the inverse design gradient is free.** Because θ → A(θ) → z* → [Cd, Cl] is a
single differentiable chain, the gradient ∂Cd/∂θ is available by automatic differentiation
at no additional cost. A forward-only surrogate (GBR, GNN) cannot provide this gradient
at all — inverting them requires either re-training a separate inverse model, or using
black-box optimisation (genetic algorithms, Bayesian search) that needs hundreds of
forward evaluations per inverse query. ParaKoop's inverse design runs 300 AdamW steps
through one model, taking under a second on CPU.

---

### 6.2 Why Near-Zero Eigenvalues Are the Correct Result

Section 5.3 reports eigenvalues with imaginary parts near zero (St_max ≈ 0.0002) — far
below the Ahmed-body Strouhal numbers reported in the literature (St ≈ 0.07 for bubble
pumping, 0.13–0.17 for wake shedding). This initially appears to be a failure: should a
physically meaningful Koopman operator not reproduce known temporal frequencies?

The answer is no — and understanding why clarifies what A(θ) actually represents.
A(θ) is a *geometric* operator: it encodes how aerodynamic performance changes as you
move from one car shape to another. It is trained on time-averaged, steady-state data —
there are no snapshots at successive time steps in the DrivAerNet++ training set, and
AhmedML is used in mean-field mode. A temporal Koopman operator (z_{t+1} = A·z_t) would
produce complex eigenvalues whose imaginary parts are temporal oscillation frequencies.
A geometric Koopman operator (fixed point: A(θ)·z* = z* + b(θ)) has no time axis, and
near-real eigenvalues are the expected signature.

The correct validation is not "do eigenvalue imaginary parts match Strouhal numbers?"
but rather "are eigenvalue magnitudes near 1.0, confirming the near-identity design of
Eq. (1)?" — which they are: |λ| ∈ [0.985, 1.098] across all three body styles, with
tight clustering. This tight spectral clustering is precisely what produces the low
condition numbers (κ = 1.03–1.23) that make the HHL interface effective.

---

### 6.3 Proximity Regularisation: Safety by Design

The ablation in §5.7 demonstrates that without proximity regularisation (λ_prox = 0),
the gradient-descent optimiser achieves the target Cd with perfect numerical accuracy
(error = 0.000) but produces geometrically implausible designs — reducing a car's height
by 300 mm and width by 312 mm to create an unrealistically narrow vehicle that achieves
low drag through sheer dimensional reduction, not aerodynamic shaping.

This failure mode is important to understand. The optimiser is not "wrong" in a
mathematical sense — the model genuinely predicts Cd = 0.23 for the suggested θ. The
problem is that the suggested θ decodes to a car shape that no production engineer would
or could build. The guardrails (Re, Ma, Eu) do not catch this, because the decoded vehicle
is still large enough to clear the Reynolds number threshold and the predicted Cd is still
in the automotive range.

Proximity regularisation (λ_prox = 2.0) addresses this by penalising deviation from the
starting geometry. Concretely, it encodes the prior that *a good engineering suggestion
modifies an existing design incrementally* rather than reinventing it. The cost is a small
accuracy penalty (mean |Cd error| increases from 0.000 to ≈ 0.015), which is acceptable
given that the suggestions are now physically credible and directly actionable.

The appropriate value of λ_prox is problem-dependent. For exploratory searches where
large design changes are permitted, a lower value (0.5–1.0) may be preferable. For
production engineering where small, low-risk changes are needed, the default of 2.0
is appropriate. This is exposed as a user-adjustable slider in the Streamlit application.

---

### 6.4 Quantum Readiness: What κ = 1.03–1.23 Means and Does Not Mean

The condition number results (§5.4) show κ(A(θ)) = 1.03–1.23 across the three body
style families, implying a theoretical Harrow–Hassidim–Lloyd (HHL) speedup of 14.8–17.8×
over classical Conjugate Gradient. This is a meaningful result — raw discretised CFD
systems typically have κ = 10²–10⁶ [18], and achieving κ < 2 is significant.

However, three important caveats apply:

**1. The theoretical speedup assumes efficient state preparation.** HHL's O(κ² log N)
complexity counts quantum operations, but it does not count the cost of encoding the
vector b(θ) as a quantum state, or reading out the solution z* as classical values. On
current Noisy Intermediate-Scale Quantum (NISQ) hardware, these overheads can dominate
the quantum circuit cost, erasing the theoretical advantage. We report κ and speedup as
*readiness characterisation*, not a demonstration of achieved quantum advantage.

**2. Simulations are classical.** All results in §5.4 are produced by a classical
eigendecomposition-based simulation of the HHL circuit. No physical quantum hardware was
used. The 100% solution fidelity is therefore a verification that the linear system is
well-posed and the HHL circuit is correctly formulated — not a quantum computation result.

**3. Low κ is necessary but not sufficient.** Even with κ close to 1, practical HHL
advantage additionally requires: (a) an efficient block-encoding of A(θ) as a Hermitian
quantum operator; (b) a quantum-RAM (qRAM) or equivalent structure to load b(θ); and
(c) fault-tolerant quantum hardware with enough coherence time for the phase estimation
circuit. All three remain engineering challenges for near-term devices.

The contribution here is establishing that *the linear systems arising naturally from
Koopman-based automotive surrogates are in the right conditioning regime for HHL* — a
finding that makes this a productive direction for future quantum hardware experiments
when those engineering challenges are resolved.

---

### 6.5 Limitations

**Cl prediction accuracy.** Cl MAE (0.1045) is substantially higher than Cd MAE (0.0378)
on the AhmedML held-out set. This reflects a training data imbalance: only 425 AhmedML
samples carry Cl labels, versus 9,163 DrivAerNet++ samples for Cd — a 21× disparity. The
Cl head is further limited by the Ahmed-body range (Cl ∈ [−0.219, 0.709]), which does not
fully represent the lift characteristics of production sedan designs. Incorporating
DrivAerNet++-scale Cl data (from wind tunnel measurements or full-car transient
simulations) would substantially reduce this gap.

**Cross-scale validation gap.** The cross-scale inverse design check (§5.5) shows
θ-distances of 0.29–1.54 between inverse-designed car geometries and the nearest AhmedML
runs. Large θ-distances mean the CFD comparison is indicative rather than a close
geometric match. The most direct validation — running OpenFOAM RANS on a genuinely novel
inverse-designed shape — remains as planned future work.

**Fixed-resolution θ.** The 8-dimensional θ captures body style, proportions, slant
angle, and cabin fraction but does not encode fine geometric details: A-pillar curvature,
underbody diffuser shape, mirror geometry, or wheel well configuration. Design suggestions
at the current resolution are directionally correct but coarser than a full aerodynamic
optimisation. Extending θ to 12–16 dimensions (adding windshield angle, diffuser slope,
frontal blockage) would improve resolution at the cost of requiring more training data
per configuration.

**Single wind speed.** Training data from DrivAerNet++ (U∞ = 38.9 m/s) and AhmedML
(U∞ = 40 m/s) cover a narrow velocity band around highway cruising speed. The model is
not validated for low-speed urban aerodynamics (U∞ ≈ 10–15 m/s) or high-speed
performance driving (U∞ > 60 m/s), where Reynolds number effects and compressibility
corrections differ from the training regime.

---

### 6.6 Future Work

Three directions follow naturally from the current results:

1. **Direct CFD closure.** Run OpenFOAM RANS on 3–5 of the top inverse-designed
   geometries (lowest |Cd error|, diverse body styles), comparing predicted Cd against
   simulation. This closes the validation loop without requiring new ML training.

2. **Expanded θ and Cl training data.** Add windshield angle and underbody geometry
   dimensions to θ; source Cl labels from wind tunnel databases or additional transient
   datasets to reduce the 21× Cl/Cd training imbalance.

3. **Hardware HHL experiment.** Use the low-κ linear systems characterised here as input
   to a near-term quantum hardware experiment (e.g., IBM Quantum or IonQ), quantifying
   the practical speedup achieved versus the theoretical 14.8–17.8× bound.
