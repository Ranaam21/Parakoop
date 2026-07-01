# Section 3: Methodology

---

## 3. The ParaKoop Architecture

ParaKoop is built around one central idea: instead of learning a separate model for drag
prediction, inverse design, and quantum solving, we learn a single mathematical object —
the **parametric Koopman operator A(θ)** — from which all three capabilities follow
automatically. Figure 1 shows the full pipeline; the sections below explain each piece
in plain terms.

> **Intuition — One object, three capabilities.**
> Think of A(θ) as a "physics fingerprint" for a car shape θ. Once you have it, you
> can read off the drag prediction (forward pass), run it backwards to find the shape
> that achieves a drag target (inverse design), or hand it to a quantum solver as a
> ready-made linear system. Nothing is computed twice from separate models.

Figure 1 gives the full architecture overview; the following subsections describe each
component in detail.

### 3.1 Structured Geometry Representation

The first step is to describe any car shape as a compact numerical vector θ. We use
eight numbers, chosen so they work equally well for large passenger cars and small
wind-tunnel models:

```
θ = [s_F,  s_N,  s_E,  h/L,  w/h,  α/90,  c_f,  δ]   ∈  ℝ⁸
```

| Symbol | Meaning | Example value |
|---|---|---|
| s_F, s_N, s_E | Body style flags (one-hot: fastback / notchback / estateback) | [1, 0, 0] = fastback |
| h/L | Height ÷ length (aspect ratio) | 0.27 for a typical sedan |
| w/h | Width ÷ height (cross-section ratio) | 0.84 for DrivAerNet++; 1.40 for Ahmed body |
| α/90 | Rear slant angle ÷ 90 (normalised to [0, 1]) | 0.28 for a 25° slant |
| c_f | Cabin fraction: cabin length ÷ body length | 0.55 |
| δ | Detail level (0 = simplified, 0.5 = Ahmed bluff body, 1 = detailed STL) | 0 or 0.5 or 1 |

> **Intuition — Why ratios instead of millimetres?**
> DrivAerNet++ cars are roughly 4,800 mm long; AhmedML bodies are 1,044 mm — a 4.6×
> difference. If we fed raw dimensions to the model, it would think they are completely
> different objects. But their height-to-length ratio (h/L) is almost identical:
> 0.27 vs 0.26. Using ratios makes the model "scale-blind" — it sees shape, not size.

**Physical bounds.** During inverse design, θ is clamped within:
h/L ∈ [0.20, 0.45], w/h ∈ [0.50, 1.80], α/90 ∈ [0, 0.55], c_f ∈ [0.25, 0.70].
This prevents the optimiser from suggesting a car shape that lies outside the training data.

---

### 3.2 Parametric Koopman Operator

The core of the model is a K × K matrix A(θ) that changes with the car shape θ.
Here K = 128 is the size of the "lifted space" — a high-dimensional coordinate system
in which aerodynamic behaviour looks linear, even though it is not linear in the original
geometry space.

> **Intuition — What does "lifting" mean?**
> Imagine plotting the drag of all possible cars as a surface over the 8-dimensional
> shape space. That surface has curves and hills — it is nonlinear. Now imagine expanding
> the description to 128 dimensions using learned features. In this bigger space, the same
> surface can be described by a simple flat (linear) map. That is the Koopman lifting:
> swap a hard nonlinear problem for an easy linear one, at the cost of working in a
> higher-dimensional space.

**Low-rank design.** Building a 128×128 matrix from scratch for every car shape would
require a very large network. Instead, A(θ) is written as the identity plus a small,
geometry-dependent correction:

```
A(θ)  =  I  +  Σ_{k=1}^{r}  α_k(θ) · (u_k ⊗ v_k^T)          (Eq. 1)
```

| Symbol | Meaning |
|---|---|
| I | Identity matrix (128×128) — the "do nothing" baseline |
| r = 16 | Number of correction modes (rank) |
| u_k, v_k | Fixed K-dimensional basis vectors, shared across all car shapes |
| α_k(θ) | Geometry-dependent weight for mode k, produced by a small neural network |
| ⊗ | Outer product: creates a rank-1 matrix from two vectors |

> **Intuition — Operator dictionary.**
> Think of {u_k, v_k} as 16 "deformation modes" of the physics space — directions
> in which aerodynamics can vary (e.g. "front-heavy drag", "slant-angle effect").
> The network α(θ) simply looks at the car shape and decides how much of each mode
> to activate. All cars share the same 16 modes; what changes is the mix.

The bias vector b(θ) captures the geometry-dependent "offset" of the fixed point and is
produced by a separate two-layer network: b(θ) = b_net(θ), b_net: ℝ⁸ → ℝ¹²⁸.

**Efficient computation.** The matrix A(θ) is never stored explicitly during training.
Instead, the matrix–vector product A(θ)·z is computed as:

```
A(θ)·z  =  z  +  Uᵀ · (α(θ) ⊙ (V·z))                        (Eq. 2)
```

This uses three fast vector operations instead of one slow 128×128 matrix multiply,
reducing training cost by a factor of 8.

---

### 3.3 Koopman State and Performance Head

**How the fixed point encodes drag.** The key equation is:

```
A(θ)·z*  =  z*  +  b(θ)                                       (Fixed-point equation)
```

z* is the steady-state "fingerprint" of the car in the 128-dimensional lifted space.
Once we have z*, drag and lift follow from a simple linear layer:

```
[Cd, Cl]  =  W · z*  +  c                                      (Eq. 3)
```

| Symbol | Meaning |
|---|---|
| z* | Fixed-point lifted state (128-dim vector) — the car's aerodynamic fingerprint |
| W | Linear weight matrix (2 × 128) |
| c | Bias vector (2-dim) |

> **Intuition — Why a fixed point?**
> A fixed point z* is a state that "maps to itself" under the operator. In fluid
> dynamics terms, it represents the steady-state flow structure that the car's
> geometry produces — the flow pattern that doesn't change once the car reaches
> its cruise speed. Reading drag off this fixed point is equivalent to reading off
> the steady-state drag from a time-averaged CFD solution, but in milliseconds
> instead of hours.

**Two paths to z*, depending on the data available:**

*Path A — Flow-field path (AhmedML, 75 cases with VTU files):*
A lifting network maps the full 3D flow field φ (13,824-dimensional: velocity and
pressure at 3,456 probe locations) to the lifted state:
```
z̄  =  lift(φ),     lift: ℝ¹³⁸²⁴ → ℝ¹²⁸
```

*Path B — Direct geometry path (DrivAerNet++, 9,163 cases, no flow fields):*
A three-layer network maps the 8-dim geometry directly to the fixed point:
```
z*(θ)  =  z_net(θ),    z_net: ℝ⁸ → ℝ¹²⁸   (3 layers, GELU)
```

Both paths share the same performance head (Eq. 3), enabling joint training across
both datasets even though DrivAerNet++ has no flow-field data.

The Cl output is masked to zero loss on DrivAerNet++ samples, which carry no Cl
label. Only the 425 AhmedML training samples contribute to Cl learning.

---

### 3.4 Training Objective

The model is trained end-to-end on the combined loss:

```
L  =  L_fp  +  λ_perf · L_perf  +  λ_ae · L_ae               (Eq. 4)
```

| Term | Formula | Purpose |
|---|---|---|
| L_fp | (1/K) · ‖A(θ)·z* − z* − b(θ)‖² | Forces A(θ) to act as a genuine fixed-point operator |
| L_perf | MSE(Cd_pred, Cd) + has_cl · MSE(Cl_pred, Cl) | Drag and lift accuracy |
| L_ae | (1/D) · ‖decode(z̄) − φ‖² | Keeps the lifted state grounded in real flow physics |
| λ_perf | 1.0 | Performance weight |
| λ_ae | 0.1 | Autoencoder weight |

> **Intuition — Why three loss terms?**
> Without L_fp, A(θ) would just learn to be a pass-through — it would let z_net
> memorise the drag without A(θ) developing any useful structure. L_fp forces
> the operator to actually "do something" geometrically meaningful. Without L_ae,
> the lifted state z̄ could be arbitrary numbers that happen to predict drag while
> losing all connection to the actual flow field. L_ae keeps it grounded.

---

### 3.5 Phi-Supervised Grounding and κ Reduction

Without flow-field supervision, the learned A(θ) matrices tend to be poorly conditioned
(condition number κ = 1.72–2.97). A high condition number means the operator is "wobbly"
— small changes in the input produce large swings in the output, making the quantum
linear solver inefficient.

For the 75 AhmedML cases with volumetric VTU output, we add a "grounding" loss that
forces A(θ) to act correctly on the *actual measured flow state*, not just on the
network's own estimate:

```
L_phi  =  ‖A(θ)·lift(φ) − lift(φ) − b(θ)‖²  +  L_ae           (Eq. 5)
```

> **Intuition — Phi grounding as a reality check.**
> Without grounding, A(θ) only has to satisfy the fixed-point equation for states
> z* that z_net itself produces — which it can "cheat" on. Phi grounding says: your
> operator must also satisfy the fixed-point equation when I plug in the *real* flow
> field from OpenFOAM. This real-world check forces A(θ) to develop physically
> meaningful structure, which happens to make it well-conditioned as a side effect.

**Effect on condition number:** κ drops from 1.72–2.97 (without grounding) to
1.03–1.23 (with grounding) — a 63% mean reduction — without any loss in Cd/Cl
prediction accuracy. This low κ is what makes the HHL quantum interface viable.

---

### 3.6 Condition Number and Its Triple Role

The condition number κ(A(θ)) = σ_max / σ_min (ratio of largest to smallest singular
value) plays three distinct roles in ParaKoop:

| Role | What κ tells you | Threshold |
|---|---|---|
| Flow regime indicator | Low κ ≈ streamlined, attached flow; high κ ≈ bluff, separated wake | κ < 1.5 = well-conditioned |
| HHL readiness gate | Theoretical speedup = K / (κ · log₂K) — only beneficial for low κ | κ < 2 to see real speedup |
| Gradient trust | Poorly conditioned A(θ) makes ∂Cd/∂θ noisy; low κ = reliable gradient | κ < 1.5 = trustworthy |

All three diagnostics are computed from a single call to `torch.linalg.cond(A)`.

---

### 3.7 HHL Quantum Interface

The trained A(θ) defines a linear system that can be handed directly to the
Harrow–Hassidim–Lloyd (HHL) quantum linear solver [15]:

```
A(θ) · z  =  z*(θ) − b(θ)                                     (Eq. 6)
```

| Symbol | Meaning |
|---|---|
| A(θ) | Parametric Koopman operator (128×128 matrix) |
| z | Unknown vector (the HHL algorithm solves for this) |
| z*(θ) − b(θ) | Right-hand side: the target fixed-point offset |

**Why is this quantum-friendly?** The HHL algorithm solves M·z = b in time
O(κ² · log(K) · log(1/ε)), compared to classical Conjugate Gradient at
O(K · κ · log(1/ε)). The speedup ratio is approximately K / (κ · log₂K).
At K = 128 and κ = 1.12 (mean across body styles):

```
Speedup  =  128 / (1.12 × 7)  ≈  16.3×
```

> **Intuition — Why 128 dimensions and 15 qubits?**
> K = 128 = 2⁷ means we need exactly 7 qubits to encode the state space.
> Phase estimation (the clock register for HHL) needs another 7 qubits.
> Plus 1 ancilla qubit = **15 qubits total** — small enough for near-term
> quantum hardware. Most CFD systems need millions of unknowns (and hence
> 20+ qubits), but Koopman lifting compresses the problem into 128 dimensions
> where quantum advantage is already meaningful.

Simulations use Qiskit 1.x on a classical simulator; no physical quantum hardware is used.
The κ study (§5) characterises readiness for actual quantum execution.

---

### 3.8 Inverse Design

Given a target drag Cd*, the model works backwards through the same differentiable
pipeline to find the car shape θ that achieves it. No new training or CFD simulation
is needed — only gradient descent on the already-trained model.

**Optimisation loss:**

```
L_inv(θ)  =  (Cd(θ) − Cd*)²
           +  λ_Cl · (Cl(θ) − Cl*)²      [only if Cl target given]
           +  λ_prox · (1/8) · ‖θ − θ₀‖²                         (Eq. 7)
```

| Symbol | Meaning | Default |
|---|---|---|
| Cd* | Target drag coefficient | User input |
| Cl* | Target lift coefficient | Optional |
| θ₀ | Starting geometry (mean shape for chosen style) | Auto-set |
| λ_Cl | Lift penalty weight | 0.5 |
| λ_prox | Proximity penalty weight | 2.0 |

> **Intuition — What does the proximity term do?**
> Without λ_prox, the optimiser will shrink the car as small as possible to achieve
> low drag — technically correct mathematically, but the result is a car 749 mm
> wide and 990 mm tall (about the size of a motorbike). λ_prox says: "don't stray
> too far from where you started." The ablation in §5 shows that λ_prox = 2.0 gives
> engineering-credible suggestions, while λ_prox = 0 gives exact Cd but implausible
> geometry.

**Settings:** AdamW optimiser, 300 steps, learning rate 5×10⁻³. θ is clamped to
physical bounds after every step. Three starting points (fastback, notchback,
estateback) are run in parallel; results are ranked by |Cd_achieved − Cd_target|.

---

### 3.9 Physics Guardrails

After inverse design, every candidate θ is checked against three dimensionless
number bounds before being shown to the user:

| Guardrail | Formula | Range | What it checks |
|---|---|---|---|
| Reynolds number | Re = U∞ · L / ν | 3×10⁶ – 3×10⁷ | Is the car within the training flow regime? |
| Mach number | Ma = U∞ / c_sound | < 0.30 | Is the flow incompressible? (above 0.3, air compresses) |
| Euler number | Eu ≈ Cd_pred | 0.18 – 0.35 | Is the predicted drag within automotive range? |

> **Intuition — Guardrails as out-of-distribution detection.**
> The surrogate was trained on cars at highway speed (~38–40 m/s). If inverse
> design produces a shape so extreme that Re, Ma, or Eu falls outside the training
> regime, the surrogate's prediction is no longer reliable — it's extrapolating
> beyond what it has seen. The guardrails flag these cases explicitly rather than
> silently returning an unreliable number.

Any candidate that fails a guardrail is flagged in the Streamlit application with
a plain-language explanation of why.

---

**Architecture summary**

| Component | Type | Parameters | Role at inference |
|---|---|---|---|
| z_net | 3-layer MLP, GELU | ~37K | θ → z* |
| ParametricOperator | Low-rank (r=16) + 2 MLPs | ~25K | A(θ), b(θ) |
| perf_head | Linear layer | 258 | z* → [Cd, Cl] |
| LiftingNet | MLP | ~3.6M | φ → z̄ (training only) |
| DecoderNet | MLP | ~3.6M | z̄ → φ̂ (training only) |
| **Total active at inference** | — | **~62K** | — |

The lifting and decoding networks (7.2M parameters total) are used only during
training to supervise the operator with real flow fields. At inference, only the
62K-parameter forward chain (z_net + operator + perf_head) runs.
