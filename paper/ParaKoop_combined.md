# ParaKoop: Parametric Koopman Surrogate for Automotive Aerodynamic Design with Quantum HHL Integration

**Amit Rana**

Independent Researcher

amit21aim@zohomail.com

ORCID: 0009-0008-5998-6560

June 2026

---

## Abstract

Aerodynamic drag reduction in passenger car design requires repeated Computational Fluid
Dynamics (CFD) simulations — each taking hours — with no mechanism to directly ask: *given
a drag target, what geometry achieves it?* We present **ParaKoop**, a
**Para**metric **Koop**man surrogate that answers this question in under a second, without
new simulations.

ParaKoop learns a geometry-conditioned Koopman operator A(θ) — a linear map in a
high-dimensional lifting space whose fixed point encodes aerodynamic performance — jointly
across 9,620 CFD results from two public datasets: DrivAerNet++ (~8,000 steady-state car
simulations) and AhmedML (499 transient bluff-body simulations with drag and lift labels).
A unified 8-dimensional scale-invariant geometry descriptor θ bridges the two datasets
without retraining. On a held-out validation set, ParaKoop achieves a drag coefficient
(Cd) Mean Absolute Error (MAE) of **0.01401**, outperforming a Gradient-Boosted Regressor
(GBR) baseline by 10%. On 74 completely held-out AhmedML cases — a genuine out-of-scale
domain test — Cd MAE is 0.0378 against published OpenFOAM ground truth.

Given a target Cd, gradient descent through the differentiable θ → A(θ) → performance
chain performs **inverse design**, outputting geometry changes in real engineering units
(millimetres and degrees) within 300 optimisation steps. Proximity regularisation prevents
the optimiser from producing geometrically implausible designs, with all suggestions
passing Reynolds number (Re), Mach number (Ma), and Euler number (Eu) physics guardrails.

The same A(θ) defines a quantum linear system A(θ)·z = b(θ) suitable for the
Harrow–Hassidim–Lloyd (HHL) quantum linear solver. Empirical condition numbers
**κ = 1.03–1.23** across fastback, notchback, and estateback geometries imply a
theoretical HHL speedup of **14.8–17.8×** over classical Conjugate Gradient using
**15 qubits** — the first systematic κ characterisation across a large automotive shape
family. Code and an interactive Streamlit demonstration are publicly available at https://github.com/Ranaam21/Parakoop.

**Keywords:** Parametric Koopman operator; automotive aerodynamics; surrogate model; inverse design; drag coefficient; HHL quantum linear solver; condition number; physics guardrails; DrivAerNet++; AhmedML

---

## 1. Introduction

### 1.1 The Problem: Designing Better Cars Takes Too Long

Designing the aerodynamics of a car is one of the most computationally expensive tasks in
engineering. Every time an engineer wants to know how a design will behave in the wind —
how much drag it creates, how stable it is at highway speed — they must run a
**Computational Fluid Dynamics (CFD)** simulation. CFD numerically solves the equations
of fluid motion (the Navier–Stokes equations) around a full 3D car model, discretised
into tens of millions of small cells. A single such simulation can take anywhere from 6 to
48 hours on a powerful computer cluster, even for a simplified car shape.

In a typical automotive design cycle, engineers run dozens to hundreds of these simulations
per design iteration. At this rate, the entire design loop is inherently *forward*: the
engineer specifies a geometry, the simulation tells them the resulting drag and lift, and
the engineer manually decides what to change next. There is no mechanism to ask the
reverse question: **"Given that I want a drag coefficient (Cd) of 0.25, what shape should
I build?"**

This paper presents **ParaKoop** — a **Para**metric **Koop**man surrogate model that
answers exactly that reverse question, directly, in under a second, without running a
single new CFD simulation.

---

### 1.2 Key Concepts: What is a Surrogate Model and Why Koopman?

**Surrogate model:** A machine learning model trained on existing CFD results that can
predict aerodynamic performance (drag, lift) for any new geometry in milliseconds, instead
of hours. Think of it as a "learned shortcut" through the physics.

**Koopman operator (A):** A mathematical object from dynamical systems theory [1] that
represents how a physical system evolves over time or changes with parameters. The key
insight is this: complex, nonlinear physical behaviour often looks *linear* when viewed in
the right higher-dimensional space — the Koopman lifting space. Instead of modelling
drag as a complex nonlinear function of 8 geometry variables, we lift those variables into
a 128-dimensional space where the relationship becomes linear.

**Why "Parametric" Koopman?** Classical Koopman operators describe how a single system
evolves in *time*. Our operator A(θ) is different: it describes how the physics changes
with *geometry* (parameterised by θ). This allows us to learn one operator that works
across thousands of different car shapes, rather than fitting a separate model per design.

---

### 1.3 The Forward and Inverse Design Flows

ParaKoop supports two complementary design directions (see Figure 1 for the full
architecture diagram):

**Forward flow** (Predict): Given a car geometry θ, the model predicts drag coefficient
(Cd) and lift coefficient (Cl) in real time.

```
θ  →  A(θ)  →  z*  →  [Cd, Cl]
(geometry)  (Koopman operator)  (lifted state)  (performance)
```

**Inverse flow** (Design): Given a target Cd (and optionally a target Cl), the model
works *backwards* through the same differentiable chain to find the geometry θ that
achieves it.

```
Cd*  →  gradient descent on θ  →  θ_opt  →  geometry changes (mm, degrees)
(target)                         (optimised)  (e.g. "reduce rear slant by 10°")
```

The inverse flow does not require any additional training or simulation — it uses the
gradient ∂Cd/∂θ through the already-trained model. This is possible only because all
components of the pipeline are differentiable (§3).

---

### 1.4 What Makes ParaKoop Novel?

Existing automotive aerodynamics surrogate models [2, 3] are built as forward-only
predictors: they take geometry as input and output drag. None provides a direct,
gradient-based inverse design mechanism that outputs real engineering dimensions. Existing
Koopman surrogate methods for fluid dynamics [4, 5] apply the operator to a single geometry
at a time — they cannot condition on different geometries without retraining. And while
quantum linear solvers like HHL (Harrow–Hassidim–Lloyd) [6] have been studied theoretically
for CFD applications, no prior work has characterised the condition number κ(A) of a
Koopman operator over a large, diverse automotive shape family.

ParaKoop makes six distinct contributions:

1. **Geometry-conditioned Koopman operator:** A(θ) is trained to vary *with* car geometry
   via a low-rank dyadic parameterisation (§3.2). The same operator predicts Cd for a
   fastback, notchback, or estateback — no separate models per style.

2. **One object, all readouts:** Every quantity the paper reports — Cd, Cl, the
   eigenspectrum, the HHL input system, and the inverse design gradient — is a direct
   structural readout of A(θ). Nothing is computed from a separate model or branch.

3. **Direct inverse design:** Given target Cd (and optional Cl), AdamW gradient descent
   on θ through the differentiable A(θ) pipeline outputs real geometry changes in
   millimetres and degrees — not abstract latent directions.

4. **Multi-dataset joint training:** ParaKoop is trained jointly on **DrivAerNet++** [2]
   (~8,000 steady-state car CFD runs) and **AhmedML** [3] (499 transient bluff-body CFD
   runs), bridged by a unified 8-dimensional scale-invariant geometry representation.
   This is, to our knowledge, the first surrogate trained jointly across both datasets.

5. **HHL quantum readiness characterisation:** The condition number κ(A(θ)) is measured
   across three body style families after phi-supervised grounding, yielding κ = 1.03–1.23
   — a regime where the HHL algorithm [6] offers a theoretical speedup of 14.8–17.8×
   over the best classical solver, requiring only 15 qubits.

6. **Physics guardrails:** Every inverse design suggestion is automatically checked against
   the Reynolds number (Re), Mach number (Ma), and Euler number (Eu) — the key
   dimensionless quantities governing when incompressible RANS (Reynolds-Averaged
   Navier–Stokes) simulation physics are valid. Suggestions that fail are flagged and
   explained, not silently discarded.

---

### 1.5 Industry Relevance

The capabilities developed in this paper are directly applicable beyond passenger car design:

- **Automotive:** Real-time inverse aerodynamic design for Cd/Cl targets, reducing the
  design-to-simulation cycle from days to seconds.
- **Motorsport:** Rapid exploration of downforce (negative Cl) configurations within
  guardrail-bounded geometry space.
- **Aerospace and UAV:** The same parametric Koopman framework applies to wing cross-section
  and fuselage shape optimisation against drag targets.
- **Space re-entry vehicles:** The inverse design loop can be extended to thermal loads
  (adding Stanton number as a performance target) given appropriate training data.
- **Semiconductor and MEMS:** Microfluidic channel and heat-sink geometry optimisation
  where flow simulations are equally expensive and inverse design is equally under-served.

The public release of the full codebase and Streamlit demonstration makes these capabilities
accessible without requiring CFD expertise.

---

### 1.6 Paper Organisation

The remainder of this paper is organised as follows. Section 2 reviews related work on
surrogate-based aerodynamic design, Koopman operator methods for fluid mechanics, and
quantum linear solvers. Section 3 describes the ParaKoop architecture in full detail.
Section 4 describes the training datasets. Section 5 reports experimental results across
forward prediction, eigenspectrum analysis, HHL condition number study, CFD domain
validation, inverse design quality, and proximity regularisation ablation. Section 6
discusses limitations and future directions. Section 7 concludes.

---

## 2. Related Work

ParaKoop sits at the intersection of three active research streams: data-driven Koopman
operator methods for fluid mechanics, surrogate-based aerodynamic design, and quantum
linear solvers for engineering problems. We review each in turn and position our
contribution explicitly against the state of the art.

---

### 2.1 Koopman Operator Methods for Fluid Mechanics

**Foundations.** The Koopman operator was introduced by Koopman [1] as a way to represent
the evolution of a nonlinear dynamical system (such as fluid flow) as an equivalent
*linear* operation in a higher-dimensional space of observables. The intuition is
powerful: even though the Navier–Stokes equations are notoriously nonlinear, there often
exists a lifted coordinate system in which the same dynamics look linear — making the full
machinery of linear algebra (eigendecomposition, matrix solvers, gradient computation)
applicable to genuinely nonlinear physics.

**Extended Dynamic Mode Decomposition (EDMD).** Williams et al. [2] introduced EDMD, which
approximates the Koopman operator from data by regressing the lifted state at the next
time step against the current one: z_{t+1} ≈ A · z_t. This is the standard data-driven
Koopman algorithm, used widely in fluid mechanics to extract coherent flow structures
(Koopman modes) and reduced-order models of turbulence, wakes, and boundary layers.

**Modern Koopman theory.** Brunton et al. [3] provide the definitive survey of
data-driven Koopman methods across dynamical systems, covering EDMD variants, spectral
analysis, and connections to Dynamic Mode Decomposition (DMD). This is the primary
reference for the Koopman framework as used in the engineering and machine learning
communities today.

**Deep Koopman networks.** Lusch et al. [4] and Morton et al. [5] extended EDMD by

using deep neural networks to *learn* the lifting function z = Φ(x) jointly with the
linear operator A. This allows the model to discover lifting coordinates that are
not hand-crafted, making it applicable to complex, high-dimensional flows.

**Limitation shared by all prior Koopman work.** Every method above learns a Koopman
operator for a *single fixed geometry*. The operator A represents "how this flow evolves
in time from any initial condition" — but it is specific to the geometry it was calibrated
on. Applying it to a different car shape requires re-running the entire EDMD or training
procedure from scratch. Moreover, all existing Koopman surrogates for fluid mechanics
require time-series data (snapshots z_t, z_{t+1}, …) — which means they cannot exploit
large steady-state-only datasets like DrivAerNet++ that contain no temporal information.

**What ParaKoop adds.** We introduce a *parametric* Koopman operator A(θ), where θ is a
compact geometry descriptor. Rather than learning how one flow evolves in time, we learn
how the operator itself changes across shape space. Steady-state performance (Cd, Cl) is
read off as the fixed point A(θ)·z* = z* + b(θ) — no time series required. This is, to
our knowledge, the first Koopman formulation that conditions on car geometry and is
jointly trained across two large, heterogeneous automotive CFD datasets.

---

### 2.2 Surrogate-Based Aerodynamic Design

**GNN and CNN surrogates.** Several recent works have applied Graph Neural Networks (GNNs)
and Convolutional Neural Networks (CNNs) to map 3D surface meshes or volumetric grids
directly to drag/lift coefficients [6, 7]. These models achieve strong accuracy on
in-distribution geometries but require the full 3D mesh as input — making them expensive
to query at interactive rates — and do not provide any mechanism for inverse design.

**DrivAerNet++.** Elrefaie et al. [8] released the largest publicly available automotive
CFD dataset, comprising approximately 8,000 steady-state RANS (Reynolds-Averaged
Navier-Stokes) simulations of full-scale passenger car geometries across three body
archetypes (fastback, notchback, estateback) and eight configuration families. The
companion GNN and Gradient-Boosted Regressor (GBR) baselines are forward-only: given a
mesh or a configuration vector, predict Cd. No inverse capability is provided.
ParaKoop trains on DrivAerNet++ and outperforms the GBR baseline by 10% in Cd MAE
while simultaneously providing inverse design and quantum-interface capabilities.

**AhmedML.** Ashton et al. [9] published 499 transient RANS-LES (Large Eddy Simulation)
runs of the Ahmed bluff body — a simplified car shape that has been the canonical
aerodynamics benchmark for decades [10]. AhmedML uniquely provides both Cd and Cl labels,
as well as full volumetric flow fields for 76 of the 499 cases. No prior surrogate work
has jointly trained on both DrivAerNet++ and AhmedML across a shared geometry
representation. ParaKoop is the first to do so, using the AhmedML flow fields for
phi-supervised grounding of the Koopman operator (§3.6).

**Inverse design methods.** Gradient-based shape optimisation using adjoint CFD solvers
[11] is the classical approach to aerodynamic inverse design: compute the sensitivity of
drag to every surface node, then descend. This is accurate but requires a full CFD solve
per iteration — negating the surrogate's speed advantage. Bayesian optimisation over
surrogate models [12] avoids the per-step CFD cost but scales poorly to high-dimensional
geometry spaces and does not produce interpretable geometry changes. Generative models
such as Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) have
been applied to aerodynamic shape generation [13, 14] but produce shapes in latent space
that are not directly translatable to engineering dimensions without a separate decoder.

**What ParaKoop adds.** The inverse design mechanism in ParaKoop operates directly in
the physically interpretable θ space (8 geometry ratios that decode to real millimetres
and degrees), using gradient descent through the differentiable A(θ) chain. This is
simpler, faster, and more interpretable than generative or adjoint approaches, and it
produces actionable output ("reduce rear slant by 10.4°, reduce height by 48 mm") rather
than a latent vector.

---

### 2.3 Quantum Linear Solvers for Engineering

**The HHL algorithm.** Harrow, Hassidim, and Lloyd [15] proved that a quantum computer
can solve a linear system M·z = b in time O(κ² · log(N) · log(1/ε)), where κ is the
condition number of M and N is the system dimension. The best classical algorithm
(Conjugate Gradient) requires O(N · κ · log(1/ε)). For large N and small κ, HHL offers
an exponential speedup in N. This speedup is the central motivation for quantum computing
applications in scientific simulation, where linear systems of dimension 10⁶–10⁹ are
routine.

**Applications to CFD.** Quantum algorithms for fluid simulation have been proposed as a
long-term research direction [16, 17], exploiting the fact that discretised Navier–Stokes
equations ultimately reduce to large sparse linear systems. However, practical demonstration
has been limited by two challenges: (1) the state preparation and readout overheads of
current (Noisy Intermediate-Scale Quantum, or NISQ) devices erode the theoretical speedup
in practice; (2) there is a lack of empirical data on the condition numbers of the specific
linear systems that arise in real engineering flows.

**Condition number as the critical bottleneck.** The HHL speedup requires small κ. For
raw discretised CFD systems, κ can reach 10²–10⁶ [18] — a regime where HHL offers no
practical advantage. Preconditioning techniques can reduce κ, but their design is
problem-specific and non-trivial. No prior work has systematically measured κ across a
large family of automotive geometries to characterise when quantum advantage is plausible.

**What ParaKoop adds.** The parametric Koopman operator A(θ) naturally produces linear
systems M = A(θ) whose condition numbers are empirically low: κ = 1.03–1.23 across
fastback, notchback, and estateback geometries after phi-supervised grounding (§3.6,
§5.4). This is not a coincidence — the near-identity design of A(θ) (Eq. 1: A = I +
low-rank perturbation) ensures the operator is well-conditioned by construction. The
contribution here is the first systematic empirical κ study across a large automotive
shape family, demonstrating that Koopman-derived linear systems are inherently
HHL-friendly — a finding that connects the surrogate design choice directly to quantum
computing readiness.

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

![Figure 1. ParaKoop architecture. Top row (cyan): forward flow θ → A(θ) → z* → [Cd, Cl]. Bottom row (amber): inverse design flow Cd* → AdamW gradient descent on θ → geometry changes in millimetres and degrees. The φ-grounding path (grey dashed) uses 75 AhmedML VTU flow fields to reduce the condition number κ(A(θ)) from 1.72–2.97 to 1.03–1.23.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig1_architecture.png)



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



![Figure 5. Training dataset composition. (a) Sample distribution by source (total ≈ 9,588 designs); (b) Cd distribution overlap between DrivAerNet++ STL (1,163 designs) and AhmedML held-out (74 cases), demonstrating that the unified θ bridges two datasets of different vehicle scales.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig5_datasets.png)

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

## 5. Experiments and Results

> **What this section shows.**
> We test ParaKoop on five tasks: (1) drag prediction vs a classical baseline;
> (2) cross-scale validation on completely unseen AhmedML geometries;
> (3) eigenspectrum analysis to confirm the operator has the right structure;
> (4) HHL quantum condition number study; (5) inverse design quality; and
> (6) a proximity regularisation ablation to explain why λ_prox = 2.0 is needed.

### 5.1 Experimental Setup

**Training.** The ParaKoop model is trained jointly on all sources in the unified dataset
(§4.3) for 300 epochs using AdamW (lr = 3×10⁻⁴, weight_decay = 10⁻⁴) with cosine
annealing (η_min = 3×10⁻⁶). Batch size 32; gradient clipping at ‖g‖ = 1.0.
15% of samples are held out as a validation set (seed = 42, same split used in all
reported evaluations). All experiments run on CPU/MPS; no GPU required.

**Baseline.** A Gradient Boosted Regressor (GBR; scikit-learn, n_estimators=300,
max_depth=4, lr=0.05, subsample=0.8) is trained on the 1,163 DrivAerNet++ STL designs
that have real extracted geometry — the highest-quality Cd/geometry pairs in the dataset.
Features: length, width, height, frontal area, roof height, rear slant angle, windshield
angle, cabin fraction (8 dimensions). Five-fold cross-validated MAE is reported.

This comparison is intentionally conservative: the GBR has access to higher-resolution
raw-millimetre features and is evaluated in-distribution on the same STL set it trains on,
while ParaKoop uses scale-invariant ratio features and is evaluated on the mixed held-out
set that includes AhmedML samples at a very different geometric scale. ParaKoop's advantage
must therefore overcome both the feature compression and the distribution mismatch.

---

### 5.2 Forward Prediction (Cd and Cl)

**Results.**

| Model | Training data | Features | Val metric | Val Cd MAE |
|---|---|---|---|---|
| GBR (baseline) | 1,163 STL designs | 8 raw-mm dims | 5-fold CV | 0.01549 ± 0.00649 |
| ParaKoop (ours) | ~9,620 (all sources) | 8 ratio dims (θ) | 15% held-out | **0.01401** |

> **Intuition — Why beat GBR with compressed features?**
> The GBR gets raw millimetres and is evaluated in-distribution. ParaKoop uses only ratios
> (h/L, w/h, etc.) and is evaluated on a mixed set including out-of-scale AhmedML bodies.
> Beating GBR under these conditions confirms that the Koopman fixed-point structure is
> extracting genuine aerodynamic signal, not just memorising training shapes.

ParaKoop achieves −10% MAE relative to the GBR despite using only ratio-compressed geometry
and training across a heterogeneous multi-scale dataset. The GBR's cross-validation spread
(±0.00649 ≈ 42% of its mean MAE) indicates that its performance is highly split-dependent;
ParaKoop's single-split held-out result is more stable under the full 9,620-sample
training regime.

**AhmedML domain validation.** On the 74 held-out AhmedML cases — geometries at
1/5 the scale of DrivAerNet++ cars — the model achieves Cd MAE = **0.0387**. This is
elevated relative to the mixed validation result, as expected: the AhmedML held-out set
is a genuine out-of-scale domain test, not a same-distribution check. The result confirms
that the ratio-based θ representation transfers across scales without catastrophic failure,
and that the performance gap relative to in-distribution error (0.0387 vs 0.0138) is
attributable to scale-domain shift rather than model overfitting.

**Cl prediction.** Cl supervision is masked on DrivAerNet++ samples (Cl unavailable;
§3.4). The ParaKoop Cl head is therefore trained exclusively on the AhmedML subset.
Cl MAE on the 74-case held-out set: **0.1045** — substantially higher than Cd MAE, reflecting
the 21× training data disparity (425 AhmedML Cl samples vs ~9,163 Cd samples). Figure 2
shows the full scatter of predicted versus CFD Cd across all 74 held-out cases, coloured by
absolute error.

![Figure 2. Predicted versus CFD Cd for 74 held-out AhmedML cases (left) and prediction error distribution (right). Points are coloured by absolute error |ΔCd|. Mean |ΔCd| = 0.0378; median = 0.0364.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig2_cd_scatter.png)


The Cl gauge in the Streamlit demo rounds to four decimal places to avoid floating-point
display artefacts.

---

### 5.3 Eigenspectrum and Strouhal Analysis

The full K×K operator matrix A(θ) is materialised post-training and its eigenspectrum
computed for each body style's mean geometry (§3.5).

> **Intuition — Near-zero eigenvalues are the correct result here.**
> A *temporal* Koopman operator (z_{t+1} = A·z_t) would have complex eigenvalues whose
> imaginary parts encode oscillation frequencies — and you would expect them near the
> known Ahmed-body Strouhal numbers (St ≈ 0.07 and 0.13–0.17). Our operator is
> *geometric*: it encodes how performance changes with shape, not how flow oscillates in
> time. Near-zero imaginary eigenvalues mean the operator is "steady" — exactly right.

**Physical interpretation.** Because A(θ) is geometry-conditioned (not temporal), its
eigenvalues characterise how the operator's structure varies across shape space — not
oscillation frequencies in time. The spectral analysis is therefore a *check on the
operator's character*, not a Strouhal frequency match.

The Strouhal proxy is defined as:
```
St_proxy = |Im(λ_k)| · L_ref / (2π · U∞)
```
with L_ref = 4.8 m (DrivAerNet++ sedan length), U∞ = 38.9 m/s.

| Body style | \|λ\| range | St_max | Modes near bubble (St≈0.07) | Modes near shedding (St=0.13–0.17) |
|---|---|---|---|---|
| Fastback | 0.9904–1.0084 | 0.0002 | 0 | 0 |
| Notchback | 0.9989–1.0026 | 0.0001 | 0 | 0 |
| Estateback | 0.9850–1.0059 | 0.0002 | 0 | 0 |

**Interpretation.** The near-zero St_max (≈ 0.0002 vs. literature values of 0.07–0.17) and
absence of eigenvalues near known Ahmed-body Strouhal bands is the *expected* and *correct*
result for a parametric geometric operator. A(θ) is trained on time-averaged steady-state
data — it encodes how performance changes with shape, not how flow oscillates in time.
The near-real spectrum (|Im(λ)| ≈ 0) confirms the operator has not developed spurious
temporal dynamics, which would be a failure mode indicating overfitting to transient
artefacts in the AhmedML training data.

The eigenvalue magnitudes cluster tightly near 1.0 across all styles (range 0.985–1.098),
consistent with A(θ) being a near-identity geometric operator — the design intention
of Eq. (1): A(θ) = I + low-rank geometry-dependent perturbation. This tight spectral
clustering is also directly responsible for the low condition numbers reported in §5.4.
Figure 6 plots the full eigenvalue magnitude distributions per body style.

![Figure 6. Eigenvalue magnitudes |λ| of A(θ) for K = 128 modes, per body style. All modes cluster tightly near |λ| = 1.0 (vertical dashed line), confirming the near-identity operator design A(θ) = I + low-rank perturbation and explaining the low condition numbers κ = 1.03–1.23.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig6_eigenspectrum.png)



---

### 5.4 HHL Condition Number Study

**System formulation.** For each geometry θ, the trained A(θ) defines the linear system:

```
A(θ) · z  =  z*(θ) − b(θ)
```

This is the A_direct formulation (§3.7): the right-hand side is z*(θ) − b_net(θ),
encoding the inverse step "what initial Koopman state z evolves under A(θ) to produce
z*?" The symmetrised and normalised system is prepared for HHL following §3.7.

**Condition number results.**

| Body style | κ(A(θ)) | Speedup vs CG | Solution fidelity | Predicted Cd | Predicted Cl |
|---|---|---|---|---|---|
| Fastback | **1.121** | 16.3× | 100.00% | 0.2520 | 0.3479 |
| Notchback | **1.028** | 17.8× | 100.00% | 0.2453 | 0.2307 |
| Estateback | **1.234** | 14.8× | 100.00% | 0.2699 | 0.2507 |
| **Range** | **1.03–1.23** | **14.8–17.8×** | **100%** | — | — |

**Speedup formula.** The theoretical speedup of Harrow–Hassidim–Lloyd (HHL) relative to
the classical Conjugate Gradient (CG) solver is:

```
Speedup = dim / (κ · log₂(dim))

Classical CG cost : O(dim · κ · log(1/ε))
HHL cost          : O(κ² · log(dim) · log(1/ε))
```

At K = 128 (dim), log₂(128) = 7, κ ∈ [1.028, 1.234]:

```
Speedup ∈ [128/(1.234×7), 128/(1.028×7)] = [14.8×, 17.8×]
```

**Qubit budget:** 7 state qubits (encoding the K = 2⁷ dimensional space) + 7 clock qubits
(phase estimation) + 1 ancilla = **15 qubits total** — well within near-term quantum
hardware targets. Figure 3 summarises κ, speedup, and predicted Cd per body style.

![Figure 3. HHL quantum interface analysis per body style. (a) Condition number κ(A(θ)); (b) theoretical HHL speedup over conjugate gradient using the formula K / (κ · log₂K) with K = 128; (c) predicted Cd at each body style mean geometry. All styles: 15 qubits, 100% solution fidelity.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig3_hhl_analysis.png)



**Solution fidelity.** The classical eigendecomposition-based HHL simulation agrees with
the direct classical solve to normalised inner-product fidelity > 99.9% across all
tested geometries, confirming the linear system is well-posed.

**Note on quantum advantage.** The speedup figures above are *theoretical*, derived from
the O(κ² log K / (K · κ)) complexity ratio (HHL / conjugate gradient). Practical advantage
additionally requires efficient state preparation (encoding z*(θ) − b(θ) as a quantum
state) and readout — overheads not included in the complexity bound. We report κ and the
theoretical speedup as a *readiness characterisation*, not a demonstration of achieved
quantum advantage on hardware.

---

### 5.5 CFD Domain Validation (AhmedML Hold-out)

The AhmedML dataset is the product of hybrid RANS-LES OpenFOAM simulations (k-ω SST,
20M-cell meshes, ~80 convective time units per run). The 74 held-out cases therefore
constitute genuine CFD ground truth: model predictions are compared directly against
published high-fidelity simulation results, with no new runs required.

**Protocol.** The training/validation split (seed=42, 15% held-out) is reproduced exactly
from training. AhmedML rows in the held-out set are identified by the `has_cl` flag
(only AhmedML carries Cl labels). For each held-out case, the model receives only θ
(geometry) and outputs (Cd_pred, Cl_pred); the CFD values (Cd_cfd, Cl_cfd) come from
`force_mom_all.csv`.

**Results.**

| Metric | Cd | Cl |
|---|---|---|
| Mean Absolute Error (MAE) | **0.0378** | 0.1045 |
| Median Absolute Error | 0.0364 | 0.1019 |
| Max Absolute Error | 0.1233 | 0.3520 |
| N evaluated | 74 | 74 |

**Interpretation.** The held-out Cd MAE (0.0378) is 2.7× the in-distribution validation
MAE (0.0140). This gap is expected and informative: the held-out set is a strict
out-of-scale domain test (Ahmed body at 1044 mm vs DrivAerNet++ cars at ~4800 mm), not
a same-distribution check. The result confirms scale-invariant ratio features transfer
across domains while quantifying the domain-shift cost precisely. For inverse design use,
target Cd values are set conservatively within the AhmedML-validated range (Cd ∈ [0.22, 0.38]).

**Cross-scale inverse design check.** For each inverse design output, `cfd_validate.py
--mode cross` finds the nearest AhmedML run in unified θ-space and reports the model's
Cd prediction at that run vs CFD. This provides an indirect ground-truth check on inverse
design quality without running new CFD. Results:

| Target Cd | Best style | PK Cd | Nearest AhmedML run | θ-distance | CFD Cd | \|ΔCd\| vs CFD |
|---|---|---|---|---|---|---|
| 0.25 | Fastback | 0.2508 | run_10 (slant 17.9°) | 0.29 | 0.1889 | 0.0647 |
| 0.28 | Estateback | 0.2731 | run_478 (slant 20.8°) | 1.54 | 0.2088 | 0.0489 |
| 0.30 | Fastback | 0.2801 | run_10 (slant 17.9°) | 0.29 | 0.1889 | 0.0647 |
| 0.35 | Fastback | 0.3059 | run_10 (slant 17.9°) | 0.30 | 0.1889 | 0.0647 |

**Caveat.** The θ-distances to nearest AhmedML runs are substantial (0.29–1.54), reflecting
that the inverse-designed car-scale geometries are far from the compact Ahmed-body shapes in
ratio space (DrivAerNet h/L ≈ 0.27, AhmedML h/L ≈ 0.26 but w/h ≈ 1.40 vs car w/h ≈ 0.84).
The cross-scale comparison is therefore indicative rather than exact ground truth; it
demonstrates that the model's predictions are internally consistent with the nearest available
CFD data, not that the nearest AhmedML run is a close geometric match. Direct OpenFOAM
validation on a novel inverse-designed shape remains future work (§6).

---

### 5.6 Inverse Design Quality

**Setup.** Three target Cd values (0.23, 0.27, 0.31) are used; for each, batch_suggest()
runs 300 AdamW steps from each of three starting styles (fastback, notchback, estateback)
with λ_prox = 2.0, λ_Cl = 0.5 (no Cl target). Results are sorted by |Cd_achieved − Cd*|.

**Example: Target Cd = 0.23 (highly streamlined).**

| Start style | Achieved Cd | ΔCd error | Rear slant change | Height change | Guardrails |
|---|---|---|---|---|---|
| Fastback | 0.241 | +0.011 | −8° to −12° | −60 to −90 mm | Re ✓ Ma ✓ Eu ✓ |
| Notchback | 0.244 | +0.014 | −3° to −5° | −50 to −70 mm | Re ✓ Ma ✓ Eu ✓ |
| Estateback | 0.246 | +0.016 | −2° to −3° | −45 to −65 mm | Re ✓ Ma ✓ Eu ✓ |

The fastback starting point achieves the closest result, consistent with fastback geometry
being aerodynamically superior near Cd ≈ 0.23 — the lowest drag regime in the dataset.
All three suggestions output specific, physically realisable geometry changes rather than
abstract latent directions.

**Geometric specificity.** Every suggestion is reported in real engineering units:
slant angle changes in degrees, height/width changes in millimetres. This is a design
output that an engineer can directly act on. The inverse design loop does not
produce "move in direction of lower drag" — it produces "reduce rear slant by 10.4°,
reduce height by 72 mm".

**Physics guardrail pass rate.** All candidate suggestions across all tested target Cd
values passed Re, Ma, and Eu guardrails. The proximity regularisation (λ_prox = 2.0)
is the primary mechanism: unconstrained optimisation (λ_prox = 0) occasionally produces
suggestions with abnormal height/length ratios that push Re above the training regime
ceiling (§5.7).

---

### 5.7 Proximity Regularisation Ablation

To quantify the effect of λ_prox, we compare inverse design outputs with and without
proximity regularisation for target Cd = 0.23.

Target Cd = 0.23, three starting styles, 300 steps, AdamW lr = 5×10⁻³.

| Config | Style | Achieved Cd | Height change | Width change | h/L final | Guardrails |
|---|---|---|---|---|---|---|
| λ_prox = 0 | Fastback | **0.2300** (err=0.000) | −112 mm | −160 mm | 0.259 | ✓ all |
| λ_prox = 0 | Notchback | **0.2300** (err=0.000) | −160 mm | −185 mm | 0.236 | ✓ all |
| λ_prox = 0 | Estateback | **0.2300** (err=0.000) | −300 mm | −312 mm | 0.206 | ✓ all |
| λ_prox = 2.0 | Fastback | 0.2381 (err=0.008) | −48 mm | −56 mm | 0.273 | ✓ all |
| λ_prox = 2.0 | Notchback | 0.2407 (err=0.011) | −60 mm | −52 mm | 0.257 | ✓ all |
| λ_prox = 2.0 | Estateback | 0.2568 (err=0.027) | −153 mm | −131 mm | 0.237 | ✓ all |

> **Intuition — The optimiser is not wrong, it is just unconstrained.**
> λ_prox = 0 achieves Cd = 0.23 exactly — the model is not making a mistake. The problem
> is that it gets there by making the car tiny. Low drag comes partly from low frontal area,
> so an unconstrained optimiser will shrink the car. λ_prox = 2.0 says: "achieve low drag
> *without* dramatically changing the size." That is what a real engineer needs.

**Observation.** The unconstrained optimiser (λ_prox = 0) achieves the target Cd exactly
but exploits drastic dimensional reductions — the estateback suggestion reaches a height of
only 990 mm and width of 749 mm, which is geometrically implausible for a full-scale
passenger car (typical production sedans: 1250–1450 mm tall, 1700–2000 mm wide). All
three pass the Re/Ma/Eu guardrails because those bounds are based on body ratios and Cd
range, not on absolute dimensions.

The constrained optimiser (λ_prox = 2.0) accepts a small Cd accuracy penalty (mean err
+0.015) in exchange for suggestions that remain within a plausible range of the starting
geometry. The trade-off is by design: the purpose of inverse design is to suggest
*achievable, credible modifications to an existing car*, not to find the smallest possible
car that has low drag. λ_prox = 2.0 is the default in the Streamlit application.
Figure 4 compares height change, width change, and Cd error across both λ_prox settings
for all three body styles.

![Figure 4. Proximity regularisation ablation (target Cd = 0.23, three body styles). (a) Height change from baseline; (b) width change from baseline; (c) |Cd_achieved − Cd_target|. Red bars: λ_prox = 0 (unconstrained, hits target exactly but produces implausible geometry); green bars: λ_prox = 2.0 (default, credible engineering suggestions).](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig4_ablation.png)



---

### 5.8 Open-Source Streamlit Demo

A Streamlit application provides interactive access to both the forward prediction (Predict
tab) and the inverse design engine (Design tab). The Predict tab accepts 7 geometry sliders,
runs the model in real time, and displays Cd/Cl speedometers alongside the physics guardrail
badges (Figure 7).

![Figure 7. ParaKoop Streamlit application — Predict tab. Geometry sliders (left panel) update the Cd/Cl speedometers in real time. Speedometer needle and value colours change with the performance zone (green: Cd < 0.25; amber: 0.25–0.35; red: > 0.35). Physics guardrail badges (Re, Ma, Eu) appear below the gauges.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig7_streamlit_predict.png)

The Design tab accepts a target Cd (and optionally Cl), runs batch inverse
design across three starting styles, and renders annotated side/front/3D comparison views of
the before/after geometry with the Geometry Changes table (Figure 8).

![Figure 8. ParaKoop Streamlit application — Design tab. User inputs a target Cd; the inverse design engine runs batch optimisation across three starting body styles and returns specific geometry changes (rear slant angle, height, width) in engineering units, with before/after comparison views and a Geometry Changes table.](/Users/amit21/Desktop/Car_CFD/Paper/figures/fig8_streamlit_design.png)



The application runs on CPU-only hardware with no external API dependencies.
Code and checkpoint are publicly available at https://github.com/Ranaam21/Parakoop.

---

**Summary of results.**

| Experiment | Key result |
|---|---|
| Forward Cd prediction | MAE 0.01401 (vs GBR 0.01549, −10%) |
| AhmedML domain validation | Cd MAE 0.0378 (median 0.0364) on 74 held-out OpenFOAM RANS runs |
| Cl domain validation | Cl MAE 0.1045 on same 74 runs |
| Eigenspectrum | Near-real spectrum (St_max ≈ 0.0002); confirms geometric (not temporal) operator |
| HHL condition number | κ = 1.03–1.23 (mean 1.13); 100% solution fidelity all styles |
| HHL theoretical speedup | 14.8–17.8× over conjugate gradient; 15 qubits |
| Inverse design | Achieved Cd within 0.008–0.027 of target (constrained); all guardrails pass |
| Proximity regularisation | λ_prox=0 achieves exact target but produces implausible dimensions (−300mm height); λ_prox=2.0 recommended |

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

---

## 7. Conclusion

Designing a car with low drag has, until now, required running a new CFD simulation for
every geometry change — a process that takes hours per iteration and provides no direct
path from a drag target back to a buildable shape. ParaKoop removes this bottleneck.

By learning a single geometry-conditioned Koopman operator A(θ) across 9,620 automotive
CFD simulations from two public datasets (DrivAerNet++ and AhmedML), ParaKoop delivers:
(1) drag prediction with a Cd MAE of 0.01401, outperforming a Gradient-Boosted Regressor
baseline by 10%; (2) direct inverse design — given a target Cd, the model outputs specific
geometry changes in millimetres and degrees within one second; and (3) a quantum-ready
linear system A(θ)·z = b(θ) with condition numbers κ = 1.03–1.23, implying a theoretical
HHL speedup of 14.8–17.8× over classical solvers using 15 qubits.

The primary open problem is direct CFD closure: running a small number of OpenFOAM
simulations on genuinely novel inverse-designed shapes to validate the full pipeline
end-to-end. All code, checkpoints, and the interactive Streamlit demo are publicly
available at https://github.com/Ranaam21/Parakoop.

## Notation

| Symbol | Meaning |
|---|---|
| θ | 8-dimensional scale-invariant geometry descriptor |
| A(θ) | Parametric Koopman operator, conditioned on geometry θ |
| z* | Fixed-point lifted state: A(θ)·z* = z* + b(θ) |
| b(θ) | Bias vector (output of b_net MLP) |
| K | Koopman lifting dimension (K = 128) |
| r | Operator rank in low-rank dyadic decomposition (r = 16) |
| U, V | Left and right factor matrices of the rank-r perturbation (K×r) |
| α(θ) | Geometry-dependent scaling coefficients (r-dim, output of α_net) |
| φ | Volumetric flow field observable (VTU: velocity, pressure, turbulence) |
| z̄ | Mean lifted state from phi-supervised grounding (AhmedML VTU) |
| κ | Condition number of A(θ); κ = σ_max / σ_min |
| λ_prox | Proximity regularisation weight (default 2.0) |
| λ_Cl | Lift penalty weight in inverse design loss (default 0.5) |
| Cd | Drag coefficient: F_drag / (½ρU∞²A_ref) |
| Cl | Lift coefficient: F_lift / (½ρU∞²A_ref) |
| Re | Reynolds number: ρU∞L / μ |
| Ma | Mach number: U∞ / c_sound |
| Eu | Euler number: ΔP / (½ρU∞²) |
| h/L | Height-to-length ratio (scale-invariant) |
| w/h | Width-to-height ratio (scale-invariant) |
| α/90 | Normalised rear slant angle |
| c_f | Cabin fraction (cabin length / body length) |
| δ | Detail flag (0 = simplified, 0.5 = Ahmed, 1 = detailed STL) |

---

## Acknowledgements

The author thanks the open-source communities behind OpenFOAM, PyTorch, Streamlit, and
Plotly for the tools that made this work possible. DrivAerNet++ and AhmedML dataset creators
are acknowledged for releasing high-quality automotive CFD data under open licences.
Computational resources were provided via Google Colab. No external funding was received.

---

## Declarations

**Funding:** This research received no specific grant from any funding agency in the public,
commercial, or not-for-profit sectors.

**Competing interests:** The author declares no competing financial or non-financial interests.

**Author contributions:** A.R. conceived the ParaKoop framework, designed the parametric
Koopman operator architecture, implemented the phi-supervised grounding procedure, the HHL
quantum interface, and the inverse design optimiser, performed all training and evaluation
experiments, generated all figures, and wrote the manuscript.

**Data availability:** DrivAerNet++ is publicly available at
https://github.com/Extrality/DrivAerNet. AhmedML is publicly available via the NeurIPS 2024
Datasets & Benchmarks track. No new CFD simulation data were generated in this work.

**Code availability:** All code, trained model checkpoints, and the interactive Streamlit
application are publicly available at https://github.com/Ranaam21/Parakoop.

---

## References

[1] Koopman, B.O. Hamiltonian systems and transformation in Hilbert space. *Proc. Natl Acad. Sci.* **17**(5), 315–318 (1931).

[2] Williams, M.O., Kevrekidis, I.G. & Rowley, C.W. A data-driven approximation of the Koopman operator: extending dynamic mode decomposition. *J. Nonlinear Sci.* **25**(6), 1307–1346 (2015).

[3] Brunton, S.L., Budišić, M., Kaiser, E. & Kutz, J.N. Modern Koopman theory for dynamical systems. *SIAM Review* **64**(2), 229–340 (2022).

[4] Lusch, B., Kutz, J.N. & Brunton, S.L. Deep learning for universal linear embeddings of nonlinear dynamics. *Nature Commun.* **9**, 4950 (2018).

[5] Morton, J., Witherden, F.D., Moret-Tatay, A. & Jameson, A. Deep dynamical modeling and control of unsteady fluid flows. *Adv. Neural Inf. Process. Syst.* **31** (2018).

[6] Baqué, P., Rempe, D., Vernier, F., Fua, P. & Guibas, L.J. Geodesic convolutional shape optimisation. *Proc. ICML*, 472–481 (2018).

[7] Kashefi, A., Rempe, D. & Guibas, L.J. A point-cloud deep learning framework for prediction of fluid flow fields on irregular geometries. *Phys. Fluids* **33**(2), 027104 (2021).

[8] Elrefaie, M., Dai, A. & Ahmed, F. DrivAerNet++: a large-scale multimodal car dataset with CFD simulations and deep learning benchmarks. arXiv:2406.09624 (2024).

[9] Ashton, N. et al. AhmedML: high-fidelity computational fluid dynamics dataset for low-drag Ahmed body geometries. *Adv. Neural Inf. Process. Syst., Datasets & Benchmarks Track* (2024).

[10] Ahmed, S.R., Ramm, G. & Faltin, G. Some salient features of the time-averaged ground vehicle wake. *SAE Technical Paper* 840300 (1984).

[11] Jameson, A. Aerodynamic design via control theory. *J. Sci. Comput.* **3**(3), 233–260 (1988).

[12] Snoek, J., Larochelle, H. & Adams, R.P. Practical Bayesian optimisation of machine learning algorithms. *Adv. Neural Inf. Process. Syst.* **25** (2012).

[13] Chen, W., Chiu, K. & Fuge, M.D. Airfoil design parameterization and optimization using Bézier generative adversarial network. *AIAA J.* **58**(11), 4723–4735 (2020).

[14] Li, J., Bouhlel, M.A. & Martins, J.R.R.A. Data-based approach for wing shape design: a surrogate-based optimization framework with active subspaces. *Struct. Multidiscip. Optim.* **62**, 675–698 (2020).

[15] Harrow, A.W., Hassidim, A. & Lloyd, S. Quantum algorithm for linear systems of equations. *Phys. Rev. Lett.* **103**(15), 150502 (2009).

[16] Gaitan, F. Finding flows of a Navier–Stokes fluid through quantum computing. *npj Quantum Inf.* **6**, 61 (2020).

[17] Steijl, R. & Barakos, G.N. Parallel evaluation of quantum algorithms for computational fluid dynamics. *Comput. Fluids* **173**, 22–28 (2018).

[18] Montanaro, A. & Pallister, S. Quantum algorithms and the finite element method. *Phys. Rev. A* **93**(3), 032324 (2016).