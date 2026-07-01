# Section 2: Related Work

<!-- Draft status: complete. Follows CFD Paper Instructions.docx:
     simple language, full acronyms, bold subheadings, inline citations [n].
     Three subsections: Koopman methods, aerodynamic surrogates, quantum solvers.
     Each entry states what the work does, what gap it leaves, how ParaKoop fills it. -->

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

