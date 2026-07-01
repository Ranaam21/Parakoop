# Section 1: Introduction

<!-- Draft status: complete. Follows CFD Paper Instructions.docx:
     simple language, full acronyms, intuitions, forward/inverse distinction,
     novelty called out, industry use cases, inline citation placeholders [n]. -->

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

