# Section 0: Abstract

<!-- Draft status: complete. ≤250 words. Written last per PAPER_PLAN order.
     Required: "parametric Koopman", "inverse design", "HHL", "κ", specific MAE numbers.
     Follows CFD Paper Instructions.docx: plain language, no jargon without brief gloss. -->

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
