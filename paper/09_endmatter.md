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
