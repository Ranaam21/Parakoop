# Paper 4 — ParaKoop Design Rationale: A Decision Log (Q&A)

**Purpose of this document:** This is a structured record of the design
discussion that produced the ParaKoop architecture described in
`Paper4_Automotive_Koopman_HHL_ParaKoop_Scope.txt`. Each entry below states a
question that came up while strengthening the original draft
(`Paper3_Automotive_Koopman_HHL_Scope.txt`), the alternatives that were on the
table, the choice that was made, and the reasoning behind it. This is written
so it can be lifted nearly directly into the paper's **Methods**, **Discussion**,
and **Limitations** sections later — "why we chose X over Y, and what we ruled
out and why" is exactly the material reviewers look for and authors usually
have to reconstruct from memory months later. Better to write it down now,
while the reasoning is fresh.

Cross-reference: every decision below maps to a specific section of
`Paper4_Automotive_Koopman_HHL_ParaKoop_Scope.txt` (cited inline as "Scope §x.x").

--------------------------------------------------------------------------------

## Q1. Why move from "Koopman validated on a disconnected canonical flow" to
## "Koopman as the surrogate's actual mechanism"?

**The question:** The original draft (Paper3_Automotive_Koopman_HHL_Scope.txt,
§3 and §5.1-5.3) validated Koopman/EDMD and HHL on a self-generated cylinder
wake — a clean, well-understood textbook flow — and *separately* trained a
GNN to regress Cd/Cl on DrivAerNet++, searched with a GA. The two halves were
connected only by a narrative "unified claim" ("Koopman bridges chaos theory
to quantum computing, and our surrogate makes that practical"). Is that
connection strong enough to survive review?

**Alternatives considered:**
- *(a) Keep the two-track structure, strengthen the narrative bridge.* Write a
  more careful Discussion section explaining how the cylinder validation
  "informs" the surrogate design.
- *(b) Drop the Koopman/HHL component entirely* and frame the paper purely as
  a topology-search/inverse-design contribution (the GNN + GA + VICES side).
- *(c) Make Koopman the literal mechanism of the surrogate itself*, so the
  question "what does Koopman do for the cars you're optimising?" has a
  structural, not narrative, answer.

**Choice made:** (c).

**Reasoning:** A reviewer's first question on the original structure would be
exactly "what does the cylinder-wake Koopman analysis have to do with the
8,000 DrivAerNet cars you're actually optimising?" — and under structure (a),
the honest answer is "nothing mechanistic; it's a validation of the *method*,
applied to a *different* problem." That's a defensible position (papers do
validate methods on canonical cases before scaling), but it's a *weaker* claim
than it could be, and it invites the obvious follow-up: "then why not validate
directly on the cars?" Option (b) throws away a genuinely interesting
quantum-computing angle that connects to the QuantumRAG work. Option (c) — making
the Koopman operator literally *be* the surrogate (i.e., reading Cd, Cl, modal
structure, and the HHL input system all off one learned operator A(theta)) —
removes the question entirely: there is no "bridge" to defend, because the
same object does both jobs. This was the single biggest structural change from
the original draft (Scope §2, "What v2 does instead").

--------------------------------------------------------------------------------

## Q2. Why use AhmedML instead of self-generating a bridging transient dataset?

**The question:** DrivAerNet++ is steady-state only — it cannot teach a model
how geometry determines *dynamics* (no time series). The original draft
recognised this gap and proposed validating the Koopman/HHL machinery on a
single self-generated cylinder-wake case (Scope v1 §3, "cylinder wake
OpenFOAM case"). But a single geometry can't teach *geometry-conditioning* —
you cannot learn how an operator varies with shape by observing only one
shape. Some bridging dataset with multiple transient geometries was needed.
Generate it ourselves, or find one?

**Alternatives considered:**
- *(a) Self-generate ~10-20 transient cases* spanning a small geometric
  parameter sweep (e.g., varying an Ahmed-body slant angle), using the
  existing OpenFOAM Docker pipeline from Paper 1.
- *(b) Search for an existing public dataset that already provides multiple
  geometries x transient CFD.*

**Choice made:** (b) — specifically AhmedML (Ashton et al. 2024,
arXiv:2407.20801): 500 geometric variants of the Ahmed body, each run
*transiently* with hybrid RANS-LES, ~80 convective time units per case,
~20M-cell meshes, released CC-BY-SA for free download.

**Reasoning:** Self-generation (a) carries three costs that a literature
search might make unnecessary: (i) *infrastructure burden* — 10-20 transient
runs at ~6-48 HPC-hours each is itself a significant compute project sitting
inside a bigger one; (ii) *small-N risk* — 10-20 geometries is a thin basis for
learning a *map* from geometry to operator (the thing v2's whole architecture
depends on); (iii) *opportunity cost* — every hour spent generating bridging
data is an hour not spent on the actual novel contribution (the inverse-design
mechanism). A targeted search turned up AhmedML, which removes all three costs
simultaneously: it is free, already validated/published, and at 500 geometries
x full transient series is *two orders of magnitude* richer than the
self-generation plan would have produced. This is the clearest "found a better
path than the one we were about to take" moment in the whole redesign — see
Scope §3 ("Dataset A") and §4 for the full comparison.

--------------------------------------------------------------------------------

## Q3. Why a *parametric* Koopman operator A(theta) instead of a *temporally-
## generalized* one?

**The question:** This was the most technically subtle correction in the whole
redesign, raised directly: "Koopman for steady state might not give anything
unless we use parametrized Koopman instead of temporal based." The original
draft's implicit plan — calibrate a (temporal) Koopman operator
`z_{t+1} = A . z_t` on the cylinder wake, then somehow have it (or a learned
embedding around it) generalise to thousands of geometrically distinct,
steady-state-only DrivAerNet cars — has a structural hole: a temporal operator
literally has nothing to act on when there is no time axis in the data, and
"generalising" an operator calibrated on ONE laminar flow to industrial car
shapes via a generic embedding is an unsupported leap, not an architecture.

**Alternatives considered:**
- *(a) Temporal operator + generic geometry embedding*, hoping the embedding
  bridges the gap (the structurally flawed default — what a naive
  generalisation of the original draft would produce).
- *(b) Two separate operators* — one temporal (for AhmedML/dynamics), one
  "steady" (for DrivAerNet/statics) — stitched together post hoc.
- *(c) One PARAMETRIC operator* `theta -> A(theta)`: the object being learned
  is not "how does the flow evolve in time" but "how does the operator itself
  change as a function of structured geometry parameters." Steady-state
  performance is then the *fixed point* `A(theta).x = b(theta)` of this
  geometry-conditioned operator — no time axis required.

**Choice made:** (c).

**Reasoning:** Option (a) is the flaw as originally stated — restating it with
an embedding doesn't fix the underlying mismatch between what a temporal
operator represents and what steady data contains. Option (b) avoids the
mismatch but reintroduces exactly the "two disconnected tracks" problem from
Q1, just one level down — now there are two operators with a narrative bridge
between them instead of one surrogate with a narrative bridge to Koopman.
Option (c) dissolves the mismatch rather than working around it: by making the
*function from geometry to operator* the learned object, both transient
behaviour (AhmedML: read off A(theta)'s eigenstructure) and steady behaviour
(DrivAerNet++: read off A(theta)'s fixed point) become two different *readouts*
of the *same* learned map, calibrated jointly. This is also what makes
inversion possible later (Q5) — `theta -> A(theta) -> performance` is a single
differentiable chain, not two operators glued together. See Scope §2 (Step 2)
and §5.2-5.3 for the full mechanism.

--------------------------------------------------------------------------------

## Q4. Why wake-recirculation strength as a first-class design objective
## (alongside Cd and L/D)?

**The question:** When asked to clarify what "flow rate" should mean as a
design-optimisation objective, the answer given was specific: "wake mass-flux
/ recirculation strength" — connected to the von Karman vortex street. Is this
worth elevating to a first-class objective alongside the standard Cd/Cl/L-D
trio, given that it requires substantially more modelling machinery (full
transient wake fields and their modal decomposition) than a scalar regression
target does?

**Alternatives considered:**
- *(a) Skip it — stick to the standard Cd/Cl/L-D objectives* that every
  automotive aero paper reports, keeping the optimisation problem simple and
  the comparison to prior work direct.
- *(b) Report wake structure as a descriptive/diagnostic add-on* (e.g., "and
  here's what the modes look like for our top designs") without making it a
  search objective.
- *(c) Make it a first-class THIRD objective* in the GA's fitness function and
  in the inverse-design target specification.

**Choice made:** (c).

**Reasoning:** Two things make this worth the extra modelling cost. First,
*it is structurally free* in the ParaKoop architecture: once A(theta)'s
eigenvalues/eigenvectors are being computed anyway (to get modal/transient
behaviour for the AhmedML calibration, Q3), the steady recirculation-bubble
size (from the fixed-point field) and the unsteady shedding-mode energy (from
the spectrum) fall out as a byproduct — no separate model is needed. Second,
*it is a quantity no scalar Cd/Cl surrogate can produce at all*, and it is
not merely academic: wake recirculation directly drives base-pressure drag,
EV battery thermal/cooling performance (hot recirculated air re-ingestion into
underbody cooling intakes), cabin acoustics/buffeting, and rear-end soiling —
a noticeably richer connection to real engineering decisions than Cd alone.
Reporting it only as a diagnostic (b) would waste this "free" capability;
making it a search objective (c) turns a structural byproduct into a genuine
selling point — a 3-objective automotive design search that is, to our
knowledge, not otherwise possible without the kind of model ParaKoop is. See
Scope §5.4 and §8 (Industry Use Cases).

--------------------------------------------------------------------------------

## Q5. Why the hybrid (grammar-conditioned generation + gradient-based
## refinement) over a pure-gradient or pure-generative inverse-design mechanism?

**The question:** Once the decision was made to flip the architecture so that
geometry is the *output* ("like in modelling y is the unknown variable and x
is known — here Y is geometry which is unknown") and to build on the
VICES/Catalan structured design space from Paper 1, what mechanism should
actually produce the suggested geometries — and should it favour precision
(staying close to known-good designs) or novelty ("out of the box" suggestions
that still meet every guardrail)?

**Alternatives considered:**
- *(a) Pure gradient-based (latent/test-time optimisation):* treat geometry
  generation as continuous optimisation — start from some point in design
  space and descend toward the target performance using the differentiable
  `theta -> A(theta) -> performance` chain.
- *(b) Pure grammar-conditioned generative model:* a generative model over the
  Catalan CSG-tree grammar, conditioned on target performance, that samples
  candidate topologies directly.
- *(c) Hybrid — grammar-conditioned generation for the discrete topology,
  gradient-based refinement for the continuous parameters within a fixed
  topology.*

**Choice made:** (c) — explicitly preferred ("I liked: Best — a hybrid:
grammar-conditioned generation for the discrete structure, gradient-based
refinement for the continuous parameters within it").

**Reasoning:** Pure gradient-based search (a) has two weaknesses for this
problem specifically: topology choice (which primitives, which boolean
operations, what arrangement) is *inherently discrete*, so gradient descent
needs a fragile continuous relaxation to even engage with it; and gradient
descent is fundamentally *local* — it refines toward a nearby optimum, which
works against the explicit goal of "out-of-the-box" suggestions that explore
the topology gap (the central argument carried over from Paper 1). Pure
generative (b) solves the discreteness and diversity problems — grammar-
constrained decoding guarantees every sample is a syntactically valid,
constructible CSG tree, and a novelty term can explicitly reward
topology-grammar diversity from the training distribution — but it leaves
unused the one thing this architecture uniquely provides: a differentiable
chain from geometry to performance that can give precise local guidance. The
hybrid (c) is the only option that uses *both* halves of what's been built:
propose broadly and validly with the grammar (diversity, validity, novelty —
the side that's already built and validated from Paper 1's VICES work),
then refine precisely with gradients (precision — the side the new
parametric-Koopman architecture uniquely supplies). It is also the only option
where guardrail compliance can be enforced *structurally* rather than
statistically: grammar-constrained decoding guarantees syntactic validity, and
a hard reject-and-resample check on physical guardrails (Re, Ma, Eu, Cd_range,
L/D_range) catches anything the generative model might otherwise learn to
"cheat" at the tails of a soft penalty. See Scope §5.6 for the full mechanism
and the structural-vs-statistical guardrail argument.

--------------------------------------------------------------------------------

## Q6. Why report kappa(A(theta)) as a single "triple-duty" diagnostic rather
## than three separate metrics?

**The question:** The condition number kappa of the learned operator A(theta)
turns out to be informative in (at least) three completely different ways —
about the flow itself, about whether quantum acceleration helps, and about
whether the inverse-design refinement step can be trusted. Should the paper
report these as three independent quantities computed from the same number
(which would be redundant and might look like padding), or unify them?

**The three roles, identified through discussion:**
1. *Flow-regime characterisation* — low kappa indicates simpler/more laminar
   behaviour, high kappa indicates complex/turbulent behaviour (a standard
   numerical-analysis reading of conditioning).
2. *Quantum-solvability / HHL speedup potential* — kappa in the range ~10-100
   is where HHL's theoretical O(kappa^2 log N) speedup is attractive; kappa
   >> 1000 erodes the advantage (a standard quantum-algorithms reading).
3. *Gradient-refinement trustworthiness* (newly recognised in this
   discussion) — a well-conditioned A(theta) means the local
   performance-vs-parameter landscape that Stage B of the inverse-design
   mechanism (Q5) descends on is smooth and well-behaved, i.e., the gradient
   signal can be trusted; a poorly-conditioned A(theta) is a warning that
   gradient refinement near that candidate may be unreliable.

**Choice made:** Report kappa(A(theta)) as one unifying diagnostic with three
named roles, rather than three separate metrics that happen to share a value.

**Reasoning:** These three readings are not three coincidentally-correlated
numbers — they are three *consequences of the same underlying mathematical
property* (how A(theta) distorts vectors / how sensitive its solution is to
perturbation), surfacing in three different parts of the same pipeline (the
surrogate's physics, the quantum solver, and the inverse-design refiner).
Presenting them as one diagnostic is more accurate (it doesn't imply three
independent measurements), more elegant (reviewers notice when a single
quantity does real structural work rather than being reported for its own
sake), and it opens a genuinely new research question worth stating explicitly
in the Discussion: *does shape topology systematically affect Koopman-operator
conditioning — and if so, does that simultaneously determine which shapes are
"quantum-friendly" to evaluate AND which are "gradient-friendly" to refine
during design search?* That question doesn't exist if kappa is reported as
three disconnected numbers. See Scope §5.4 ("kappa's three roles").

--------------------------------------------------------------------------------

## Q7. Why does VICES/Catalan serve double duty as both the inverse-design
## OUTPUT representation and the CONDITIONING structure for A(theta)?

**The question:** Paper 1's VICES (CSG/SDF synthesis over a Catalan-enumerated
topology grammar) was built and validated for a different purpose — generating
ALD showerhead geometries for a forward surrogate to evaluate. The explicit
instruction for this work was to "ensure we use previous work like Vices/
Catalans topology as well." Is reusing it as one input representation enough
to satisfy that, or is there a more central role for it?

**Alternatives considered:**
- *(a) Use VICES only to generate the novel candidate geometries that get fed
  through the (separately-built) forward surrogate* — i.e., reuse it as a
  geometry-generation utility, structurally peripheral to the main model.
- *(b) Use VICES/Catalan as the literal STRUCTURED PARAMETER SPACE theta that
  the parametric Koopman operator A(theta) is conditioned on* — making it
  central to the surrogate itself — *and simultaneously* as the representation
  the inverse-design model decodes its suggestions into, guaranteeing every
  suggestion is syntactically valid and constructible by construction.

**Choice made:** (b) — explicit double duty.

**Reasoning:** Option (a) would technically satisfy "use previous work," but
it under-uses what was actually built: VICES/Catalan is not just a geometry
generator, it is a *structured, combinatorial, already-validated description
of the design space* — exactly the kind of object a parametric operator needs
to be conditioned on (Q3) and exactly the kind of object a generative
inverse-design model needs to decode into to *guarantee* validity (Q5). Using
it for only one of these roles would mean either (i) building a second,
redundant structured representation for the other role, or (ii) using an
unstructured representation for the other role and losing its guarantees.
Recognising that the *same* representation can do both jobs is what makes the
guardrail-compliance argument in Q5 "structural rather than statistical" — the
grammar's validity guarantee and the operator's conditioning structure are the
same object wearing two hats. This is also a clean, genuine instance of
cross-domain infrastructure reuse (ALD -> automotive) that strengthens rather
than dilutes the paper's claims about Paper 1's formalism being broadly
applicable. See Scope §5.1 and §5.7, and Novelty Claim 7 (Scope §8).

--------------------------------------------------------------------------------

## Q8. Why did OpenFOAM's role shrink from data-generation to validation-only —
## and why is that a feature rather than a gap?

**The question:** The original draft used the existing OpenFOAM Docker
pipeline (reused from Paper 1) to *generate* the bridging transient training
data (the cylinder-wake case). Once AhmedML was found to supply that role for
free (Q2), and DrivAerNet++ already supplies the steady-state training data,
what is left for OpenFOAM to do — and is a shrunken role for a piece of
already-built infrastructure something to be concerned about?

**Alternatives considered:**
- *(a) Keep generating substantial new training data with OpenFOAM* regardless
  — e.g., expand the bridging-dataset plan to make better use of the existing
  pipeline investment.
- *(b) Reduce OpenFOAM's role to ground-truth VALIDATION ONLY* — a small
  number of runs (mirroring Paper 1's CFD_val_rank{1,2,3} validation cases) on
  the top novel suggested-and-refined topologies that the inverse-design loop
  produces, i.e., shapes that exist in *neither* AhmedML nor DrivAerNet++.

**Choice made:** (b).

**Reasoning:** Generating more data with OpenFOAM (a) just to "use the
pipeline" would be solving a problem that no longer exists — both training-data
needs (transient dynamics, steady-state scale) are now covered by free,
larger, already-validated public datasets (Q2). Continuing to generate data
anyway would reintroduce the cost, risk, and opportunity-cost issues that
finding AhmedML was specifically meant to avoid. The shrunken role (b) is
*exactly* the right-sized role for OpenFOAM in this architecture: its job is
no longer to teach the model anything (the datasets do that) but to answer the
one question the datasets structurally cannot — "does the pipeline's
prediction hold on a genuinely novel shape that has never been simulated
before?" That is precisely what a small ground-truth validation round answers,
and it mirrors how Paper 1 used CFD_val_rank cases (validate top candidates
from the search, not generate the training set). The net effect is a *smaller*
CFD infrastructure burden than the original draft required, freeing real
effort for the part of the project that is actually novel — the architecture
and the inverse-design mechanism (Q1, Q5). A shrinking role for one reused
component, in service of a stronger overall design, is a sign the redesign is
well-targeted, not a gap to explain away. See Scope §5.10.

--------------------------------------------------------------------------------

## Q9. Why are Strouhal-number validation and HHL/kappa validation kept as two
## SEPARATE checks rather than combined into one "Koopman/HHL validation" step?

**The question:** The pipeline runs two validations that both touch the
operator A(theta) — (i) checking that A(theta)'s extracted modal frequencies
match known Strouhal numbers from the Ahmed-body literature (Scope §5.2), and
(ii) checking A(theta)'s condition number kappa and the theoretical HHL
speedup it implies (Scope §5.5). Since both are "checks on A(theta)," should
they be merged into a single combined validation step to keep the Results
section tighter — or does keeping them apart matter?

**Alternatives considered:**
- *(a) Merge them* into one "Koopman/HHL validation" section — report Strouhal
  match and kappa/speedup together as a combined "is the operator any good?"
  verdict.
- *(b) Keep them strictly separate*, but state explicitly — in the paper, not
  just in internal notes — WHY they're separate and how they relate: Strouhal
  validation certifies that A(theta) is a physically faithful operator (a
  spectral/modal correctness check on the INPUT); kappa/speedup analysis
  characterises how that operator behaves as a quantum-linear-algebra object
  (a numerical-conditioning check on HHL's BEHAVIOUR on that input).

**Choice made:** (b).

**Reasoning:** Merging them (a) would actually make the paper's quantum claim
*weaker*, not tighter — because it would blur two logically different
questions into one number. "Is A(theta) physically correct?" (Strouhal) and
"does solving A(theta).x = b(theta) on a quantum computer offer an advantage,
and is that advantage trustworthy?" (kappa/speedup) are independent questions
with independent failure modes: an operator can be physically faithful (good
Strouhal match) yet poorly conditioned (kappa too high for HHL to help, e.g.
in strongly turbulent regimes) — or, in principle, well-conditioned yet
*wrong* (a spuriously well-behaved operator that doesn't actually capture the
physics, which would make a "good kappa" result meaningless). Reporting them
together risks exactly the kind of conflation a sharp reviewer will probe:
"how do you know the systems your HHL analysis is solving aren't numerically
convenient nonsense?" Keeping them separate — and stating the DEPENDENCY
explicitly (Strouhal validation is what *earns the right* to trust the kappa/
speedup analysis; kappa/speedup analysis is only meaningful *given* that
trust) — turns a potential weak point into a demonstration of rigor: it shows
the paper has already asked, and answered, the question a reviewer would
otherwise have to raise. This dependency is now stated directly in Scope §5.5
("VALIDATION DEPENDENCY"), with a pointer back to §5.2.

--------------------------------------------------------------------------------

## Cross-reference summary

| Decision | Resolves | Scope doc section |
|---|---|---|
| Q1 — Koopman as mechanism, not validation | The "two disconnected tracks" weakness | §2, §5.2-5.5 |
| Q2 — AhmedML over self-generated data | The need for multi-geometry transient training data | §3 (Dataset A), §4 |
| Q3 — Parametric A(theta), not temporal | The steady/temporal mismatch | §2 (Step 2), §5.2-5.3 |
| Q4 — Wake recirculation as first-class objective | "out of the box yet meeting all guardrails" + richer business story | §5.4, §8 |
| Q5 — Hybrid grammar + gradient inverse design | "Geometry should be the output... out of the box yet meeting all guardrails" | §5.6 |
| Q6 — kappa as triple-duty diagnostic | Avoiding redundant reporting; opens a new research question | §5.4 |
| Q7 — VICES/Catalan double duty | "ensure we use previous work like Vices/Catalans" | §5.1, §5.7, Novelty Claim 7 |
| Q8 — OpenFOAM validation-only | Right-sizing reused infrastructure to the new architecture | §5.10 |
| Q9 — Strouhal vs. kappa/HHL kept as separate validations | Pre-empting "are the systems HHL solves physically meaningful?" | §5.2, §5.5 ("VALIDATION DEPENDENCY") |

================================================================================
END OF DESIGN-RATIONALE Q&A — companion to Paper4_Automotive_Koopman_HHL_ParaKoop_Scope.txt
================================================================================
