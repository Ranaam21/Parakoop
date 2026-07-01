# Section 7: Conclusion

<!-- Draft status: complete. ≤200 words. Core claim + 3 key numbers + one open problem.
     Follows CFD Paper Instructions.docx: plain language, no new content not in body. -->

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
