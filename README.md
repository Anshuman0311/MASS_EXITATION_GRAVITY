# Gravitational Soliton Search in Nonminimally Coupled Scalar–Tensor Gravity

**Author:** Anshuman Mishra  
BS-MS (Mathematics & Data Science), HBTU Kanpur, India  

This repository contains the full numerical code and data used in the paper:

> **“Can Mass Arise as a Gravitational Field Excitation?  
> A Rigorous Test in Nonminimally Coupled Scalar–Tensor Gravity”**

The project investigates whether inertial mass can emerge as a localized,
self-sustained excitation of the gravitational field, in close analogy with
photons as excitations of the electromagnetic field.

---

## 🧠 Physical Question

Can gravity, possibly modified by a nonminimally coupled scalar field, support
static, regular, asymptotically flat soliton-like configurations that carry
finite ADM mass — and could such objects be interpreted as purely gravitational
realizations of “mass”?

---

## 📐 Theoretical Model

We study the Jordan-frame action:
\[
S = \int d^4x \sqrt{-g} \left[
F(\phi) R - \frac12 (\nabla \phi)^2 - V(\phi)
\right],
\]
with  
\[
F(\phi) = \frac{1}{8\pi G} - \xi \phi^2, \quad
V(\phi) = \frac12 \mu^2 \phi^2.
\]

Assuming static spherical symmetry, the field equations reduce to a stiff system
of coupled nonlinear ODEs for metric functions \(A(r), B(r)\) and scalar field
\(\phi(r)\), solved as a two-point boundary value problem.

---

## 🖥️ Numerical Method

- Solver: `scipy.integrate.solve_bvp` with adaptive mesh refinement  
- Boundary conditions:
  - Regular origin: \(B(0)=1\), \(\phi'(0)=0\)
  - Asymptotic flatness: \(A(\infty)=1\), \(\phi(\infty)=0\)
- ADM mass extracted from:
  \[
  M(r) = \frac{r}{2}\left(1 - \frac{1}{B(r)}\right).
  \]
- True residual check:
  \[
  \max_r \lvert dy/dr - f(r,y) \rvert < 10^{-6}.
  \]

A convergence study with increasing grid resolution verifies numerical stability.

---

## 🔍 Parameter Scan

Systematic scan over:
- Nonminimal coupling: \(\xi \in [-20, 20]\)
- Scalar mass:     \(\mu \in [0.1, 2.0]\)
- Central amplitude:  \(\phi(0) \in [10^{-4}, 0.1]\)

Typical domain size: \(r_{\max} \sim 50/\mu\),  
Grid points: up to \(N \sim 3000\).

Total runtime (full scan): **~10.4 hours** on a standard laptop.

---

## 📊 Main Result

> **No nontrivial, regular, asymptotically flat solutions with finite ADM mass
were found.**

Across the entire scanned parameter space:
\[
\phi(r) \equiv 0, \quad A(r)=B(r)\equiv 1, \quad M_{\rm ADM} \approx 0.
\]

All solutions relax to the trivial Minkowski vacuum.

This provides strong numerical evidence against the existence of static
gravitational solitons in this class of scalar–tensor theories, supporting the
main conclusion of the paper.

---

## 📁 Repository Contents

Generated: **2025-12-28**  
Total execution time: **37501 s (~10.4 hours)**

### Files and folders:

- `scan_results.csv` — Complete parameter scan results  
- `summary.json` — Summary statistics of convergence and solutions  
- `conclusion.json` — Final research conclusion  
- `plots/` — Scalar profiles, mass functions, phase diagrams  
- `data/` — Raw solution arrays (HDF5)  
- `*.py` — Solver, scan, and analysis scripts  

---

## ▶️ How to Run

```bash
python main_solver.py
