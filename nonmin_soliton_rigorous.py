"""
gravitational_soliton_solver.py

Rigorous numerical solver for static, spherically symmetric solutions
in nonminimally coupled scalar-tensor gravity.

Implementation of field equations from:
"Can Mass Arise as a Gravitational Field Excitation?"
Anshuman Mishra, HBTU Kanpur
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import json
import h5py
import os
from datetime import datetime
from pathlib import Path

# ==================== SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Physical constants (geometric units: G = c = 1)
G_N = 1.0
F0 = 1.0 / (8.0 * np.pi * G_N)  # Bare gravitational coupling

# ==================== DATA STRUCTURES ====================
@dataclass
class SolitonParameters:
    """Container for model parameters."""
    xi: float        # Non-minimal coupling
    mu: float        # Scalar mass [M_pl]
    sigma0: float    # Central field value
    r_max: float     # Domain radius
    n_grid: int      # Grid points
    
    def __post_init__(self):
        assert self.mu > 0, "Mass must be positive"
        assert self.r_max > 0, "Domain must be positive"
        assert self.n_grid > 10, "Grid too coarse"

@dataclass
class SolitonSolution:
    """Container for converged solution."""
    params: SolitonParameters
    r: np.ndarray
    A: np.ndarray      # g_tt component
    B: np.ndarray      # g_rr component
    phi: np.ndarray    # Scalar field
    phi_prime: np.ndarray
    converged: bool
    residual_norm: float
    iterations: int
    
    @property
    def mass_function(self) -> np.ndarray:
        """Compute M(r) = r/2 * (1 - 1/B(r))"""
        return 0.5 * self.r * (1.0 - 1.0 / self.B)
    
    @property
    def adm_mass(self) -> float:
        """Arnowitt-Deser-Misner mass"""
        return float(self.mass_function[-1])

# ==================== FIELD EQUATIONS ====================
def field_equations(r: np.ndarray, y: np.ndarray, params: SolitonParameters) -> np.ndarray:
    """
    ODE system for static spherical symmetry.
    y = [A, B, phi, phi_prime]
    Returns dy/dr where dA/dr = A*(A'/A) and dB/dr = B*(B'/B)
    """
    A, B, phi, p = y
    
    # Regularization
    eps = 1e-15
    r_safe = np.maximum(r, eps)
    A_safe = np.maximum(A, eps)
    B_safe = np.maximum(B, eps)
    
    # Coupling functions
    F = F0 - params.xi * phi**2
    F_prime = -2.0 * params.xi * phi * p
    V = 0.5 * params.mu**2 * phi**2
    
    # Denominator (regularized)
    denom = F_prime + 2.0 * F / r_safe
    denom = np.where(np.abs(denom) < eps, eps * np.sign(denom), denom)
    
    # Paper Equations (16)-(17): A'/A and B'/B expressions
    Apr_over_A = (p**2 - 2*B_safe*V - 4*F_prime/r_safe - 2*F*(B_safe-1)/r_safe**2) / denom
    Bpr_over_B = (p**2 + 2*B_safe*V + 4*F_prime/r_safe + 2*F*(B_safe-1)/r_safe**2) / denom
    
    # Now multiply by A and B to get actual derivatives
    dA_dr = A_safe * Apr_over_A
    dB_dr = B_safe * Bpr_over_B
    
    # Ricci scalar (for scalar equation)
    R = (2*(1 - 1/B_safe)/r_safe**2 
         - 2*Apr_over_A/(B_safe * r_safe)
         + (Apr_over_A * Bpr_over_B)/(2 * B_safe))
    
    # Scalar field second derivative (Paper Eq 18)
    d2phi_dr2 = (B_safe * (params.mu**2 * phi + params.xi * R * phi) 
                 - (2/r_safe + 0.5*Apr_over_A - 0.5*Bpr_over_B) * p)
    
    return np.vstack([dA_dr, dB_dr, p, d2phi_dr2])

def boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
    """
    Boundary conditions:
    1. phi'(0) = 0 (regularity)
    2. B(0) = 1 (regularity)
    3. A(∞) = 1 (asymptotic flatness)
    4. phi(∞) = 0 (localization)
    """
    A0, B0, phi0, p0 = ya
    A_inf, B_inf, phi_inf, p_inf = yb
    return np.array([p0, B0 - 1.0, A_inf - 1.0, phi_inf])

# ==================== SOLVER CLASS ====================
class SolitonSolver:
    """Main solver for gravitational solitons."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.solutions: List[SolitonSolution] = []
        
        # Solver parameters
        self.solver_params = {
            'tol': 1e-8,
            'max_nodes': 100000,
            'verbose': 0
        }
    
    def solve_single(self, params: SolitonParameters) -> SolitonSolution:
        """Solve BVP for single parameter set with proper residual computation."""
        logger.info(f"Solving: ξ={params.xi:.3f}, μ={params.mu:.3f}, φ0={params.sigma0:.3e}")
        
        # Generate initial guess
        r = np.linspace(1e-8, params.r_max, params.n_grid)
        sigma = params.sigma0
        r0 = 1.0 / params.mu
        
        A_guess = np.ones_like(r)
        B_guess = np.ones_like(r)
        phi_guess = sigma * np.exp(-(r/r0)**2)
        p_guess = -2 * sigma * r / r0**2 * np.exp(-(r/r0)**2)
        
        y0 = np.vstack([A_guess, B_guess, phi_guess, p_guess])
        
        # Solve BVP
        sol = solve_bvp(
            lambda r, y: field_equations(r, y, params),
            boundary_conditions,
            r, y0,
            tol=self.solver_params['tol'],
            max_nodes=self.solver_params['max_nodes'],
            verbose=self.solver_params['verbose']
        )
        
        # Interpolate to fine grid for analysis
        r_fine = np.linspace(1e-8, params.r_max, 2000)
        A_fine = sol.sol(r_fine)[0]
        B_fine = sol.sol(r_fine)[1]
        phi_fine = sol.sol(r_fine)[2]
        p_fine = sol.sol(r_fine)[3]
        
        # COMPUTE TRUE ODE RESIDUAL
        y_fine = np.vstack([A_fine, B_fine, phi_fine, p_fine])
        rhs_val = field_equations(r_fine, y_fine, params)
        dy_num = np.gradient(y_fine, r_fine, axis=1)
        residual_norm = float(np.max(np.abs(dy_num - rhs_val)))
        
        solution = SolitonSolution(
            params=params,
            r=r_fine,
            A=A_fine,
            B=B_fine,
            phi=phi_fine,
            phi_prime=p_fine,
            converged=sol.success,
            residual_norm=residual_norm,
            iterations=getattr(sol, 'niter', 0)
        )
        
        self.solutions.append(solution)
        
        if solution.converged:
            logger.info(f"  ✓ M_ADM={solution.adm_mass:.6e}, residual={residual_norm:.2e}")
        else:
            logger.warning(f"  ✗ Failed to converge")
        
        return solution
    
    def parameter_scan(self, xi_list: List[float], mu_list: List[float], 
                      sigma0_list: List[float], r_max_factor: float = 50.0) -> pd.DataFrame:
        """Systematic parameter scan."""
        results = []
        
        total_runs = len(xi_list) * len(mu_list) * len(sigma0_list)
        run_count = 0
        
        for xi in xi_list:
            for mu in mu_list:
                for sigma0 in sigma0_list:
                    run_count += 1
                    logger.info(f"Run {run_count}/{total_runs}: ξ={xi:.3f}, μ={mu:.3f}, φ0={sigma0:.3e}")
                    
                    # Adaptive domain and grid
                    r_max = r_max_factor / mu
                    n_grid = max(500, int(2000 / mu))
                    
                    params = SolitonParameters(
                        xi=xi, mu=mu, sigma0=sigma0,
                        r_max=r_max, n_grid=n_grid
                    )
                    
                    try:
                        solution = self.solve_single(params)
                        
                        if solution.converged:
                            max_phi = np.max(np.abs(solution.phi))
                            is_trivial = max_phi < 1e-6
                            
                            result = {
                                'xi': xi, 'mu': mu, 'sigma0': sigma0,
                                'adm_mass': solution.adm_mass,
                                'max_phi': max_phi,
                                'A_inf': solution.A[-1],
                                'B_inf': solution.B[-1],
                                'converged': True,
                                'trivial': is_trivial,
                                'residual': solution.residual_norm
                            }
                            results.append(result)
                            
                            # Save nontrivial solutions
                            if not is_trivial:
                                self._save_solution(solution, f"nontrivial_xi{xi:.3f}_mu{mu:.3f}")
                    
                    except Exception as e:
                        logger.error(f"Error: {str(e)}")
                        results.append({
                            'xi': xi, 'mu': mu, 'sigma0': sigma0,
                            'adm_mass': np.nan, 'max_phi': np.nan,
                            'A_inf': np.nan, 'B_inf': np.nan,
                            'converged': False, 'trivial': True,
                            'residual': np.nan
                        })
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "scan_results.csv", index=False)
        self._analyze_results(df)
        
        return df
    
    def _save_solution(self, solution: SolitonSolution, filename: str):
        """Save solution to HDF5."""
        filepath = self.output_dir / f"{filename}.h5"
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('r', data=solution.r)
            f.create_dataset('A', data=solution.A)
            f.create_dataset('B', data=solution.B)
            f.create_dataset('phi', data=solution.phi)
            f.create_dataset('phi_prime', data=solution.phi_prime)
            f.attrs['adm_mass'] = solution.adm_mass
            f.attrs['xi'] = solution.params.xi
            f.attrs['mu'] = solution.params.mu
            f.attrs['sigma0'] = solution.params.sigma0
    
    def _analyze_results(self, df: pd.DataFrame):
        """Analyze scan results."""
        converged = df[df['converged']]
        nontrivial = converged[~converged['trivial']]
        
        summary = {
            'total_runs': len(df),
            'converged': len(converged),
            'nontrivial': len(nontrivial),
            'convergence_rate': len(converged)/len(df)*100,
            'nontrivial_rate': len(nontrivial)/len(converged)*100 if len(converged) > 0 else 0
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary: {summary}")

# ==================== ANALYSIS TOOLS ====================
def verify_solution_quality(solution: SolitonSolution) -> Dict[str, bool]:
    """Verify solution meets publication-quality standards."""
    checks = {}
    
    # 1. Convergence flag from solver
    checks['solver_converged'] = solution.converged
    
    # 2. True ODE residual (Paper 1 standard: < 1e-6)
    checks['residual_small'] = solution.residual_norm < 1e-6
    
    # 3. Asymptotic flatness
    A_inf = solution.A[-1]
    B_inf = solution.B[-1]
    checks['asymptotically_flat'] = (abs(A_inf - 1.0) < 1e-4 and 
                                    abs(B_inf - 1.0) < 1e-4)
    
    # 4. Scalar field localization
    phi_inf = solution.phi[-1]
    checks['field_localized'] = abs(phi_inf) < 1e-6
    
    # 5. Regularity at origin
    B0 = solution.B[0]
    phi_prime0 = solution.phi_prime[0]
    checks['regular_at_origin'] = (abs(B0 - 1.0) < 1e-4 and 
                                   abs(phi_prime0) < 1e-6)
    
    # 6. Mass positivity
    checks['mass_nonnegative'] = solution.adm_mass >= -1e-10
    
    # 7. Effective coupling positivity
    F = F0 - solution.params.xi * solution.phi**2
    checks['coupling_positive'] = np.all(F > 0)
    
    return checks

def compute_residual_breakdown(solution: SolitonSolution) -> pd.DataFrame:
    """Compute residual breakdown for each equation."""
    r = solution.r
    y = np.vstack([solution.A, solution.B, solution.phi, solution.phi_prime])
    rhs = field_equations(r, y, solution.params)
    dy_num = np.gradient(y, r, axis=1)
    
    residuals = dy_num - rhs
    max_residuals = np.max(np.abs(residuals), axis=1)
    rms_residuals = np.sqrt(np.mean(residuals**2, axis=1))
    
    df = pd.DataFrame({
        'equation': ['A_eq', 'B_eq', 'phi_eq', 'phi_prime_eq'],
        'max_residual': max_residuals,
        'rms_residual': rms_residuals,
        'target': [1e-6, 1e-6, 1e-6, 1e-6]
    })
    
    return df

def plot_solution(solution: SolitonSolution, save_path: Path):
    """Create comprehensive plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Scalar field
    axes[0,0].plot(solution.r, solution.phi, 'b-', linewidth=2)
    axes[0,0].set_xlabel('r', fontsize=12)
    axes[0,0].set_ylabel('φ(r)', fontsize=12)
    axes[0,0].set_title('Scalar Field Profile', fontsize=14)
    axes[0,0].grid(True, alpha=0.3)
    
    # Metric components
    axes[0,1].plot(solution.r, solution.A, 'r-', label='A(r) = -gₜₜ', linewidth=2)
    axes[0,1].plot(solution.r, 1/solution.B, 'b--', label='1/B(r) = gʳʳ', linewidth=2)
    axes[0,1].set_xlabel('r', fontsize=12)
    axes[0,1].set_ylabel('Metric Components', fontsize=12)
    axes[0,1].set_title('Metric Functions', fontsize=14)
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    
    # Mass function
    M_r = solution.mass_function
    axes[0,2].plot(solution.r, M_r, 'g-', linewidth=2)
    axes[0,2].axhline(solution.adm_mass, color='r', linestyle='--', 
                     label=f'ADM = {solution.adm_mass:.6f}')
    axes[0,2].set_xlabel('r', fontsize=12)
    axes[0,2].set_ylabel('M(r)', fontsize=12)
    axes[0,2].set_title('Mass Function', fontsize=14)
    axes[0,2].legend(fontsize=10)
    axes[0,2].grid(True, alpha=0.3)
    
    # Ricci scalar
    R = (2*(1-1/solution.B)/solution.r**2 
         - 2*np.gradient(np.log(solution.A), solution.r)/(solution.B*solution.r)
         + np.gradient(np.log(solution.A), solution.r)*np.gradient(np.log(solution.B), solution.r)/(2*solution.B))
    axes[1,0].plot(solution.r, R, 'purple', linewidth=2)
    axes[1,0].set_xlabel('r', fontsize=12)
    axes[1,0].set_ylabel('R', fontsize=12)
    axes[1,0].set_title('Ricci Scalar', fontsize=14)
    axes[1,0].grid(True, alpha=0.3)
    
    # Energy density
    rho = -0.5*solution.phi_prime**2/solution.B - 0.5*solution.params.mu**2*solution.phi**2
    axes[1,1].plot(solution.r, rho, 'orange', linewidth=2)
    axes[1,1].set_xlabel('r', fontsize=12)
    axes[1,1].set_ylabel('ρ = -Tᵗₜ', fontsize=12)
    axes[1,1].set_title('Energy Density', fontsize=14)
    axes[1,1].grid(True, alpha=0.3)
    
    # Effective coupling
    F = F0 - solution.params.xi * solution.phi**2
    axes[1,2].plot(solution.r, F, 'brown', linewidth=2)
    axes[1,2].axhline(F0, color='k', linestyle='--', label='F₀ = 1/(8πG)')
    axes[1,2].set_xlabel('r', fontsize=12)
    axes[1,2].set_ylabel('F(φ)', fontsize=12)
    axes[1,2].set_title('Effective Gravitational Coupling', fontsize=14)
    axes[1,2].legend(fontsize=10)
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle(f"ξ={solution.params.xi:.3f}, μ={solution.params.mu:.3f}, "
                f"φ(0)={solution.params.sigma0:.3e}, M_ADM={solution.adm_mass:.6e}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution with rigorous analysis."""
    solver = SolitonSolver("results")
    
    # Create subdirectories
    (Path("results") / "plots").mkdir(exist_ok=True)
    (Path("results") / "data").mkdir(exist_ok=True)
    
    # Parameter ranges for systematic scan
    xi_list = np.linspace(-20, 20, 21)       # Coupling constant
    mu_list = np.linspace(0.1, 2.0, 10)      # Scalar mass
    sigma0_list = np.linspace(1e-4, 0.1, 20) # Central field value
    
    logger.info("="*70)
    logger.info("RIGOROUS PARAMETER SCAN")
    logger.info(f"ξ ∈ [{xi_list[0]}, {xi_list[-1]}], "
               f"μ ∈ [{mu_list[0]}, {mu_list[-1]}], "
               f"φ(0) ∈ [{sigma0_list[0]}, {sigma0_list[-1]}]")
    logger.info("="*70)
    
    # Perform systematic parameter scan
    results_df = solver.parameter_scan(xi_list, mu_list, sigma0_list)
    
    # Analyze results
    converged = results_df[results_df['converged']]
    
    if len(converged) > 0:
        logger.info(f"\nConverged solutions: {len(converged)}/{len(results_df)}")
        
        # Quality analysis
        high_quality = converged[converged['residual'] < 1e-6]
        logger.info(f"High quality (residual < 1e-6): {len(high_quality)}")
        
        nontrivial = converged[~converged['trivial']]
        logger.info(f"Nontrivial solutions: {len(nontrivial)}")
        
        if len(nontrivial) > 0:
            logger.info(f"\nFOUND {len(nontrivial)} NONTRIVIAL SOLUTIONS!")
            
            # Analyze best nontrivial solution
            best_idx = nontrivial['adm_mass'].abs().idxmax()
            best_params = nontrivial.loc[best_idx]
            
            # Recompute with higher resolution
            params = SolitonParameters(
                xi=best_params['xi'],
                mu=best_params['mu'],
                sigma0=best_params['sigma0'],
                r_max=50.0/best_params['mu'],
                n_grid=2000
            )
            
            solution = solver.solve_single(params)
            
            # Verify solution quality
            quality = verify_solution_quality(solution)
            logger.info("\nSolution Quality Checks:")
            for check, passed in quality.items():
                status = "✓" if passed else "✗"
                logger.info(f"  {status} {check}")
            
            # Compute residual breakdown
            res_breakdown = compute_residual_breakdown(solution)
            logger.info("\nResidual Breakdown:")
            for idx, row in res_breakdown.iterrows():
                logger.info(f"  {row['equation']}: max={row['max_residual']:.2e}, RMS={row['rms_residual']:.2e}")
            
            # Plot and save
            plot_solution(solution, Path("results") / "plots" / "best_nontrivial_solution.png")
            
            # Save detailed analysis
            analysis = {
                'parameters': {
                    'xi': float(best_params['xi']),
                    'mu': float(best_params['mu']),
                    'sigma0': float(best_params['sigma0']),
                    'adm_mass': float(best_params['adm_mass'])
                },
                'quality_checks': quality,
                'residual_breakdown': res_breakdown.to_dict('records'),
                'is_trivial': False
            }
            
            with open(Path("results") / "nontrivial_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)
            
        else:
            logger.info("\nNO NONTRIVIAL SOLUTIONS FOUND")
            
            # Create phase diagram
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(converged['xi'], converged['mu'], 
                           c=converged['adm_mass'].abs(),
                           s=50, alpha=0.6, cmap='viridis',
                           norm=plt.cm.colors.LogNorm(vmin=1e-10, vmax=1))
            plt.colorbar(sc, label='|M_ADM|')
            plt.xlabel('Non-minimal coupling ξ', fontsize=14)
            plt.ylabel('Scalar mass μ [M_pl]', fontsize=14)
            plt.title('Phase Diagram: All Solutions Trivial (M_ADM ≈ 0)', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.savefig(Path("results") / "plots" / "phase_diagram.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save conclusion
            conclusion = {
                'total_solutions': len(converged),
                'nontrivial_solutions': 0,
                'conclusion': 'No regular, asymptotically flat, nontrivial static solutions found.',
                'parameter_space': {
                    'xi_range': [float(xi_list[0]), float(xi_list[-1])],
                    'mu_range': [float(mu_list[0]), float(mu_list[-1])],
                    'sigma0_range': [float(sigma0_list[0]), float(sigma0_list[-1])]
                },
                'max_adm_mass': float(converged['adm_mass'].abs().max()),
                'min_adm_mass': float(converged['adm_mass'].abs().min())
            }
            
            with open(Path("results") / "conclusion.json", 'w') as f:
                json.dump(conclusion, f, indent=2)
            
            logger.info("\n" + "="*70)
            logger.info("CONCLUSION FOR PAPER:")
            logger.info("No regular, asymptotically flat, nontrivial static solutions")
            logger.info(f"found across {len(converged)} converged solutions.")
            logger.info(f"Parameter space: ξ ∈ [-20, 20], μ ∈ [0.1, 2.0], φ(0) ∈ [1e-4, 0.1]")
            logger.info("="*70)
    
    else:
        logger.warning("No converged solutions found. Check parameter ranges or initial guesses.")
    
    logger.info("\nAnalysis complete.")

# ==================== RUN SCRIPT ====================
if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    try:
        main()
        elapsed = time.time() - start_time
        logger.info(f"\nTotal execution time: {elapsed:.2f} seconds")
        
        # Generate summary README
        readme = f"""
        # Gravitational Soliton Research Results
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Execution time: {elapsed:.2f} seconds
        
        ## Files:
        - scan_results.csv: Complete parameter scan results
        - summary.json: Summary statistics
        - conclusion.json: Main research conclusion
        - plots/: Visualization of results
        - data/: Raw solution data
        
        ## Code implements:
        - Field equations from Paper 1
        - True ODE residual computation (max|dy/dr - f(r,y)|)
        - Systematic parameter scanning
        - Quality verification
        - ADM mass extraction
        """
        
        with open("results/README.md", "w") as f:
            f.write(readme)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise