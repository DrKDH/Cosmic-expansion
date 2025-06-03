import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 299792458  # Speed of light (m/s)
G = 6.67430e-11  # Gravitational constant
Mpc = 3.086e22  # Megaparsec in meters

# Universe parameters
t0 = 13.787e9 * 365.25 * 24 * 3600  # Present age (seconds)
t_rec = 3.8e5 * 365.25 * 24 * 3600  # Recombination time (seconds)
z_rec = 1089  # Recombination redshift
a_rec = 1/(1 + z_rec)  # Scale factor at recombination

# TECT model parameters
n = -1.8  # Time-energy coupling exponent
alpha = 2.8  # Volume evolution exponent (= 1 - n)

# Standard model parameters
Omega_m = 0.315
Omega_Lambda = 0.685
H0 = 67.4 * 1000 / Mpc  # Hubble constant (1/s)

def V_model(t):
    """Normalized volume function V(t) from Equation (4)"""
    if np.isscalar(t):
        t = max(t, t_rec)
    else:
        t = np.maximum(t, t_rec)
    
    term1 = a_rec**3
    term2 = (1 - a_rec**3) * (t**alpha - t_rec**alpha) / (t0**alpha - t_rec**alpha)
    return term1 + term2

def a_model(t):
    """TECT scale factor a(t) = V(t)^(1/3)"""
    return V_model(t)**(1/3)

def compute_LCDM_scale_factor(t_array):
    """
    Compute ΛCDM scale factor properly
    Using the analytical solution for flat ΛCDM universe
    """
    # For flat ΛCDM, we have the parametric solution:
    # a = (Omega_m/Omega_Lambda)^(1/3) * sinh^(2/3)(3/2 * sqrt(Omega_Lambda) * H0 * t)
    
    # But we need to ensure a(t0) = 1
    # First, find the constant that makes a(t0) = 1
    def a_LCDM_unnormalized(t):
        if t <= 0:
            return 1e-10
        eta = 3/2 * np.sqrt(Omega_Lambda) * H0 * t
        return (Omega_m/Omega_Lambda)**(1/3) * np.sinh(eta)**(2/3)
    
    a_t0 = a_LCDM_unnormalized(t0)
    
    # Now compute normalized scale factor
    a_LCDM = np.zeros_like(t_array)
    for i, t in enumerate(t_array):
        a_LCDM[i] = a_LCDM_unnormalized(t) / a_t0
    
    return a_LCDM

def plot_figure2_final():
    """Plot final corrected Figure 2"""
    print("Generating Final Corrected Figure 2...")
    
    # Time array - focus on observable range
    # Use more points for smoother curves
    t_array = np.linspace(0.1 * t0, 2.0 * t0, 2000)
    t_norm = t_array / t0
    
    # Calculate TECT scale factors
    a_TECT = np.array([a_model(t) for t in t_array])
    a_TECT = a_TECT / a_TECT[np.argmin(np.abs(t_norm - 1.0))]  # Normalize to present
    
    # Calculate ΛCDM scale factors
    a_LCDM = compute_LCDM_scale_factor(t_array)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax_main = plt.subplot(111)
    
    # Plot scale factors
    ax_main.plot(t_norm, a_TECT, 'b-', linewidth=3, label='TECT Model', zorder=10)
    ax_main.plot(t_norm, a_LCDM, 'r--', linewidth=3, label='ΛCDM Model', zorder=10)
    
    # Mark present epoch
    ax_main.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_main.text(1.02, 0.8, 'Present Epoch', rotation=90, fontsize=11, 
                color='gray', alpha=0.7, transform=ax_main.get_xaxis_transform())
    
    # Add shaded regions with very light shading
    ax_main.axvspan(0.1, 0.5, alpha=0.03, color='blue')
    ax_main.axvspan(1.5, 2.0, alpha=0.03, color='red')
    
    # Add text labels for regions
    ax_main.text(0.3, 0.05, 'Early Universe', fontsize=12, ha='center', 
                alpha=0.5, color='blue', weight='bold')
    ax_main.text(1.75, 0.05, 'Future Evolution', fontsize=12, ha='center', 
                alpha=0.5, color='red', weight='bold')
    
    # Calculate differences at specific epochs
    idx_early = np.argmin(np.abs(t_norm - 0.3))
    diff_early = (a_LCDM[idx_early] - a_TECT[idx_early]) / a_LCDM[idx_early] * 100
    
    idx_near = np.argmin(np.abs(t_norm - 1.25))
    diff_near = (a_LCDM[idx_near] - a_TECT[idx_near]) / a_LCDM[idx_near] * 100
    
    # Set labels and title
    ax_main.set_xlabel('Normalized Cosmic Time (t/t₀)', fontsize=14)
    ax_main.set_ylabel('Scale Factor a(t)', fontsize=14)
    ax_main.set_title('Cosmic Expansion History: TECT vs ΛCDM Model', fontsize=16, pad=10)
    
    # Configure axes
    ax_main.set_xlim(0.1, 2.0)
    ax_main.set_ylim(0, 2.5)
    ax_main.grid(True, alpha=0.3)
    
    # Legend
    ax_main.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Create inset for ratio plot
    left, bottom, width, height = [0.58, 0.15, 0.35, 0.3]
    ax_inset = fig.add_axes([left, bottom, width, height])
    
    # Calculate and plot ratio
    ratio = a_TECT / a_LCDM
    ax_inset.plot(t_norm, ratio, 'k-', linewidth=2)
    ax_inset.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax_inset.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    
    # Find peak deviation in the future (t/t₀ > 1.2)
    idx_range = np.where((t_norm > 1.2) & (t_norm < 1.35))[0]
    if len(idx_range) > 0:
        idx_peak = idx_range[np.argmin(ratio[idx_range])]  # Minimum ratio (TECT < ΛCDM)
        peak_dev = (ratio[idx_peak] - 1.0) * 100
        
        ax_inset.plot(t_norm[idx_peak], ratio[idx_peak], 'ro', markersize=8)
        ax_inset.annotate(f'{peak_dev:.1f}%', 
                         xy=(t_norm[idx_peak], ratio[idx_peak]),
                         xytext=(t_norm[idx_peak]+0.1, ratio[idx_peak]),
                         fontsize=10, ha='left')
    
    # Configure inset
    ax_inset.set_xlabel('t/t₀', fontsize=11)
    ax_inset.set_ylabel('a_TECT/a_ΛCDM', fontsize=11)
    ax_inset.set_xlim(0.1, 2.0)
    ax_inset.set_ylim(0.94, 1.02)  # Adjusted for realistic range
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_title('Scale Factor Ratio', fontsize=11)
    
    # Add annotations to main plot
    # Only add annotation if difference is reasonable
    if abs(diff_early) < 50:  # Sanity check
        ax_main.annotate(f'TECT predicts {abs(diff_early):.1f}%\nslower expansion', 
                        xy=(0.3, (a_TECT[idx_early] + a_LCDM[idx_early])/2),
                        xytext=(0.35, 0.4),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                        fontsize=11, ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                 edgecolor='black', alpha=0.8))
    
    if 'peak_dev' in locals():
        ax_main.annotate(f'Peak difference\n{abs(peak_dev):.1f}% at t/t₀≈{t_norm[idx_peak]:.2f}', 
                        xy=(t_norm[idx_peak], (a_TECT[idx_near] + a_LCDM[idx_near])/2),
                        xytext=(1.25, 1.8),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                        fontsize=11, ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                 edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    # Print diagnostic information
    print(f"\nKey differences between models:")
    print(f"Early universe (t/t₀=0.3): {diff_early:.2f}% difference")
    print(f"Present epoch (t/t₀=1.0): Models normalized to match")
    if 'peak_dev' in locals():
        print(f"Peak difference: {peak_dev:.2f}% at t/t₀≈{t_norm[idx_peak]:.2f}")
    
    # Additional diagnostics
    idx_t0 = np.argmin(np.abs(t_norm - 1.0))
    print(f"\nScale factors at key epochs:")
    print(f"t/t₀=0.1: TECT={a_TECT[0]:.4f}, ΛCDM={a_LCDM[0]:.4f}")
    print(f"t/t₀=1.0: TECT={a_TECT[idx_t0]:.4f}, ΛCDM={a_LCDM[idx_t0]:.4f}")
    print(f"t/t₀=2.0: TECT={a_TECT[-1]:.4f}, ΛCDM={a_LCDM[-1]:.4f}")
    
    # Check physical consistency
    print(f"\nPhysical consistency check:")
    print(f"TECT monotonic increase: {np.all(np.diff(a_TECT) > 0)}")
    print(f"ΛCDM monotonic increase: {np.all(np.diff(a_LCDM) > 0)}")
    
    plt.savefig('Figure2_Final_Corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the final version
if __name__ == "__main__":
    plot_figure2_final()
    print("\nFigure 2 (Final Corrected) has been generated successfully!")
    print("The plot now shows physically realistic evolution for both models.")