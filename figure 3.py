import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
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
n = -1.8
alpha = 2.8
H0_TECT = 67.7 * 1000 / Mpc  # TECT H0 (1/s)

# ΛCDM parameters
H0_LCDM = 67.4 * 1000 / Mpc  # Planck H0 (1/s)
Omega_m = 0.315
Omega_Lambda = 0.685

def V_model(t):
    """Normalized volume function V(t)"""
    if np.isscalar(t):
        t = max(t, t_rec)
    else:
        t = np.maximum(t, t_rec)
    
    term1 = a_rec**3
    term2 = (1 - a_rec**3) * (t**alpha - t_rec**alpha) / (t0**alpha - t_rec**alpha)
    return term1 + term2

def a_model(t):
    """Scale factor a(t) = V(t)^(1/3)"""
    return V_model(t)**(1/3)

def H_model(t):
    """Hubble parameter H(t)"""
    C = (1 - a_rec**3) * alpha / (t0**alpha - t_rec**alpha)
    dVdt = C * t**(alpha - 1)
    return (1/3) * dVdt / V_model(t)

def z_of_t(t):
    """Convert time to redshift"""
    return 1/a_model(t) - 1

def t_of_z(z):
    """Convert redshift to time"""
    if z <= 0:
        return t0
    
    a_z = 1/(1 + z)
    V_z = a_z**3
    
    if V_z <= a_rec**3:
        return t_rec
    
    t_alpha = t_rec**alpha + (V_z - a_rec**3) * (t0**alpha - t_rec**alpha) / (1 - a_rec**3)
    return t_alpha**(1/alpha)

def H_of_z(z):
    """Hubble parameter as function of redshift"""
    t = t_of_z(z)
    return H_model(t)

def d_L_TECT(z):
    """TECT luminosity distance"""
    if z <= 0:
        return 0
    
    z_array = np.linspace(0, z, max(int(z*200), 200))
    H_array = np.array([H_of_z(zi) for zi in z_array])
    
    integrand = 1/H_array
    integral = simpson(integrand, z_array)
    
    return (1 + z) * c * integral

def H_LCDM(z):
    """ΛCDM Hubble parameter"""
    return H0_LCDM * np.sqrt(Omega_m * (1+z)**3 + Omega_Lambda)

def d_L_LCDM(z):
    """ΛCDM luminosity distance"""
    if z <= 0:
        return 0
    
    z_array = np.linspace(0, z, max(int(z*200), 200))
    H_array = np.array([H_LCDM(zi) for zi in z_array])
    
    integrand = 1/H_array
    integral = simpson(integrand, z_array)
    
    return (1 + z) * c * integral

def plot_figure3_improved():
    """Plot improved Figure 3 with all enhancements"""
    print("Generating Improved Figure 3...")
    
    # Observational data from Table 2
    # BAO data (H(z) measurements converted to luminosity distance)
    bao_data = {
        'z': [0.32, 0.57],
        'H_obs': [78.7, 99.3],  # km/s/Mpc
        'H_err': [4.7, 2.8]
    }
    
    # Type Ia Supernova data
    sn_data = {
        'z': [0.01, 0.1, 0.3, 0.7, 1.0, 1.5],
        'dL_obs': [44.3, 458.6, 1578.3, 4513.8, 6907.5, 11213.4],  # Mpc
        'dL_err': [2.2, 23.0, 79.0, 226.0, 345.0, 561.0]  # Assuming ~5% error
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Main plot
    ax_main = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    # Calculate model predictions
    z_theory = np.linspace(0.001, 1.6, 500)
    dL_TECT_theory = np.array([d_L_TECT(z) / Mpc for z in z_theory])
    dL_LCDM_theory = np.array([d_L_LCDM(z) / Mpc for z in z_theory])
    
    # Plot theoretical curves
    ax_main.plot(z_theory, dL_TECT_theory, 'orange', linewidth=2.5, 
                label='TECT Prediction', zorder=5)
    ax_main.plot(z_theory, dL_LCDM_theory, 'b--', linewidth=2, 
                label='ΛCDM Prediction', alpha=0.7, zorder=4)
    
    # Plot Type Ia Supernova data
    ax_main.errorbar(sn_data['z'], sn_data['dL_obs'], yerr=sn_data['dL_err'],
                    fmt='o', color='red', markersize=8, capsize=5, capthick=2,
                    label='Type Ia Supernovae', zorder=10)
    
    # For BAO, we don't plot on luminosity distance plot but mention in legend
    # ax_main.plot([], [], '^', color='green', markersize=10, label='BAO (in H(z) space)')
    
    # Configure main plot
    ax_main.set_xlabel('Redshift z', fontsize=14)
    ax_main.set_ylabel('Luminosity Distance d$_L$(z) [Mpc]', fontsize=14)
    ax_main.set_title('Luminosity Distance Comparison: TECT vs Observations', fontsize=16, pad=10)
    ax_main.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, 1.6)
    ax_main.set_ylim(0, 12000)
    
    # Calculate chi-squared for SN data
    chi2_total = 0
    for i, z in enumerate(sn_data['z']):
        dL_TECT_pred = d_L_TECT(z) / Mpc
        dL_obs = sn_data['dL_obs'][i]
        dL_err = sn_data['dL_err'][i]
        chi2 = ((dL_TECT_pred - dL_obs) / dL_err)**2
        chi2_total += chi2
    
    # Add chi-squared text
    chi2_text = f'χ²/dof = {chi2_total/len(sn_data["z"]):.2f}'
    ax_main.text(0.02, 0.95, chi2_text, transform=ax_main.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residual plot
    ax_residual = plt.subplot2grid((3, 1), (2, 0))
    
    # Calculate residuals for SN data
    residuals_TECT = []
    residuals_LCDM = []
    for i, z in enumerate(sn_data['z']):
        dL_obs = sn_data['dL_obs'][i]
        dL_err = sn_data['dL_err'][i]
        
        dL_TECT_pred = d_L_TECT(z) / Mpc
        dL_LCDM_pred = d_L_LCDM(z) / Mpc
        
        res_TECT = (dL_TECT_pred - dL_obs) / dL_obs * 100  # Percentage
        res_LCDM = (dL_LCDM_pred - dL_obs) / dL_obs * 100
        
        residuals_TECT.append(res_TECT)
        residuals_LCDM.append(res_LCDM)
    
    # Plot residuals
    ax_residual.errorbar(sn_data['z'], residuals_TECT, 
                        yerr=[err/obs*100 for err, obs in zip(sn_data['dL_err'], sn_data['dL_obs'])],
                        fmt='o', color='orange', markersize=8, capsize=5,
                        label='TECT', zorder=10)
    ax_residual.errorbar([z+0.01 for z in sn_data['z']], residuals_LCDM,  # Slight offset for clarity
                        yerr=[err/obs*100 for err, obs in zip(sn_data['dL_err'], sn_data['dL_obs'])],
                        fmt='s', color='blue', markersize=7, capsize=5,
                        label='ΛCDM', alpha=0.7, zorder=9)
    
    ax_residual.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_residual.fill_between([0, 1.6], -5, 5, alpha=0.2, color='gray', label='5% band')
    
    # Configure residual plot
    ax_residual.set_xlabel('Redshift z', fontsize=14)
    ax_residual.set_ylabel('Residual [%]', fontsize=14)
    ax_residual.set_xlim(0, 1.6)
    ax_residual.set_ylim(-15, 15)
    ax_residual.grid(True, alpha=0.3)
    ax_residual.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total χ² for TECT: {chi2_total:.2f}")
    print(f"Degrees of freedom: {len(sn_data['z'])}")
    print(f"Reduced χ²: {chi2_total/len(sn_data['z']):.2f}")
    
    print(f"\nMean residuals:")
    print(f"TECT: {np.mean(residuals_TECT):.2f}%")
    print(f"ΛCDM: {np.mean(residuals_LCDM):.2f}%")
    
    print(f"\nRMS residuals:")
    print(f"TECT: {np.sqrt(np.mean(np.array(residuals_TECT)**2)):.2f}%")
    print(f"ΛCDM: {np.sqrt(np.mean(np.array(residuals_LCDM)**2)):.2f}%")
    
    plt.savefig('Figure3_Luminosity_Distance_Improved.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the improved version
if __name__ == "__main__":
    plot_figure3_improved()
    print("\nFigure 3 (Improved) has been generated successfully!")
    print("The plot now includes error bars, model comparison, and residual analysis.")