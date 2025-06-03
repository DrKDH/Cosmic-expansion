import numpy as np

# Physical constants
c = 299792458  # Speed of light (m/s)
G = 6.67430e-11  # Gravitational constant
hbar = 1.054571817e-34  # Reduced Planck constant
Mpc = 3.086e22  # Megaparsec in meters

# Universe parameters
t0 = 13.787e9 * 365.25 * 24 * 3600  # Present age (seconds)
t_rec = 3.8e5 * 365.25 * 24 * 3600  # Recombination time (seconds)
z_rec = 1089  # Recombination redshift
a_rec = 1/(1 + z_rec)  # Scale factor at recombination

# TECT model parameters
n = -1.8  # Time-energy coupling exponent
alpha = 2.8  # Volume evolution exponent (= 1 - n)

def V_model(t):
    """Normalized volume function V(t) from Equation (4)"""
    if np.isscalar(t):
        t = max(t, t_rec)
    else:
        t = np.maximum(t, t_rec)
    
    term1 = a_rec**3
    term2 = (1 - a_rec**3) * (t**alpha - t_rec**alpha) / (t0**alpha - t_rec**alpha)
    return term1 + term2

def dVdt_model(t):
    """Time derivative of volume: dV/dt"""
    if np.isscalar(t):
        t = max(t, t_rec)
    else:
        t = np.maximum(t, t_rec)
    
    C = (1 - a_rec**3) * alpha / (t0**alpha - t_rec**alpha)
    return C * t**(alpha - 1)

# Calculate energy components
t_array = np.logspace(np.log10(t_rec), np.log10(t0), 1000)
T_total = 1.0
mc2_fraction = 0.27
V_t0 = V_model(t0)
temporal_fraction_t0 = 0.10
spacetime_fraction_t0 = 1 - mc2_fraction - temporal_fraction_t0
kappa = spacetime_fraction_t0 * T_total / (t0 * V_t0)

# Initialize arrays
T_temporal = np.zeros_like(t_array)
T_matter = np.full_like(t_array, mc2_fraction)
T_spacetime = np.zeros_like(t_array)

# Calculate energy components for each time
for i, t in enumerate(t_array):
    V_t = V_model(t)
    T_spacetime[i] = kappa * t * V_t
    T_temporal[i] = T_total - mc2_fraction - T_spacetime[i]
    
    # Ensure non-negative temporal energy
    if T_temporal[i] < 0:
        T_temporal[i] = 0
        T_spacetime[i] = T_total - mc2_fraction

# Apply smoothing
for i in range(1, len(t_array)):
    if T_temporal[i] > T_temporal[i-1]:
        T_temporal[i] = T_temporal[i-1]
        T_spacetime[i] = T_total - mc2_fraction - T_temporal[i]

# Normalize
T_sum = T_temporal + T_matter + T_spacetime
T_temporal = T_temporal / T_sum
T_matter = T_matter / T_sum
T_spacetime = T_spacetime / T_sum

# Convert time to normalized units
t_normalized = t_array / t0

# Find crossover point
idx_matter_spacetime = np.where(T_spacetime > T_matter)[0]
if len(idx_matter_spacetime) > 0:
    idx_crossover = idx_matter_spacetime[0]
    t_crossover = t_normalized[idx_crossover]
    print(f"Matter-Spacetime Crossover occurs at:")
    print(f"  t/t₀ = {t_crossover:.3f}")
    print(f"  Index = {idx_crossover}")
    print(f"  Matter energy at crossover = {T_matter[idx_crossover]:.3f}")
    print(f"  Spacetime energy at crossover = {T_spacetime[idx_crossover]:.3f}")
    
    # Check nearby points for more precision
    print(f"\nNearby points:")
    for i in range(max(0, idx_crossover-2), min(len(t_array), idx_crossover+3)):
        print(f"  t/t₀ = {t_normalized[i]:.4f}: Matter = {T_matter[i]:.4f}, Spacetime = {T_spacetime[i]:.4f}")
else:
    print("No crossover found!")

# Also find where temporal energy equals spacetime energy
idx_temporal_spacetime = np.where(T_spacetime > T_temporal)[0]
if len(idx_temporal_spacetime) > 0:
    idx_temp_cross = idx_temporal_spacetime[0]
    print(f"\nTemporal-Spacetime Crossover occurs at:")
    print(f"  t/t₀ = {t_normalized[idx_temp_cross]:.3f}")