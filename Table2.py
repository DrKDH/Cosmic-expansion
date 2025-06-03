import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# 물리 상수
c = 299792458  # m/s
Mpc = 3.086e22  # m
H0 = 67.4  # km/s/Mpc (Planck 2018)

# TECT 모델 매개변수
t0 = 13.787e9 * 365.25 * 24 * 3600  # 현재 우주 나이 (초)
trec = 3.8e5 * 365.25 * 24 * 3600   # 재결합 시점 (초)
z_rec = 1089
a_rec = 1/(1+z_rec)
alpha = 2.8  # = 1 - n, where n = -1.8

# 실제 관측 데이터 (Table 2에서)
observational_data = {
    'z': [0.01, 0.1, 0.3, 0.7, 1.0, 1.5],
    'H_obs': [67.5, 74.1, 92.1, 146.6, 182, 252.7],  # km/s/Mpc
    'H_ref': [22, 22, 22, 21, 21, 20],  # 참고문헌
    'dL_obs': [43.7, 426.1, 1285.7, 4340.2, 6626.1, 10866.3],  # Mpc
    'dL_ref': [16, 16, 16, 16, 16, 16]  # 참고문헌
}

# TECT 모델 함수들
def V_model(t):
    """정규화된 부피 함수 V(t)"""
    return a_rec**3 + (1 - a_rec**3) * ((t**alpha - trec**alpha)/(t0**alpha - trec**alpha))

def t_of_z(z):
    """적색편이 z에서의 우주 시간"""
    a = 1/(1+z)
    Vt = a**3
    t_alpha = trec**alpha + (Vt - a_rec**3)*(t0**alpha - trec**alpha)/(1 - a_rec**3)
    return t_alpha**(1/alpha)

def H_model(z):
    """TECT 모델의 허블 매개변수 H(z)"""
    t = t_of_z(z)
    C = (1 - a_rec**3) * alpha / (t0**alpha - trec**alpha)
    dVdt = C * t**(alpha-1)
    H = dVdt/(3*V_model(t))
    return H * Mpc / 1000  # km/s/Mpc로 변환

def dL_model(z):
    """TECT 모델의 광도거리"""
    z_array = np.linspace(0, z, 500)
    Hz_array = np.array([H_model(zi) for zi in z_array])
    
    # H(z)를 SI 단위로 변환 (km/s/Mpc -> 1/s)
    Hz_SI = Hz_array * 1000 / Mpc
    
    # 적분 계산
    integrand = 1 / Hz_SI
    integral = simpson(integrand, z_array)
    
    # 광도거리 (Mpc 단위)
    dL = (1 + z) * c * integral / Mpc
    return dL

# 결과 계산
results = []
for i, z in enumerate(observational_data['z']):
    # 관측값
    H_obs = observational_data['H_obs'][i]
    dL_obs = observational_data['dL_obs'][i]
    
    # TECT 예측값
    H_TECT = H_model(z)
    dL_TECT = dL_model(z)
    
    # χ² 계산
    # H의 경우: σ_H = 5 km/s/Mpc
    chi2_H = ((H_TECT - H_obs) / 5)**2
    
    # dL의 경우: 10% 상대 오차
    sigma_dL = 0.1 * dL_obs
    chi2_dL = ((dL_TECT - dL_obs) / sigma_dL)**2
    
    results.append({
        'z': z,
        'H_obs (km/s/Mpc)': f"{H_obs:.1f}",
        'H_TECT (km/s/Mpc)': f"{H_TECT:.1f}",
        'χ²_H': f"{chi2_H:.2f}",
        'dL_obs (Mpc)': f"{dL_obs:.1f}",
        'dL_TECT (Mpc)': f"{dL_TECT:.1f}",
        'χ²_dL': f"{chi2_dL:.2f}"
    })

# DataFrame 생성 및 출력
df = pd.DataFrame(results)
print("\nTable 2. Comparison of BAO and Supernova Ia Data")
print("="*80)
print(df.to_string(index=False))
print("-"*80)

# 총 χ² 계산
chi2_H_values = [float(r['χ²_H']) for r in results]
chi2_dL_values = [float(r['χ²_dL']) for r in results]
total_chi2_H = sum(chi2_H_values)
total_chi2_dL = sum(chi2_dL_values)

print(f"Total χ²_H: {total_chi2_H:.2f}")
print(f"Total χ²_dL: {total_chi2_dL:.2f}")
print(f"χ²_H per d.o.f.: {total_chi2_H/len(results):.2f}")
print(f"χ²_dL per d.o.f.: {total_chi2_dL/len(results):.2f}")

# Figure 3 생성: Luminosity Distance 비교
plt.figure(figsize=(10, 7))
z_plot = np.linspace(0.01, 1.6, 200)
dL_plot = [dL_model(z) for z in z_plot]

# TECT 예측 곡선
plt.plot(z_plot, dL_plot, 'orange', linewidth=2, label='TECT Prediction')

# 관측 데이터 점들
plt.scatter(observational_data['z'], observational_data['dL_obs'], 
           color='red', s=80, marker='o', label='Observational Data', zorder=5)

# 오차 막대
dL_errors = [0.1 * dL for dL in observational_data['dL_obs']]
plt.errorbar(observational_data['z'], observational_data['dL_obs'], 
            yerr=dL_errors, fmt='none', color='red', alpha=0.5)

plt.xlabel('Redshift z', fontsize=14)
plt.ylabel('Luminosity Distance d$_L$(z) [Mpc]', fontsize=14)
plt.title('Figure 3. Luminosity Distance Comparison: TECT vs Observations', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1.6)
plt.ylim(0, 12000)
plt.tight_layout()
plt.savefig('Figure3_luminosity_distance_comparison.png', dpi=300)
plt.show()

# 추가 분석: 잔차 플롯
plt.figure(figsize=(10, 6))
residuals = [(dL_model(z) - dL_obs) / dL_obs * 100 
             for z, dL_obs in zip(observational_data['z'], observational_data['dL_obs'])]

plt.scatter(observational_data['z'], residuals, color='blue', s=80)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.fill_between([0, 1.6], [-10, -10], [10, 10], alpha=0.2, color='gray', 
                 label='±10% range')

plt.xlabel('Redshift z', fontsize=14)
plt.ylabel('Residual [(TECT - Obs)/Obs] × 100%', fontsize=14)
plt.title('Luminosity Distance Residuals', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1.6)
plt.ylim(-15, 15)
plt.tight_layout()
plt.savefig('Figure_residuals.png', dpi=300)
plt.show()