"""
Confronto diretto tra implementazione in straddle_gpt.ipynb e util.py
"""
import numpy as np
from datetime import date, timedelta

print("=" * 80)
print("CONFRONTO IMPLEMENTAZIONI")
print("=" * 80)

# Parametri identici a straddle_gpt.ipynb
position = -1.0  # short straddle
dt = 1.0 / 252
r = 0.0
q = 0.0
sigma_bar = 0.09
vol_of_vol = 0.50
S0 = 1.20

# Simula 2 giorni semplici
np.random.seed(42)
S = [S0, 1.21, 1.19]
sigma_imp = [0.20, 0.20, 0.20]  # costante!
sigma0_trade = sigma_imp[0]

print("\nDATI TEST:")
print(f"  Position: {position} (short)")
print(f"  S: {S}")
print(f"  sigma_imp: {sigma_imp} (costante)")
print(f"  sigma0_trade: {sigma0_trade}")

# Implementazione STRADDLE_GPT.IPYNB (toy model)
print("\n" + "=" * 80)
print("IMPLEMENTAZIONE STRADDLE_GPT.IPYNB")
print("=" * 80)

# Funzioni BS semplificate
def norm_pdf(x): return (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*x*x)
def bs_d1(S, K, r, q, sigma, tau):
    if sigma * np.sqrt(tau) < 1e-10: return np.sign(np.log(S/K))*np.inf
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
def bs_gamma(S, K, r, q, sigma, tau):
    if tau<=0 or sigma<=0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,tau); return np.exp(-q*tau)*norm_pdf(d1)/(S*sigma*np.sqrt(tau))
def bs_vega(S, K, r, q, sigma, tau):
    if tau<=0 or sigma<=0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,tau); return S*np.exp(-q*tau)*norm_pdf(d1)*np.sqrt(tau)

# Parametri opzione
K = S[0]
tau = 21 / 252  # 1 mese

gamma_toy = 0.0
vega_toy = 0.0

for t in range(1, 3):
    # Calcola greeks
    gamma_call = bs_gamma(S[t-1], K, r, q, sigma_imp[t-1], tau)
    vega_call = bs_vega(S[t-1], K, r, q, sigma_imp[t-1], tau)
    gamma_straddle = 2.0 * gamma_call
    vega_straddle = 2.0 * vega_call
    
    # Gamma*
    Gamma_star = np.exp(-r * tau) * (S[t-1]**2) * gamma_straddle
    
    # Return
    r_t = (S[t] - S[t-1]) / S[t-1]
    
    # Gamma term
    gamma_inc = position * (0.5 * Gamma_star * ((r_t**2 / dt) - sigma0_trade**2) * dt)
    gamma_toy += gamma_inc
    
    # Vega term (dovrebbe essere 0 con sigma costante)
    # [codice originale usa F_t, ma con sigma costante F_t = F_prev = 0]
    
    print(f"\nDay {t}:")
    print(f"  S: {S[t-1]:.4f} -> {S[t]:.4f}, return: {r_t:.4%}")
    print(f"  Gamma_straddle: {gamma_straddle:.6f}")
    print(f"  Gamma_star: {Gamma_star:.6f}")
    print(f"  r_t^2/dt: {r_t**2/dt:.6f}")
    print(f"  sigma0^2: {sigma0_trade**2:.6f}")
    print(f"  gamma_inc: {gamma_inc:.6f}")

print(f"\nTOTALE Gamma Term (toy model): {gamma_toy:.6f}")

# Ora confronta con cosa fa util.py
print("\n" + "=" * 80)
print("COSA DOVREBBE FARE UTIL.PY")
print("=" * 80)

print("""
Il problema potrebbe essere:

1. SEGNO DEL POSITION:
   - toy model usa position = -1 per short
   - util.py usa side = -1 
   - Ma poi moltiplica di nuovo per side?

2. ACTUAL P&L vs FIRST ORDER:
   - Actual P&L = mark-to-market dell'opzione + hedge P&L
   - First Order = solo decomposizione greche
   - QUESTI NON DEVONO ESSERE UGUALI!
   
3. INTERPRETAZIONE CORRETTA:
   - First Order spiega PARTE del P&L
   - Non tutto il P&L!
   - Manca theta, manca P&L da roll, etc.

POSSIBILE SOLUZIONE:
   La formula decompone il P&L della strategia delta-hedged, ma:
   - Potrebbe assumere hedging continuo (noi facciamo discreto)
   - Potrebbe assumere no transaction costs
   - Il "P&L" potrebbe riferirsi a qualcosa di specifico

VERIFICA NECESSARIA:
   Controllare nel paper/articolo originale cosa significa esattamente
   quella formula. Non è il P&L totale dello straddle!
""")

print("=" * 80)

