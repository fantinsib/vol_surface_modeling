import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import brentq
import matplotlib.pyplot as plt



class CallOption:

    def __init__(self, S, K, T, r):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        

    def d1(self, sigma):
        return (np.log(self.S / self.K) + (self.r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))

    def d2(self, sigma):
        return self.d1(sigma) - sigma * np.sqrt(self.T)

    def price(self, sigma):
        """
        Calcule le prix du call avec Black-Scholes à partir d'une volatilité donnée
        """
        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    

    def implied_volatility(self, market_price, sigma_bounds=(1e-6, 5)):
        """
        Calcule la volatilité implicite à partir d'un prix de marché du call.
        """
        def bs_price(sigma):
            d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

        try:
            implied_vol = brentq(lambda sigma: bs_price(sigma) - market_price, *sigma_bounds)
            return implied_vol
        except ValueError:
            return np.nan





def svi_total_variance(k, a, b, rho, m, sigma):
    """
    Calcule la variance implicite totale w(k) selon le modèle SVI.

    Paramètres :
    - k : log-moneyness (float ou array)
    - a, b, rho, m, sigma : paramètres SVI

    Retour :
    - w : variance implicite totale (float ou array)
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_objective(params, k_obs, w_obs, lam=0):
    a, b, rho, m, sigma = params
    w_model = svi_total_variance(k_obs, a, b, rho, m, sigma)
    mse = np.mean((w_model - w_obs)**2)
    reg = lam * (a**2 + b**2 + rho**2 + m**2 + sigma**2)
    return mse + reg

def svi_implied_vol(K, S, T, r,a, b, rho, m, sigma):
    """
    Calcule la volatilité implicite selon le modèle SVI pour un log-moneyness k et une maturité T.
    
    Arguments :
        k     : float ou np.array, log-moneyness
        T     : float, maturité en années
        a,b,rho,m,sigma : float, paramètres SVI
        
    Retour :
        vol : float ou np.array, volatilité implicite

    """

    F = S*np.exp(r*T)
    k = np.log(K/F)
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))  
    w = np.maximum(w, 1e-8)
    vol = np.sqrt(w / T)  
    
    return vol

def svi_implied_vol_with_k(k,T, a, b, rho, m, sigma):
    """
    Calcule la volatilité implicite selon le modèle SVI pour un log-moneyness k et une maturité T.
    
    Arguments :
        k     : float ou np.array, log-moneyness
        T     : float, maturité en années
        a,b,rho,m,sigma : float, paramètres SVI
        
    Retour :
        vol : float ou np.array, volatilité implicite

    """
    
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))  

    
     
    w = np.maximum(w, 1e-8)
    vol = np.sqrt(w / T)  
    return vol




def synthetic_options_dataset(n_maturities, strikes_per_maturity, spot, r, base_vol, noise=0, vol_effect=0.5, horizon=10):
    

    maturities = np.linspace(0.5, horizon, n_maturities)
    data = []

    for T in maturities:
        F = spot * np.exp(r * T)
        strikes = (np.linspace(0.8 * spot, 1.2 * F, strikes_per_maturity)).round()
        for K in strikes:
            log_moneyness = np.log(K / F)
    
            vol = base_vol + vol_effect * (log_moneyness**2) + np.random.normal(0, noise)
            data.append({
                "OptionType": "call",
                "Strike": K,
                "Maturity": T,
                "Volatility": vol,
                "Spot": spot,
                "RiskFreeRate": r
            })

    df_synthetic = pd.DataFrame(data)
    return(df_synthetic)