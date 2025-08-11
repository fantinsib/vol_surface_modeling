import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import tools as t



def svi_calibration(df, lam=0.0001):
    """
    Takes a df with Maturity, Spot, RiskFreeRate in columns, and lam, the regularisation parameter for the minimisation (L2 regularization)
    Returns a dict containing with maturities as key associated with a dict containing a, b, rho, m and sigma 
    """
    svi_params_by_maturity = {}
    maturities = sorted(df['Maturity'].unique())


    for i, T in enumerate(maturities):

        df_slice = df[df['Maturity'] == T]
        
        F = df_slice['Spot'].iloc[0] * np.exp(df_slice['RiskFreeRate'].iloc[0] * T)
        
        k = np.log(df_slice['Strike'] / F)
        
        total_var_market = (df_slice['Volatility'] ** 2) * T

        init_guess = [0.01, 0.1, 0.0, 0.0, 0.1]
        bounds = [(-1, 1), (1e-5, 10), (-0.999, 0.999), (-5, 5), (1e-5, 5)]
        
        
        result = minimize(
            t.svi_objective,
            x0=init_guess,
            args=(k.values, total_var_market.values, lam),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        a, b, rho, m, sigma = result.x
        svi_params_by_maturity[T] = {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma}
        
    return(svi_params_by_maturity)



def interpolation_svi(svi_params_by_maturity, ridge_degree = 3, ridge_alpha = 0.1):
    """
    Takes a dictionnary containing SVI parameters for discrete maturity values and returns a returns a fitted interpolation pipeline for each parameter
    """
    Ts = []
    params = []

    for T, param_dict in svi_params_by_maturity.items():
        Ts.append(T)
        params.append([
            param_dict['a'],
            param_dict['b'],
            param_dict['rho'],
            param_dict['m'],
            param_dict['sigma']
        ])

    Ts = np.array(Ts)
    params = np.array(params) 
    sorted_indices = np.argsort(Ts)
    Ts_sorted = Ts[sorted_indices]
    params_sorted = params[sorted_indices]

    #Interpolation of each SVI parameters through multiple T using a polynomial ridge regression
    
    models = []
    for i in range(params.shape[1]):
        y = params_sorted[:, i]
        model = make_pipeline(
            PolynomialFeatures(ridge_degree),
            Ridge(alpha=ridge_alpha)
        )
        model.fit(Ts_sorted.reshape(-1,1), y)
        models.append(model)
    return(models)

    
def svi_params(T, interpolation_models):
    """
    Returns the interpolated parameters for any given maturity for a set of fitted interpolation pipeline
    """
    T_arr = np.array(T).reshape(-1, 1)
    param_pred = np.array([model.predict(T_arr).flatten() for model in interpolation_models]).T
    
    if param_pred.shape[0] == 1:
        param_pred = param_pred[0]
    #restreint les paramètres dans un range réaliste
    if param_pred.ndim == 1:
        a, b, rho, m, sigma = param_pred
        b = max(b, 1e-4)
        sigma = max(sigma, 1e-4)
        rho = np.clip(rho, -0.999, 0.999)
        return np.array([a, b, rho, m, sigma])
    else:
        a = param_pred[:,0]
        b = np.maximum(param_pred[:,1], 1e-4)
        rho = np.clip(param_pred[:,2], -0.999, 0.999)
        m = param_pred[:,3]
        sigma = np.maximum(param_pred[:,4], 1e-4)
        return np.vstack([a,b,rho,m,sigma]).T

def volatility_surface(df,interpolation_models, spot = 100, r = 0.01):

    Ts = df['Maturity'].unique()

    T_grid = np.linspace(min(Ts), max(Ts), 50)  
    k_grid = np.linspace(-0.5, 0.5, 30)         

    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
    vol_surface = np.zeros_like(T_mesh)

    for i in range(T_mesh.shape[0]):
        for j in range(T_mesh.shape[1]):
            T_val = T_mesh[i, j]
            k_val = k_mesh[i, j]
            a, b, rho, m, sigma = svi_params(T_val, interpolation_models)  
            
            F = spot * np.exp(r * T_val)  
            K = F * np.exp(k_val)           
            
            vol_surface[i, j] = t.svi_implied_vol(K, spot, T_val, r, a, b, rho, m, sigma)

    return vol_surface



def disp_vol_surface(Ts, models, spot = 100, r = 0.01):

    """
    Displays a static 3D graph of the volatility surface given a set of maturities T 
    """
    T_grid = np.linspace(min(Ts), max(Ts), 50)   
    k_grid = np.linspace(-0.5, 0.5, 30)   

    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)

    vol_surface = np.zeros_like(T_mesh)

    for i in range(T_mesh.shape[0]):
        for j in range(T_mesh.shape[1]):
            T_val = T_mesh[i, j]
            k_val = k_mesh[i, j]
            a, b, rho, m, sigma = svi_params(T_val, models)  # parametres SVI interpolés
            
            F = spot * np.exp(r * T_val)           
            K = F * np.exp(k_val)                  
            
            vol_surface[i, j] = t.svi_implied_vol(K, spot, T_val, r, a, b, rho, m, sigma)

    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_mesh, k_mesh, vol_surface, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Maturity T (years)')
    ax.set_ylabel('Log-Moneyness k')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('SVI Implied Volatility Surface')
    ax.view_init(elev=20, azim=110)  
    plt.tight_layout()
    plt.show()


def interactive_vol_surface(Ts, models, data_points=None, spot = 100, r =0.01):

   

    
    T_grid = np.linspace(min(Ts), max(Ts), 50)   
    k_grid = np.linspace(-0.5, 0.5, 30)   

    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)

    vol_surface = np.zeros_like(T_mesh)


    for i in range(T_mesh.shape[0]):
        for j in range(T_mesh.shape[1]):
            T_val = T_mesh[i, j]
            k_val = k_mesh[i, j]
            a, b, rho, m, sigma = svi_params(T_val, models)  
            
            F = spot * np.exp(r * T_val)  
            K = F * np.exp(k_val)           
            
            vol_surface[i, j] = t.svi_implied_vol(K, spot, T_val, r, a, b, rho, m, sigma)

    fig = go.Figure(data=[go.Surface(
        x=T_mesh,
        y=k_mesh,
        z=vol_surface,
        colorscale='Viridis'
    )])

    fig.update_layout(
        title='SVI Implied Volatility Surface interactive',
        scene=dict(
            xaxis_title='Maturity T (years)',
            yaxis_title='Log-Moneyness k',
            zaxis_title='Implied Volatility'
        ),
        autosize=True,
        height=600,
    )
   


    if data_points is not None:
        F_options = data_points['Spot'] * np.exp(data_points['RiskFreeRate'] * data_points['Maturity'])
        k_options = np.log(data_points['Strike'] / F_options)
        scatter = go.Scatter3d(
        x=data_points['Maturity'],
        y=k_options,
        z=data_points['Volatility'],
        mode='markers',
        marker=dict(
            size=2,
            color='red'
            
        ),
        name='Market Data'
    )

        fig.add_trace(scatter)
        
    fig.show()



def display_dupire_surface(vol_s, Ts, models, k_max=1):


    T_grid = np.linspace(min(Ts), max(Ts), 50)    
    k_grid = np.linspace(-k_max, k_max, 30) 
    dT = T_grid[1] - T_grid[0]  
    dk = k_grid[1] - k_grid[0]  

    
    dvol_dT = np.gradient(vol_s, dT, axis=1) 
    dvol_dk = np.gradient(vol_s, dk, axis=0)
    d2vol_dk2 = np.gradient(dvol_dk, dk, axis=0) 
    T_mat = T_grid[np.newaxis, :]   
    k_mat = k_grid[:, np.newaxis]   

    num = (
        vol_s**2 +
        2 * T_mat * vol_s * dvol_dT +
        2 * 0.01 * T_mat * k_mat * vol_s * dvol_dk
    )

    den = (
        (1 + (k_mat / vol_s) * dvol_dk)**2 +
        T_mat * (d2vol_dk2 + (dvol_dk**2) / vol_s)
    )

    local_var = num / den

    local_var = np.maximum(local_var, 0)  
    local_vol = np.sqrt(local_var)


    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)

    fig = go.Figure(data=[go.Surface(
        x=T_mesh,
        y=k_mesh,
        z=local_vol,
        colorscale='Viridis',
        colorbar=dict(title='Local Volatility')
    )])

    fig.update_layout(
        title='Dupire Local Volatility Surface',
        scene=dict(
            xaxis_title='Maturity T (years)',
            yaxis_title='Log-Moneyness k',
            zaxis_title='Local Volatility σ_loc'
        ),
        autosize=True,
        width=900,
        height=700,
    )

    fig.show()








