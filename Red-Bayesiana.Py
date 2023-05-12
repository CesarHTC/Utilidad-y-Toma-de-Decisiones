import numpy as np
import pandas as pd
import pymc3 as pm

# datos de entrenamiento
temperaturas = np.array([22.1, 23.3, 24.8, 25.9, 26.5, 27.2, 28.3, 29.1, 29.5, 29.9])
presiones = np.array([1001, 1005, 1007, 1008, 1009, 1010, 1009, 1007, 1005, 1004])
clima = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 indica que el clima está frío, 1 indica que está caliente

with pm.Model() as modelo:
    # distribuciones previas
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
    
    # media del modelo
    mu = alpha + beta1 * temperaturas + beta2 * presiones
    
    # distribución de probabilidad para la observación
    clima_obs = pm.Bernoulli('clima_obs', p=pm.math.sigmoid(mu), observed=clima)
    
    # inferencia
    traza = pm.sample(1000, tune=1000)
    
pm.traceplot(traza)


def predecir_clima(temperatura, presion, traza):
    with pm.Model() as modelo:
        # variables latentes
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        
        # media del modelo
        mu = alpha + beta1 * temperatura + beta2 * presion
        
        # distribución de probabilidad para la observación
        clima_pred = pm.Bernoulli('clima_pred', p=pm.math.sigmoid(mu))
        
        # inferencia
        traza_pred = pm.sample_posterior_predictive(traza, samples=1000, vars=[clima_pred])
        
    return np.round(traza_pred['clima_pred'].mean(axis=0)) #La función np.round() se utiliza para redondear 
    #la predicción a 0 o 1, que corresponden a los valores de clima frío o caliente, respectivament
