# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:31:53 2020

@author: Luan
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score
import pandas as pd
import seaborn as sns
import itertools
from scipy.stats import kendalltau,ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.optimize import linprog

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b,method = 'interior-point')
    return lp.success

def transforms(df,cols):
        CDF = [ECDF(df[col]) for col in cols]
        #ICDF = [interp1d(ECDF(df[col]).y,ECDF(df[col]).x,kind = 'zero') for col in cols]
        ICDF = [interp1d(ECDF(df[col]).y,ECDF(df[col]).x,kind = 'zero',bounds_error=False,fill_value=(ECDF(df[col]).y[1],ECDF(df[col]).y[-2])) for col in cols]
        return CDF,ICDF

def mapping(transform,df):
    N = len(df)
    df_transf = pd.DataFrame(columns = df.columns)
    df.replace([np.inf, -np.inf],np.nan,inplace = True)
    if df.isnull().values.any():
        df.dropna(inplace = True)
        df = pd.concat([df,df.sample(n = N-len(df),replace = True)])
        #df = df.sample(n = N,replace = True)
    for transf,var in zip(transform,df.columns):
            #try:
            df_transf[var] = transf(df[var])
            #except ValueError:
            #print(df[var])
    
    df_transf.replace([np.inf, -np.inf],np.nan,inplace = True)
    if df_transf.isnull().values.any():
        df_transf.dropna(inplace = True)
        df_transf = pd.concat([df_transf,df_transf.sample(n = N-len(df_transf),replace = True)])
    return df_transf

def sampler(N_samples,frac,df):
    samples = []
    for i in range(N_samples):
        #weights = []
        #for i in range(int(frac*(len(df)))):
            #weights.append(np.random.uniform(0,1-sum(weights)))#np.random.uniform(0,1-sum(weights))
        #np.array(weights)
            weights = np.random.dirichlet(np.ones(int(frac*len(df)))/len(df))
            #sample = df.sample(frac = int(frac*(len(df)))/len(df),replace = False).values.T.dot(weights)
            samples.append(df.sample(frac = int(frac*(len(df)))/len(df),replace = False).values.T.dot(weights))
    return pd.DataFrame(np.array(samples),columns = df.columns) 
    

df = pd.read_csv('xllaby_data-componentes.csv')
df.fillna(value = df.describe().loc['mean'],inplace = True)

CDF,ICDF = transforms(df,df.columns)

df_U = mapping(CDF,df)
df_O = mapping(ICDF,sampler(500,0.01,df_U))
#df_O.to_csv('xllaby_data-componentes_fake.csv',index = False)
