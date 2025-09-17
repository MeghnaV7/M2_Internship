import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from astropy.table import Table
import sys
import os
from astropy import units as u
from os import path
import glob
import re
from scipy.special import gammaincinv
#import galstars
data_list = []

def get_the_catalog(filename,snapnum):
    hf = h5py.File(filename)[f'output_{snapnum:05d}']
    datadict = {k: hf.get(k)[()] for k in hf.keys()}
    cat = pd.DataFrame(datadict, index=None)
    return cat

def define_mask(N, r, rmin, rmax, type):
    stars_true = N>0
    if type == 'exp':
        mask = np.logical_and(r>1.5*rmin, r<rmax)
        mask = np.logical_and(stars_true, mask)
    elif type == 'sersic':
        mask = np.logical_and(r>rmin, r<0.5*rmax)
        mask = np.logical_and(stars_true, mask)
    elif type =='comb':
        mask = np.logical_and(r>rmin, r<rmax)
        mask = np.logical_and(stars_true, mask)
    
    return mask
def exp_func(r, Sigma0, rs, delta_r):
    rexp = rs + delta_r
    return Sigma0 * np.exp(-r/rexp)

def sersic_func(r, Sigma0, rsersic, n):
    bn = gammaincinv(2*n, 0.5)
    return Sigma0 * np.exp(-bn * (((r / rsersic) ** (1. / n)) - 1))

def comb_func(r, s0_sers, s0_exp, rs, delta_r, n):
    #re = rs + delta_r
    return sersic_func(r, s0_sers, rs, n) + exp_func(r, s0_exp, rs, delta_r)

def fit_exp(df, rmin, rmax, r50, rvir):
    ''''Fit an exponential profile to the surface density profile
    Fitting parameters: sigma0, rsersic, delta_r
    rmin, rmax: fitting range'''
    
    minr_fit = 35e-3
    r, sd, N = df['R'].values, df['SD'].values, df['stars_per_bin'].values
    mask = define_mask(N, r, minr_fit , rmax, 'exp')
    r_fit, sd_fit, N_fit = r[mask], sd[mask], N[mask]
    
    fit_params, fit_cov = curve_fit(exp_func, r_fit, sd_fit, sigma=sd_fit/np.sqrt(N_fit), p0=[sd_fit[0], r50, 0.],
                bounds=[(1e-5, np.minimum(rmin, r50), 0.), (1e60, rmax, 2*rvir)], maxfev=100000) #profile,r,x,y,sigma,)
    yfit = exp_func(r, *fit_params)
    chi2 = np.sum((sd_fit - yfit[mask])**2 / (sd_fit/np.sqrt(N_fit))**2)
    dof = len(sd_fit)-len(fit_params)

    return fit_params, yfit, chi2/dof

def fit_sersic(df, rmin, rmax, r50):
    '''Fit a Sersic profile to the surface density profile
    Fitting parameters: sigma0, rsersic, n'''
   
    minr_fit = 35e-3
    r, sd, N = df['R'].values, df['SD'].values, df['stars_per_bin'].values
    mask = define_mask(N, r, minr_fit, rmax, 'sersic')
    r_fit, sd_fit, N_fit = r[mask], sd[mask], N[mask]

    fit_params, fit_cov = curve_fit(sersic_func, r_fit, sd_fit, sigma=sd_fit/np.sqrt(N_fit), p0=[sd_fit[0], r50, 4.],\
                bounds=[(1e-5, np.minimum(rmin, r50), 0.5), (1e60, rmax, 10)], maxfev=100000) #profile,r,x,y,sigma,)
    yfit = sersic_func(r, *fit_params)
    chi2 = np.sum((sd_fit - yfit[mask])**2 / (sd_fit/np.sqrt(N_fit))**2)
    dof = len(sd_fit)-len(fit_params)
   
    return fit_params, yfit, chi2/dof

def fit_comb(df, rmin, rmax, r50, rvir):
    '''Fit a combined profile (exponential + sersic) to the surface density profile
    Fitting parameters: sigma0_sersic, sigma0_exp, rsersic, delta_r, n'''
    
    minr_fit = 35e-3
    r, sd, N = df['R'].values, df['SD'].values, df['stars_per_bin'].values
    mask = define_mask(N, r, minr_fit, rmax, 'comb')
    r_fit, sd_fit, N_fit = r[mask], sd[mask], N[mask]
    fit_params, fit_cov = curve_fit(comb_func, r_fit, sd_fit, sigma=sd_fit/np.sqrt(N_fit), \
            p0=[sd_fit[0], sd_fit[0], r50, 0., 4.],  bounds=[(1e-5, 1e-5, np.minimum(rmin, r50), 0., 0.5), (1e60, 1e60, rmax, 2*rvir, 10)], maxfev=100000)
    yfit = comb_func(r, *fit_params)
    chi2 = np.sum((sd_fit - yfit[mask])**2 / (sd_fit/np.sqrt(N_fit))**2)
    dof = len(sd_fit)-len(fit_params)

    return fit_params, yfit, chi2/dof

    
def main():
    r_params = pd.read_csv('newvar_data/R_params_massive.csv')
    main_catalog = '/data/mtrebitsch/Obelisk/Obelisk_catalogue.h5'
    snapnum = 75
    cat = get_the_catalog(main_catalog, snapnum)
    id = r_params['ID'].values
    
    for i in range(len(id)):
        print('fitting for id', id[i])
        df = pd.read_csv(f'newvar_data/surface_density_{id[i]}.txt', sep=' ', names=['R', 'SD', 'stars_per_bin'])
        row_cat, rparam = cat[cat['ID'] == int(id[i])], r_params[r_params['ID'] == id[i]]
        r, sd, N = df['R'].values, df['SD'].values, df['stars_per_bin'].values
        rvir = row_cat['rvir'].values[0]
        rmin_lim, rmax_lim = r[np.argmax(sd)], rparam['r95'].values[0]
        r50 = rparam['r50'].values[0]

        exp_params, exp_yfit, exp_chi2 = fit_exp(df, rmin_lim, rmax_lim, r50, rvir)
        s0_exp, rs_exp, delta_r_exp = exp_params
        sersic_params, sersic_yfit, sersic_chi2 = fit_sersic(df, rmin_lim, rmax_lim, r50)
        s0_sers, rs_sers, n_sers = sersic_params
        comb_params, comb_yfit, comb_chi2 = fit_comb(df, rmin_lim, rmax_lim, r50, rvir)

        sc0_sers, sc0_exp, rs, delta_r_comb, n_comb = comb_params
        sersic_part = sersic_func(r, sc0_sers, rs, n_comb)
        exp_part = exp_func(r, sc0_exp, rs, delta_r_comb)

        t = Table([r, sd, sersic_yfit, exp_yfit, comb_yfit, sersic_part, exp_part], names=('R', 'SD', 'Sersic_fit', 'Exp_fit', 'Combined_fit', 'Sersic_part', 'Exp_part'))
        np.savetxt(f'correction_0.5/fits_{id[i]}.txt', t, fmt=['%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e'], header=' '.join(t.colnames), comments='')

        data_list.append({'ID': id[i], 
                          'R50': r50, 
                          'Rmin_limit': rmin_lim, 'Rmax_limit': rmax_lim, 
                          'sigma0_sers' :s0_sers, 'rd_sers':rs_sers, 'n_sers':n_sers,
                          'sigma0_exp':s0_exp,'rd_exp':(rs_exp+delta_r_exp),
                          'sigma0_comb_exp':sc0_exp, 'sigma0_comb_sers':sc0_sers, 'rd_comb_sers':rs, 'rd_comb_exp':(rs+delta_r_comb), 'n_comb':n_comb,
                          'chi2_reduced_sersic': sersic_chi2, 
                          'chi2_reduced_exp': exp_chi2, 
                          'chi2_reduced_combined': comb_chi2})
        
        data = pd.DataFrame(data_list)
        data.to_csv('correction_0.5/Fp_massive.csv')

    return

if __name__ == "__main__":
    main()
