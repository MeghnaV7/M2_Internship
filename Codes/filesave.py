import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from astropy.table import Table
#import yt
#from yt.units import Mpc, Msun, km, s
import sys
import os
from astropy import units as u
#import emcee
#import corner
import glob
import re
#import MCMC_Script
from os import path

script_dir = os.getcwd()
scripts_dir = os.path.join(script_dir, '..', '..','scripts')
print(scripts_dir)  # Just to check
sys.path.append(scripts_dir)

import galstars

file = '/data/mtrebitsch/Obelisk/Obelisk_catalogue.h5'
hf = h5py.File(file)['output_00075']
datadict = {k: hf.get(k)[()] for k in hf.keys()}
df = pd.DataFrame(datadict, index=None)

def projection(p,g):
    normal_vector = np.array([p['angular_momentum'][0], p['angular_momentum'][1], p['angular_momentum'][2]])
    normal_vector_norm = normal_vector/np.linalg.norm(normal_vector)
    rstar_array = np.array([p['position'][0]-g['position_x'], p['position'][1]-g['position_y'], p['position'][2]-g['position_z']])
    #r = np.sqrt(rstar_array[0]**2 + rstar_array[1]**2 + rstar_array[2]**2)
    rstar_along_j = np.dot(rstar_array.T, normal_vector_norm)
    rstar_along_j = rstar_along_j.reshape(rstar_along_j.size,1)
    normal_vector_norm = normal_vector_norm.reshape(3,1)
    rstar_perpendicular_j = rstar_array.T - rstar_along_j*normal_vector_norm.T

    return rstar_perpendicular_j, rstar_along_j

def find_rmax(rstar, mass_array):
    idx = np.argsort(rstar)
    # rstar_sorted = rstar_array[:,idx]c
    rstar = rstar.iloc[idx]
    mass_sorted = mass_array.iloc[idx]

    cumulative_mass_profile = np.cumsum(mass_sorted)
    cumulative_mass = cumulative_mass_profile.iloc[-1]

    mmax = 0.95*cumulative_mass
    m50 = 0.5*cumulative_mass
    r50_index = np.searchsorted(cumulative_mass_profile, m50)
    r50 = rstar.iloc[r50_index]
    rmax_index = np.searchsorted(cumulative_mass_profile, mmax)
    rmax = rstar.iloc[rmax_index]

    return rmax, r50

def surface_density(rstar_perpendicular, rmax, r50, mass):
    r = np.sqrt(rstar_perpendicular[:,0]**2 + rstar_perpendicular[:,1]**2+ rstar_perpendicular[:,2]**2)*1e3
    bins = np.linspace(10e-3, 1.5*rmax, 51)
    mass_per_bin, edges = np.histogram(r, bins=bins, weights=mass)
    bin_center = (edges[1:] + edges[:-1])/2
    bin_width = edges[1:] - edges[:-1]
    

    sd = mass_per_bin/(2*np.pi*bin_center*bin_width)

    N, _ = np.histogram(r, bins=bins)


    return sd, bin_center, N

mcut_off = df['mstar'] > 1e7
df = df.loc[mcut_off]
id = df['ID']
data_list = []
params_list = []
for i in range(len(id)):
    p,g = galstars.read_galfile(f"/data/mtrebitsch/Obelisk/Stars/GAL_00075/gal_stars_{id[i]:07d}", longint=True)
    rperp, _ = projection(p,g)
    rstar = np.sqrt((p['position'][0]-g['position_x'])**2 + (p['position'][1]-g['position_y'])**2 + (p['position'][2]-g['position_z'])**2)*1e3
    mass_arr = g['mass']*1e11
    rmin_lim = 35e-3
    rmax_lim, r50 = find_rmax(rstar, mass_arr)
    sd, r, N = surface_density(rperp, rmax_lim, r50, mass_arr)
    t = Table([r, sd, N], names=('R', 'SD', 'stars_per_bin'))
    np.savetxt(f'newvar_data/surface_density_{id[i]}.txt', t, fmt=['%.6f', '%.6f', '%.6f'])

    params = {'ID': id[i], 'r95': rmax_lim, 'r50': r50}
    params = params_list.append(params)
    params = pd.DataFrame(params_list)

params.to_csv('newvar_data/R_params_massive.csv')




