import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = '/data/mtrebitsch/Obelisk/Obelisk_catalogue.h5'
hf = h5py.File(file)['output_00075']
datadict = {k: hf.get(k)[()] for k in hf.keys()}
cat = pd.DataFrame(datadict, index=None)

def plot_profile(galaxy_id, rmax, chi_co, M):
    '''Function to plot the surface density profiles and the fits'''
    
    fits = pd.read_csv(f'correction_0.5/fits_{galaxy_id}.txt', sep=' ')
    #M = cat[cat['ID'] == galaxy_id]['mstar'].values[0]
    r = fits['R'].values
    sd = fits['SD'].values
    plt.figure()
    plt.scatter(r, sd, s=1, c='k')
    #plt.plot(r, ysers_fit, c='r', label='Sersic fit')
    plt.plot(r,fits['Sersic_part'], c='r', linestyle='--', label='Sersic part')
    plt.plot(r, fits['Exp_part'], c='r', linestyle=':', label='Exp part')
    #plt.plot(r, yexp_fit, c='b', label='Exponential fit')
    plt.plot(r, fits['Combined_fit'], c='r', label='Combined fit')
    plt.axvline(rmax, c='k', linestyle='--', label='$R_{95}$')
    plt.xscale('linear')
    plt.yscale('log')
    #plt.ylim(sd.min()/3, sd.max()*3)
    plt.xlabel('R [kpc]')
    plt.ylabel(r'$\Sigma$ [M$_\odot$ kpc$^{-2}$]')
    plt.title(f'Surface Density: {galaxy_id}')
    bbox = dict(boxstyle='round', fc='white', ec = 'k', alpha=0.5)
    plt.text(1.1, 0.9, f'Mass: {M:.4e} $M_\\odot$\n' f'$\chi^2$: {chi_co:.4e}', bbox=bbox, transform = plt.gca().transAxes, fontsize=10)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'correction_0.5/SDfits/Sd_profile_{galaxy_id}.png')
    plt.close()

    return

params = pd.read_csv('correction_0.5/Fp_massive.csv', sep=',')
id = params['ID'].values
for i in range(len(id)):
    rmax_lim = params['Rmax_limit'].values[i]
    var = params['chi2_reduced_combined'].values[i]
    plot_profile(id[i], rmax_lim, var, cat[cat['ID'] == id[i]]['mstar'].values[0])
