#!/usr/bin/env python

# Code to create a plot to show the
# depths of the used catalogues in
# both magnitude and flux.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import colorcet as cc
import cmasher as cmr
# from adjustText import adjust_text
import global_variables as gv
import global_functions as gf

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

def VegaToAB(mag, delta) -> float:
    return mag + delta

def ABToVega(mag, delta) -> float:
    return mag - delta

def magToFlux(mag_name, mag_unit) -> u.quantity.Quantity:
    if mag_unit.unit == u.mag(u.AB):
        return mag_unit.to(u.uJy)
    elif mag_unit.unit == u.mag:
        mag_unit = VegaToAB(mag_unit.value, vega_shift[mag_name]) * u.mag(u.AB)
        return mag_unit.to(u.uJy)
    else:
        return mag_unit.to(u.uJy)

def FluxLimToSigma(flux_lim, old_sigma, new_sigma) -> u.quantity.Quantity:
    flux_sigma = flux_lim / old_sigma * new_sigma
    return flux_sigma

c = 299_792_458 * u.m / u.s

filter_names = ['g', 'r', 'i', 'z', 'y', 'J', 'H', 'Ks',\
                'W1-CW', 'W2-CW', 'W3-AW', 'W4-AW', 'VLAS82',\
                'LoTSS']  # Without 'W1-AW', 'W2-AW', 'FUVmag', 'NUVmag', 'VLASS', 'TGSS', 'LoLSS'

# Limits as quoted from original reference
filt_initial_sigma = {'FUV': 5, 'NUV': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5,\
                    'y': 5, 'J': 3, 'H': 3, 'Ks': 3, 'W1-CW': 5, 'W2-CW': 5, 'W1-AW': 5,\
                    'W2-AW': 5, 'W3-AW': 5, 'W4-AW': 5, 'LoLSS': 1, 'LoTSS': 1, 'TGSS': 1, 'VLAS82': 1, 'VLASS': 5}

# Limits in their original unit. From reference
filt_initial_limit = {'FUV': 20 * u.mag(u.AB), 'NUV': 21 * u.mag(u.AB),\
                     'g': 23.3 * u.mag(u.AB), 'r': 23.2 * u.mag(u.AB), 'i': 23.1 * u.mag(u.AB),\
                     'z': 22.3 * u.mag(u.AB), 'y': 21.4 * u.mag(u.AB), 'J': 17.1 * u.mag,\
                     'H': 16.4 * u.mag, 'Ks': 15.3 * u.mag, 'W1-CW': 17.43 * u.mag,\
                     'W2-CW': 16.47 * u.mag, 'W1-AW': 16.9 * u.mag, 'W2-AW': 16.0 * u.mag,\
                     'W3-AW': 11.5 * u.mag, 'W4-AW': 8.0 * u.mag, 'LoLSS': 5 * u.mJy,\
                     'LoTSS': 71 * u.uJy, 'TGSS': 24.5 * u.mJy, 'VLAS82': 52 * u.uJy, 'VLASS': 3 * u.mJy}

# Delta between Vega and AB magnitudes
vega_shift     = {'W1-CW': 2.699, 'W2-CW': 3.339, 'W1-AW': 2.699, 'W2-AW': 3.339, 'W3-AW': 5.174,\
                      'W4-AW': 6.620, 'J': 0.910, 'H': 1.390, 'Ks': 1.850}

# Band effective wavelength/frequency
filt_central_pos = {'FUV': 1549.02 * u.AA, 'NUV': 2304.74 * u.AA, 'g': 4810.88 * u.AA,\
                     'r': 6156.36 * u.AA, 'i': 7503.68 * u.AA, 'z': 8668.56 * u.AA,\
                     'y': 9613.45 * u.AA, 'J': 12350 * u.AA, 'H': 16620 * u.AA,\
                     'Ks': 21590 * u.AA, 'W1-CW': 33526 * u.AA, 'W2-CW': 46028 * u.AA,\
                     'W1-AW': 33526 * u.AA, 'W2-AW': 46028 * u.AA, 'W3-AW': 115608 * u.AA,\
                     'W4-AW': 220883 * u.AA, 'LoLSS': 54 * u.MHz, 'LoTSS': 144 * u.MHz,\
                     'TGSS': 150 * u.MHz, 'VLAS82': 1.4 * u.GHz, 'VLASS': 3 * u.GHz}

# Band width in original units
filt_band_width = {'FUV': 265.57 * u.AA, 'NUV': 768.31 * u.AA, 'g': 1053.08 * u.AA,\
                     'r': 1252.41 * u.AA, 'i': 1206.63 * u.AA, 'z': 997.71 * u.AA,\
                     'y': 638.99 * u.AA, 'J': 1624.32 * u.AA, 'H': 2509.40 * u.AA,\
                     'Ks': 2618.87 * u.AA, 'W1-CW': 6626.42 * u.AA, 'W2-CW': 10422.66 * u.AA,\
                     'W1-AW': 6626.42 * u.AA, 'W2-AW': 10422.66 * u.AA, 'W3-AW': 55055.71 * u.AA,\
                     'W4-AW': 41016.83 * u.AA, 'LoLSS': 24 * u.MHz, 'LoTSS': 48 * u.MHz,\
                     'TGSS': 10 * u.MHz, 'VLAS82': 50 * u.GHz, 'VLASS': 2 * u.GHz}


# Convert all quantities to uJy
filt_initial_lim_flux = {}
for filt_name in filter_names:
    filt_initial_lim_flux[filt_name] = magToFlux(filt_name, filt_initial_limit[filt_name])

# Transform all limits to 5-sigma values
filt_5sigma_lim_flux = {}
for filt_name in filter_names:
    filt_5sigma_lim_flux[filt_name] = FluxLimToSigma(filt_initial_lim_flux[filt_name], filt_initial_sigma[filt_name], 5)

# Transform all 5-sigma limit fluxes to ABmag
filt_5sigma_lim_AB = {}
for filt_name in filter_names:
    filt_5sigma_lim_AB[filt_name] = filt_5sigma_lim_flux[filt_name].to(u.mag(u.AB))

# Transform all band centers and widths to wavelength
filt_central_pos_wave = {}
filt_band_width_wave  = {}
for filt_name in filter_names:
    if filt_central_pos[filt_name].unit == u.Angstrom:
        filt_central_pos_wave[filt_name] = filt_central_pos[filt_name].to(u.um)
        filt_band_width_wave[filt_name] = filt_band_width[filt_name].to(u.um)
    elif (filt_central_pos[filt_name].unit == u.MHz) or (filt_central_pos[filt_name].unit == u.GHz):
        filt_central_pos_wave[filt_name] = (c / filt_central_pos[filt_name].to(u.Hz)).to(u.um)
        low_limit_freq = filt_central_pos[filt_name].to(u.Hz) - filt_band_width[filt_name].to(u.Hz) / 2
        up_limit_freq  = filt_central_pos[filt_name].to(u.Hz) + filt_band_width[filt_name].to(u.Hz) / 2
        up_limit_wave  = (c / low_limit_freq).to(u.AA)
        low_limit_wave = (c / up_limit_freq).to(u.AA)
        band_width_wave = up_limit_wave - low_limit_wave
        # filt_band_width_wave[filt_name] = (c / filt_band_width[filt_name].to(u.Hz)).to(u.AA)
        filt_band_width_wave[filt_name] = band_width_wave.to(u.um)

# Dictionaries to numpy arrays
# print(np.array(list(filt_central_pos_wave.values())))
central_pos_um       = np.array([filt_central_pos_wave[key_].value for key_ in list(filt_central_pos_wave.keys())])
central_pos_width_um = np.array([filt_band_width_wave[key_].value  for key_ in list(filt_band_width_wave.keys())])
depth_5sigma_AB      = np.array([filt_5sigma_lim_AB[key_].value    for key_ in list(filt_5sigma_lim_AB.keys())])
depth_5sigma_flux    = np.array([filt_5sigma_lim_flux[key_].value  for key_ in list(filt_5sigma_lim_flux.keys())])

AGN_sed  = 'observed'  # 'rest-frame' or 'observed'
AGN_name = 'mrk231'    # 'mrk231', '3c273'
if AGN_sed == 'observed':
    # Load SED from AGN (observed)
    # Mrk 231 
    # The Spectral Energy Distributions of Active Galactic Nuclei
    # Brown et al. (2019)
    # http://dx.doi.org/10.17909/t9-3dbt-8734 
    file_name = f'hlsp_agnsedatlas_multi_multi_{AGN_name}_multi_v1_spec-obs.txt'
    data_AGN  = np.loadtxt(gv.cat_path + file_name, usecols=(0, 1, 2, 3))
    AGN_wave  = data_AGN[:, 0] * u.um  # Observed wavelength (um)
    AGN_flux  = data_AGN[:, 1] * u.Jy  # f_nu (Jy)

elif AGN_sed == 'rest-frame':
    # Load SED from AGN (rest-frame)
    # Mrk 231 
    # The Spectral Energy Distributions of Active Galactic Nuclei
    # Brown et al. (2019)
    # http://dx.doi.org/10.17909/t9-3dbt-8734
    file_name  = f'hlsp_agnsedatlas_multi_multi_{AGN_name}_multi_v1_spec-rest.txt'
    data_AGN   = np.loadtxt(gv.cat_path + file_name, usecols=(0, 1, 2, 3))
    AGN_wave   = data_AGN[:, 0] * u.um # Column 1: Restframe wavelength (um)
    AGN_flux_l = data_AGN[:, 2] * u.erg / u.s / (u.cm)**2 / u.AA # Column 3: f_lam (erg/s/cm2/A)
    # Convert f_lam (erg/s/cm2/A) to f_nu (Jy)
    AGN_flux   = (AGN_flux_l * AGN_wave**2 / c).to(u.Jy)

# Plot bands and depths
save_plot_flag = True
show_plot_flag = False

# original redshift from source
orig_z = 0.0422  # Mrk231
# orig_z = 0.1583  # 3C273

fig             = plt.figure(figsize=(8,3.5))
ax1             = fig.add_subplot(111, xscale='log', yscale='linear')

# Plot band limits in magnitude vs wavelength axes
for count, (cent_pos, depth, band_width) in enumerate(zip(central_pos_um, depth_5sigma_AB, central_pos_width_um)):
    ax1.errorbar(cent_pos, depth, xerr=band_width/2, ls='None', marker='None',\
         ecolor=plt.get_cmap(gv.cmap_bands, len(filter_names))(count / len(filter_names)),\
              elinewidth=4, path_effects=gf.pe1, zorder=10)
band_texts = []
for count, filt_name in enumerate(filter_names):
    centering = 'center'
    valign    = 'bottom'
    if 'W2' in filt_name: centering = 'right'
    if 'W2' in filt_name: centering = 'left'
    if 'y'  in filt_name: centering = 'left'
    if 'Ks' in filt_name: centering = 'left'
    # if 'NUV' in filt_name: centering = 'left'
    # if 'CW' in filt_name: valign = 'top'
    band_texts.append(ax1.annotate(filt_name.replace('-AW', '').replace('-CW', ''), (central_pos_um[count], depth_5sigma_AB[count]),\
         textcoords='offset points', xytext=(-3, 3.5), fontsize=14,\
         ha=centering, path_effects=gf.pe2, zorder=10, va=valign))
#n_iter_adjust = adjust_text(band_texts, arrowprops=dict(arrowstyle="-", color='k', lw=1.5))
#print(f'Text adjustment took {n_iter_adjust} iterations')
ax1.set_ylim(bottom=4.0, top=24.0)
ax1.invert_yaxis()

# Add extra axis in flux units
ax2      = ax1.twinx()
lims_AB  = ax1.get_ylim() * u.mag(u.AB)
lims_uJy = np.array(lims_AB.to(u.uJy).value)
ax2.set_ylim(tuple(lims_uJy))
ax2.tick_params(which='both', top=False, right=True, direction='in')
ax2.tick_params(which='both', bottom=False, left=False, direction='in')
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(which='major', length=8, width=1.5)
ax2.tick_params(which='minor', length=4, width=1.5)
ax2.set_ylabel('$\mathrm{Flux}_{5\sigma\, \mathrm{Depth}}\, [\mu \mathrm{Jy}]$', size=18)
ax2.set_yscale('log')

# Add extra axis in frequency units
ax3      = ax1.twiny()
lims_um  = ax1.get_xlim()
lims_GHz = c.value * 1e-9 / (np.array(lims_um) * 1e-6)
ax3.set_xlim(tuple(lims_GHz))
ax3.tick_params(which='both', top=True, right=False, direction='in')
ax3.tick_params(which='both', bottom=False, left=False, direction='in')
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.tick_params(which='major', length=8, width=1.5)
ax3.tick_params(which='minor', length=4, width=1.5)
ax3.set_xlabel('$\mathrm{Frequency\, [GHz]}$', size=18)
ax3.set_xscale('log')

# Add AGN SED
z_zero_proxy         = 1e-2  # closest to z = 0

# Define rest-frame SED
if AGN_sed == 'observed':
    AGN_wave_rf     = AGN_wave * (1 + orig_z)  # Rest-frame wavelength
    AGN_flux_uJy    = AGN_flux.to(u.uJy)
    # AGN_flux_rf_uJy = AGN_flux_uJy * redshift_factor_orig
    AGN_flux_rf_uJy = AGN_flux_uJy * (1 + orig_z)
elif AGN_sed == 'rest-frame':
    AGN_wave_rf     = AGN_wave
    AGN_wave        = AGN_wave * (1 + orig_z)
    AGN_flux_rf_uJy = AGN_flux.to(u.uJy)
    AGN_flux_uJy    = AGN_flux_rf_uJy

#ax2.plot(AGN_wave_rf.value, AGN_flux_rf_uJy.value, zorder=1, color='k',\
#     lw=2.5, label=f'Mrk231 - rest-frame')  # rest-frame
ax2.plot(AGN_wave.value, AGN_flux_rf_uJy.value / (1 + orig_z),\
     zorder=1, color='indigo', lw=2.5, label=f'Mrk231 - z={orig_z}', alpha=1.0)  # observed
max_z_plot = 7
for z in np.linspace(z_zero_proxy, max_z_plot, 8):
    ax2.plot(AGN_wave_rf.value * (1 + z), AGN_flux_uJy.value / (1 + z),\
         zorder=0, color='Gray', lw=1, alpha=np.abs(1 - z / (max_z_plot + 2)), ls='--')
    y_pos = int(np.where(AGN_wave_rf.value * (1 + z) >= AGN_wave_rf.value[450])[0][0])
    ax2.annotate(f'z={z:.2f}', (AGN_wave_rf.value[450], (AGN_flux_uJy.value / (1 + z))[y_pos]),\
         textcoords='offset points', xytext=(0, 0), fontsize=7,\
         ha='left', zorder=10, va='bottom', alpha=np.abs(1 - z / (max_z_plot + 2)))
ax1.set_xlim(left=2e-1, right=5e6)

ax1.tick_params(which='both', top=False, right=False, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
ax1.set_xlabel('$\mathrm{Wavelength}\, [\mu \mathrm{m}]$', size=18)
ax1.set_ylabel('$m_{5\sigma\, \mathrm{Depth}}\, \mathrm{[AB]}$', size=18)
plt.setp(ax2.spines.values(), linewidth=3.5)
plt.setp(ax2.spines.values(), linewidth=3.5)
ax2.legend(loc=8, fontsize=12, title='Model AGN', title_fontsize=12)
plt.tight_layout()

ax1.set_zorder(ax2.get_zorder()+1)
ax1.set_frame_on(False)

if save_plot_flag:
    plt.savefig(gv.plots_path + 'surveys_depth_HETDEX.pdf')
if show_plot_flag:
    plt.show()
if not show_plot_flag:
    plt.clf()
