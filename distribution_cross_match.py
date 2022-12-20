#!/usr/bin/env python

# Code to understand the distribution 
# of separations between sources in 
# CatWISE2020 and their counterparts in 
# Pan-STARRS DR1. These results can be 
# used to select optimal search radius 
# for cross-matching.

import numpy as np
from numpy.random import default_rng
from itertools import combinations
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from mocpy import MOC
import pandas as pd
import global_variables as gv

# MOC files
file_name_HETDEX_MOC         = 'TAP_1_J_A+A_622_A1_lotssdr1_MOC_8_level.fits'
file_name_S82_MOC            = 'CDS-J-AJ-142-3-stripe82_MOC_8_level.fits'

# Files with sources in the HETDEX field
file_name_CatWISE2020_HETDEX = 'CatWISE2020_LoTSS_area_large.fits'
file_name_Pan_STARRS_HETDEX  = 'PS1_LoTSS_area_large.fits'

file_name_CatWISE2020_S82    = 'CatWISE2020_LoTSS_area_large.fits'
file_name_Pan_STARRS_S82     = 'PS1_VLAS82_area_large.fits'

save_plot_flag = False

print('-' * 40)
print('Reading files')
CW_tab_HETDEX  = Table.read(gv.cat_path + file_name_CatWISE2020_HETDEX, 
                            format='fits')[['RA_ICRS', 'DE_ICRS']]
CW_tab_S82     = Table.read(gv.cat_path + file_name_CatWISE2020_S82, 
                            format='fits')[['RA_ICRS', 'DE_ICRS']]
PS1_tab_HETDEX = Table.read(gv.cat_path + file_name_Pan_STARRS_HETDEX, 
                            format='fits')[['RAJ2000', 'DEJ2000']]
PS1_tab_S82    = Table.read(gv.cat_path + file_name_Pan_STARRS_S82, 
                            format='fits')[['RAJ2000', 'DEJ2000']]

HETDEX_MOC     = MOC.from_fits(gv.cat_path + file_name_HETDEX_MOC)
# S82_MOC        = MOC.from_fits(gv.cat_path + file_name_S82_MOC)

print(len(PS1_tab_HETDEX))
dens_sources_CW_HETDEX  = len(PS1_tab_HETDEX) / (gv.area_HETDEX * 3600**2)
dens_sources_CW_S82     = len(PS1_tab_S82) / (gv.area_S82 * 3600**2)
print(dens_sources_CW_HETDEX)

# Generate uniform distribution of sources in fields
rng                     = default_rng(seed=gv.seed)
RA_PS1_HETDEX_rand      = rng.uniform(low=np.nanmin(PS1_tab_HETDEX['RAJ2000']), 
                                      high=np.nanmax(PS1_tab_HETDEX['RAJ2000']), 
                                      size=len(PS1_tab_HETDEX)) * u.deg
DEC_PS1_HETDEX_rand     = rng.uniform(low=np.nanmin(PS1_tab_HETDEX['DEJ2000']), 
                                      high=np.nanmax(PS1_tab_HETDEX['DEJ2000']), 
                                      size=len(PS1_tab_HETDEX)) * u.deg
coords_PS1_HETDEX_rand  = SkyCoord(RA_PS1_HETDEX_rand, DEC_PS1_HETDEX_rand, unit=u.deg)

# Look for (and discard) sources outside proper area
idx_out_PS1_HETDEX_rand = HETDEX_MOC.contains(ra=coords_PS1_HETDEX_rand.ra, dec=coords_PS1_HETDEX_rand.dec, keep_inside=False)




# radius_range            = np.arange(0.0, 10, 0.3) # in arcsec in steps of 0.05 arcsec
# radius_range[0]         = 0.1
# num_matches_HETDEX_real = np.zeros_like(radius_range)
# num_matches_HETDEX_rand = np.zeros_like(radius_range)
# num_matches_S82_real    = np.zeros_like(radius_range)
# num_matches_S82_rand    = np.zeros_like(radius_range)


# coords_CW
# coords_PS1

# for count, radius in enumerate(radius_range):
#     id_CW_real, id_PS1_real, sep2d_real, sep3d_real = search_around_sky(coords_CW_HETDEX, coords_PS1_HETDEX_real, seplimit=radius)
#     id_CW_rand, id_PS1_rand, sep2d_rand, sep3d_rand = search_around_sky(coords_CW_HETDEX, coords_PS1_HETDEX_rand, seplimit=radius)
#     num_matches_HETDEX_real[count]                  = len(np.unique(id_CW))
#     num_matches_HETDEX_rand[count]                  = len(np.unique(id_CW))
    
# for count, radius in enumerate(radius_range):
#     id_CW_real, id_PS1_real, sep2d_real, sep3d_real = search_around_sky(coords_CW_S82, coords_PS1_S82_real, seplimit=radius)
#     id_CW_rand, id_PS1_rand, sep2d_rand, sep3d_rand = search_around_sky(coords_CW_S82, coords_PS1_S82_rand, seplimit=radius)
#     num_matches_S82_real[count]                     = len(np.unique(id_CW))
#     num_matches_S82_rand[count]                     = len(np.unique(id_CW))





# for count, radius in enumerate(radius_range):
#     area_radius               = np.pi * radius**2
#     num_sources_HETDEX        = dens_sources_CW_HETDEX * area_radius
#     num_matches_HETDEX[count] = num_sources_HETDEX
#     num_sources_S82           = dens_sources_CW_S82 * area_radius
#     num_matches_S82[count]    = num_sources_S82
    
# plt.plot(radius_range, num_matches_HETDEX)
# plt.plot(radius_range, num_matches_S82)
# plt.axhline(y=0.1, ls='--')
# plt.yscale('log')
# plt.show()


