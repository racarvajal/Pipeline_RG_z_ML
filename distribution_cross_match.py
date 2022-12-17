#!/usr/bin/env python

# Code to understand the distribution 
# of separations between sources in 
# CatWISE2020 and their counterparts in 
# Pan-STARRS DR1. These results can be 
# used to select optimal search radius 
# for cross-matching.

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
import pandas as pd
import global_variables as gv

# Files with sources in the HETDEX field
file_name_CatWISE2020_HETDEX = 'CatWISE2020_LoTSS_area_large.fits'
file_name_Pan_STARRS_HETDEX  = 'PS1_LoTSS_area_large.fits'

file_name_CatWISE2020_S82 = 'CatWISE2020_LoTSS_area_large.fits'
file_name_Pan_STARRS_S82  = 'PS1_VLAS82_area_large.fits'

save_plot_flag = False


print('-' * 40)
print('Reading files')
PS1_tab_HETDEX = Table.read(gv.cat_path + file_name_Pan_STARRS_HETDEX,  format='fits')[['RAJ2000', 'DEJ2000']]
PS1_tab_S82    = Table.read(gv.cat_path + file_name_Pan_STARRS_S82,  format='fits')[['RAJ2000', 'DEJ2000']]

print(len(PS1_tab_HETDEX))
dens_sources_CW_HETDEX = len(PS1_tab_HETDEX) / (gv.area_HETDEX * 3600**2)
dens_sources_CW_S82    = len(PS1_tab_S82) / (gv.area_S82 * 3600**2)
print(dens_sources_CW_HETDEX)

radius_range       = np.arange(0.0, 10, 0.1) # in arcsec in steps of 0.05 arcsec
radius_range[0]    = 0.1
num_matches_HETDEX = np.zeros_like(radius_range)
num_matches_S82    = np.zeros_like(radius_range)

for count, radius in enumerate(radius_range):
    area_radius               = np.pi * radius**2
    num_sources_HETDEX        = dens_sources_CW_HETDEX * area_radius
    num_matches_HETDEX[count] = num_sources_HETDEX
    num_sources_S82           = dens_sources_CW_S82 * area_radius
    num_matches_S82[count]    = num_sources_S82
    
plt.plot(radius_range, num_matches_HETDEX)
plt.plot(radius_range, num_matches_S82)
plt.axhline(y=0.1, ls='--')
plt.yscale('log')
plt.show()


