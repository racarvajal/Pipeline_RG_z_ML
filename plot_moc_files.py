#!/usr/bin/env python

# Plot MOC files to
# display coverage
# of catalogues

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mocpy import MOC, World2ScreenMPL
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
import colorcet as cc
import cmasher as cmr
import global_variables as gv
import global_functions as gf

mpl.rcdefaults()

save_flag   = False
show_flag   = False

used_area   = 'HETDEX'  # 'HETDEX', 'S82', 'COSMOS'

file_HETDEX = 'CDS-J-A+A-622-A1-LoTSSDR1_MOC.fits'
file_S82    = 'CDS-J-AJ-142-3-VLA_STRIPE82_MOC.fits'
file_COSMOS = 'CDS-J-A+A-602-A1-table1_VLA_COSMOS3GHZ_MOC.fits'

center_HETDEX = SkyCoord('13:02:25.27982 +52:45:00.0000', unit=(u.hourangle, u.deg), frame='icrs')
center_S82    = SkyCoord('00:13:07.85666 -00:07:37.6704', unit=(u.hourangle, u.deg), frame='icrs')
center_COSMOS = SkyCoord('10:00:26.37901 +02:13:45.7597', unit=(u.hourangle, u.deg), frame='icrs')

fov_HETDEX    = 50 * u.deg
fov_S82       = 70 * u.deg
fov_COSMOS    = 3  * u.deg

figsize_HETDEX = (7, 2.5)
figsize_S82    = (10, 2)
figsize_COSMOS = (5.6, 4.5)

file_names   = {'HETDEX': file_HETDEX, 'S82': file_S82, 'COSMOS': file_COSMOS}
center_names = {'HETDEX': center_HETDEX, 'S82': center_S82, 'COSMOS': center_COSMOS}
field_fovs   = {'HETDEX': fov_HETDEX, 'S82': fov_S82, 'COSMOS': fov_COSMOS}
field_fsizes = {'HETDEX': figsize_HETDEX, 'S82': figsize_S82, 'COSMOS': figsize_COSMOS}

file_name    = file_names[used_area]
field_center = center_names[used_area]
field_fov    = field_fovs[used_area]
field_fsize  = field_fsizes[used_area]

moc = MOC.load(gv.moc_path + file_name, 'fits')
moc = moc.degrade_to_order(9)

fig = plt.figure(111, figsize=field_fsize)
with World2ScreenMPL(fig, 
        fov=field_fov,
        center=field_center,
        coordsys='icrs',
        rotation=Angle(0, u.degree),
        projection='AIT') as wcs:
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    # Call fill with a matplotlib ax and the `~astropy.wcs.WCS` wcs object.
    moc.fill(ax=ax, wcs=wcs, alpha=1.0, fill=True, color=plt.get_cmap('cet_gouldian')(0.5))
    moc.border(ax=ax, wcs=wcs, alpha=1.0, color='k', lw=3.0)
plt.title(used_area, size=12)
plt.grid(color='k', linestyle='dashed', alpha=0.5)
ax.tick_params(which='major', direction='in')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(which='major', length=8, width=1.5)
ax.tick_params(which='minor', length=4)
ax.set_xlabel('ra', size=12)
ax.set_ylabel('dec', size=12)
plt.setp(ax.spines.values(), linewidth=3.5)
# plt.tight_layout()
if save_flag:
    plt.savefig(gv.plots_path + f'field_area_moc_{used_area}.pdf')
if show_flag:
    plt.show()
if not show_flag:
    plt.clf()