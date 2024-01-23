#!/usr/bin/env python

# Plot MOC files to
# display coverage
# of catalogues

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
# from mocpy import MOC, WCS
from mocpy import MOC
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from astropy.visualization import simple_norm
from astropy import units as u
from astropy.io import fits
import colorcet as cc
import cmasher as cmr
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

save_flag   = False
show_flag   = False

used_area   = 'HETDEX'  # 'HETDEX', 'S82', 'COSMOS'

plot_all_area = True

file_HETDEX = 'CDS-J-A+A-622-A1-LoTSSDR1_MOC.fits'
file_S82    = 'CDS-J-AJ-142-3-stripe82_MOC.fits'  # 'CDS-J-AJ-142-3-VLA_STRIPE82_MOC.fits'
file_COSMOS = 'CDS-J-A+A-602-A1-table1_VLA_COSMOS3GHZ_MOC.fits'

center_HETDEX = SkyCoord('13:02:25.27982 +52:45:00.0000', unit=(u.hourangle, u.deg), frame='icrs')
center_S82    = SkyCoord('00:13:07.85666 -00:07:37.6704', unit=(u.hourangle, u.deg), frame='icrs')
center_COSMOS = SkyCoord('10:00:26.37901 +02:13:45.7597', unit=(u.hourangle, u.deg), frame='icrs')

fov_HETDEX    = 20 * u.deg
fov_S82       = 20 * u.deg
fov_COSMOS    = 2.2  * u.deg

figsize_HETDEX = (8, 3.5)
figsize_S82    = (10, 3)
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
moc = moc.degrade_to_order(8)

# load unWISE cutout covering the respective areas
# could use unWISE colours: 'color-W2-W1W2-W1'
# or one single unWISE band: 'W1'
band_2_plot = 'W1' # 'color-W2-W1W2-W1', 'W1'
unWISE_HETDEX = f'https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips=CDS%2FP%2FunWISE%2F{band_2_plot}&width=2000&height=630&fov=50&projection=AIT&coordsys=icrs&rotation_angle=0.0&ra=195.60533258333334&dec=52.75&format=fits'
unWISE_S82    = f'https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips=CDS%2FP%2FunWISE%2F{band_2_plot}&width=2000&height=200&fov=65&projection=AIT&coordsys=icrs&rotation_angle=0.0&ra=3.2827360833333334&dec=-0.12713066666666667&format=fits'
unWISE_COSMOS = f'https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips=CDS%2FP%2FunWISE%2F{band_2_plot}&width=1000&height=1000&fov=2.1&projection=AIT&coordsys=icrs&rotation_angle=0.0&ra=150.10991254166663&dec=2.2293776944444446&format=fits'

field_unWISE = {'HETDEX': unWISE_HETDEX, 'S82': unWISE_S82, 'COSMOS': unWISE_COSMOS}
with fits.open(field_unWISE[used_area]) as hdul:
    # create WCS from unWISE image header
    if band_2_plot == 'color-W2-W1W2-W1':
        unwise_wcs  = WCS(header=hdul[0].header).dropaxis(2)
        data_unWISE = np.transpose(hdul[0].data, (1, 2, 0))
    if band_2_plot == 'W1':
        unwise_wcs  = WCS(header=hdul[0].header)
        data_unWISE = hdul[0].data
    # to compute skycoords for every pixel of the image
    width  = hdul[0].header["NAXIS1"]
    height = hdul[0].header["NAXIS2"]

if not plot_all_area:
    xv, yv      = np.meshgrid(np.arange(0, width), np.arange(0, height))
    skycoords   = unwise_wcs.pixel_to_world(xv, yv)
    ra, dec     = skycoords.icrs.ra.deg, skycoords.icrs.dec.deg
    mask_in_moc = moc.contains_lonlat(ra * u.deg, dec * u.deg)

    img_inverse = data_unWISE.copy()
    try:
        img_inverse[~mask_in_moc] = np.nan
    except ValueError:
        img_inverse[~mask_in_moc] = 1

if plot_all_area:
    im_to_plot = data_unWISE
if not plot_all_area:
    im_to_plot = img_inverse

fig = plt.figure(111, figsize=field_fsize)
ax = fig.add_subplot(1, 1, 1, projection=unwise_wcs)
im = ax.imshow(
    im_to_plot,
    origin='lower', cmap=plt.get_cmap('cet_CET_L17'),
    norm=simple_norm(data_unWISE, 'log', min_percent=0, max_percent=99.73),
)
# if plot_all_area:
#     moc.fill(ax=ax, wcs=unwise_wcs, alpha=0.6, 
#     color='w', fill=True, linewidth=1.0)
moc.border(ax=ax, wcs=unwise_wcs, alpha=1.0, color='w', lw=1.5)
plt.grid(color='k', linestyle='dashed', alpha=0.5)
ax.tick_params(which='major', direction='in')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(which='major', length=0, width=1.5)
ax.tick_params(which='minor', length=0)
ax.set_xlabel('$\mathrm{R.A.}$', size=12)
ax.set_ylabel('$\mathrm{Dec}$', size=12)
plt.setp(ax.spines.values(), linewidth=2.5)
plt.tight_layout()
if save_flag:
    plt.savefig(gv.plots_path + f'field_unWISE_back_area_moc_{used_area}.pdf', bbox_inches='tight')
if show_flag:
    plt.show()
if not show_flag:
    plt.clf()
