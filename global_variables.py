#!/usr/bin/env python

# File with most used
# variables in this project.
# File paths, file names, etc.

# Paths
cat_path           = '../../Catalogs/'  # relative path to the same directory
plots_path         = 'plots/'
models_path        = 'models/'
preds_path         = 'pred_rAGN/'
moc_path           = 'moc_files/'
tmp_shap_path      = 'tmp_shap/'
indices_path       = 'subsets_indices/'

# Boolean flags
use_5sigma         = True  # use files with 5-sigma magnitude imputation

# File versions
mqc_version        = '7_4d'  # '7_2'

# Catalogue file names
# # Original fits files
# fits_HETDEX         = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16.fits'  # 6729647 objects (6.7e6)
# fits_S82            = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16.fits'
# fits_COSMOS         = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16.fits'
# fits_S82_Ananna_17  = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_Ananna_17_zsp.fits'
# # Non imputed h5 files
# file_non_imp_HETDEX = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16_non_imp.h5'
# file_non_imp_S82    = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_non_imp.h5'
# file_non_imp_COSMOS = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16_non_imp.h5'
# # Imputed h5 files
# file_HETDEX         = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16_5sigma_imp.h5'  # 6729647 objects (6.7e6)
# file_S82            = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_5sigma_imp.h5'
# file_COSMOS         = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16_5sigma_imp.h5'
# file_S82_Ananna_17  = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_Ananna_17_zsp_5sigma_imp.h5'
# New Catalogue file names
# fits_HETDEX         = 'CatWISE2020_LoTSS_area_large_LoTSS_PS1_ALLWISE_2MASS_MQC_74D_SDSS_DR16_5arcsec.fits'  # 15136878 objects (1.5e7)
# fits_S82            = 'CatWISE2020_VLAS82_area_large_VLAS82_PS1_ALLWISE_2MASS_MQC_74D_SDSS_DR16_5arcsec.fits'  # 3590306 objects
# fits_COSMOS         = ''
# fits_S82_Ananna_17  = 'CatWISE2020_VLAS82_area_large_VLAS82_PS1_ALLWISE_2MASS_MQC_74D_SDSS_DR16_5arcsec_Ananna_17.fits'  # 2558 objects
# New Catalogue file names
fits_HETDEX         = 'CatWISE2020_LoTSS_area_large_LoTSS_PS1_ALLWISE_2MASS_MQC_74D_SDSS_DR16_1_1arcsec.fits'  # 15136878 objects (1.5e7)
fits_S82            = 'CatWISE2020_VLAS82_area_large_VLAS82_PS1_ALLWISE_2MASS_MQC_74D_SDSS_DR16_1_1arcsec.fits'  # 3590306 objects
fits_COSMOS         = ''
fits_S82_Ananna_17  = 'CatWISE2020_VLAS82_area_large_VLAS82_PS1_ALLWISE_2MASS_MQC_74D_SDSS_DR16_1_1arcsec_Ananna_17.fits'  # 804 objects
# Non imputed h5 files
file_non_imp_HETDEX = fits_HETDEX.replace('.fits', '_non_imp.h5')
file_non_imp_S82    = fits_S82.replace('.fits', '_non_imp.h5')
file_non_imp_COSMOS = fits_COSMOS.replace('.fits', '_non_imp.h5')
# Imputed h5 files
file_HETDEX         = fits_HETDEX.replace('.fits', '_imp.h5')  # 15136878 objects (1.5e7)
file_S82            = fits_S82.replace('.fits', '_imp.h5')
file_COSMOS         = fits_COSMOS.replace('.fits', '_imp.h5')
file_S82_Ananna_17  = fits_S82_Ananna_17.replace('.fits', '_imp.h5')

# Fields properties
# Areas (deg2)
area_HETDEX         = 424
area_S82            = 92
area_COSMOS         = 4  # Not real value. Placeholder

# Model names with train, test, calibration, and validation sub-sets
# # Old Stacked models
# star_model         = 'classification_star_no_star_ago_29_2022'
# AGN_gal_model      = 'classification_AGN_galaxy_ago_30_2022'
# radio_model        = 'classification_LOFAR_detect_sep_07_2022'
# full_z_model       = 'regression_z_sep_09_2022'
# high_z_model       = 'regression_high_z_sep_10_2022'
# # Old Calibrated models
# cal_str_model      = 'cal_classification_star_no_star_ago_29_2022.joblib'
# cal_AGN_gal_model  = 'cal_classification_AGN_galaxy_ago_30_2022.joblib'
# cal_radio_model    = 'cal_classification_LOFAR_detect_sep_07_2022.joblib'
# Stacked models
# star_model         = 'classification_star_no_star_dec_07_2022'
# AGN_gal_model      = 'classification_AGN_galaxy_dec_08_2022'
# radio_model        = 'classification_LOFAR_detect_dec_09_2022'
# full_z_model       = 'regression_z_dec_10_2022'
# high_z_model       = 'regression_high_z_dec_11_2022'

# star_model         = 'classification_star_no_star_dec_12_2022'
# AGN_gal_model      = 'classification_AGN_galaxy_dec_13_2022'
# radio_model        = 'classification_LOFAR_detect_dec_14_2022'
# full_z_model       = 'regression_z_dec_15_2022'
# high_z_model       = 'regression_high_z_dec_16_2022'

star_model         = 'classification_star_no_star_dec_17_2022'
AGN_gal_model      = 'classification_AGN_galaxy_dec_18_2022'
radio_model        = 'classification_LOFAR_detect_dec_19_2022'
full_z_model       = 'regression_z_dec_20_2022'
high_z_model       = 'regression_high_z_dec_121_2022'
# Calibrated models
cal_str_model      = 'cal_' + star_model    + '.joblib'
cal_AGN_gal_model  = 'cal_' + AGN_gal_model + '.joblib'
cal_radio_model    = 'cal_' + radio_model   + '.joblib'

# Seeds
seed               = 42

# Thresholds
# Beta for beta-scores
beta_F             = 1.1  # beta positive real value
# Naive values
naive_star_thresh  = 0.5
naive_AGN_thresh   = 0.5
naive_radio_thresh = 0.5
# Values obtained with train, test, calibration, and validation sub-sets
# PR-optimised models (with train+test sub-set)
# star_thresh        = 0.1873511777
# AGN_thresh         = 0.4347533096
# radio_thresh       = 0.4999910122
# PR-optimised models (with train+test sub-set) new data
# star_thresh        = 0.1873511777 # old value
# AGN_thresh         = 0.4933696292
# radio_thresh       = 0.1253210681
# 
star_thresh        = 0.1873511777 # old value
AGN_thresh         = 0.5000115951
radio_thresh       = 0.9815369877

# Calibrated and PR-optimised models (with calibration sub-set)
# cal_str_thresh     = 0.6007345636412931
# cal_AGN_thresh     = 0.39889114096089423
# cal_radio_thresh   = 0.3174334601810781
# Calibrated and PR-optimised models (with calibration sub-set) new data
# cal_str_thresh     = 0.6007345636412931 # old value
# cal_AGN_thresh     = 0.40655238948425537
# cal_radio_thresh   = 0.20125615854582377
#
cal_str_thresh     = 0.6007345636412931 # old value
cal_AGN_thresh     = 0.34895396724527294
cal_radio_thresh   = 0.2046047064139296
# High redshift limit
high_z_limit       = 2.0  # 3.6

# Colours and colormaps
cmap_bands         = 'cmr.pride'
cmap_shap          = 'cmr.guppy'  # cmr.pride, cet_CET_R3 cmr.wildfire cmr.guppy
cmap_conf_matr     = 'cet_dimgray_r'
cmap_z_plots       = 'cet_linear_kryw_5_100_c64_r'