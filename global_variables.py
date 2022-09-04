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

# Boolean flags
use_5sigma         = True  # use files with 5-sigma magnitude imputation

# File versions
mqc_version        = '7_4d'  # '7_2'

# Catalogue file names
# Original fits files
fits_HETDEX         = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16.fits'  # 6729647 objects (6.7e6)
fits_S82            = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16.fits'
fits_COSMOS         = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16.fits'
fits_S82_Ananna_17  = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_Ananna_17_zsp.fits'
# Non imputed h5 files
file_non_imp_HETDEX = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16_non_imp.h5'
file_non_imp_S82    = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_non_imp.h5'
file_non_imp_COSMOS = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16_non_imp.h5'
# Imputed h5 files
file_HETDEX         = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16_5sigma_imp.h5'  # 6729647 objects (6.7e6)
file_S82            = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_5sigma_imp.h5'
file_COSMOS         = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16_5sigma_imp.h5'
file_S82_Ananna_17  = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_Ananna_17_zsp_5sigma_imp.h5'

# Model file names
# Old model names with train, test, and validation sub-sets
# Stacked models
# star_model         = 'classification_star_no_star_jun_30_2022'
# AGN_gal_model      = 'classification_AGN_galaxy_ago_03_2022'
# radio_model        = 'classification_radio_detect_ago_02_2022'
# full_z_model       = 'regression_z_ago_04_2022'
# high_z_model       = 'regression_z_jul_19_2022'
# # Calibrated models
# cal_str_model      = 'cal_classification_star_no_star_ago_25_2022.joblib'
# cal_AGN_gal_model  = 'cal_classification_AGN_galaxy_ago_03_2022.joblib'
# cal_radio_model    = 'cal_classification_radio_detect_ago_02_2022.joblib'
# New model names with train, test, calibration, and validation sub-sets
# Stacked models
star_model         = 'classification_star_no_star_ago_29_2022'
AGN_gal_model      = 'classification_AGN_galaxy_ago_30_2022'
radio_model        = 'classification_radio_detect_ago_31_2022'
full_z_model       = 'regression_z_sep_01_2022'
high_z_model       = 'regression_high_z_sep_02_2022'
# Calibrated models
cal_str_model      = 'cal_classification_star_no_star_ago_29_2022.joblib'
cal_AGN_gal_model  = 'cal_classification_AGN_galaxy_ago_30_2022.joblib'
cal_radio_model    = 'cal_classification_radio_detect_ago_31_2022.joblib'

# Seeds
seed               = 42

# Thresholds
# Naive values
naive_star_thresh  = 0.5
naive_AGN_thresh   = 0.5
naive_radio_thresh = 0.5
# PR-optimised
# Values obtained with train, test, and validation sub-sets
# star_thresh        = 0.45967
# AGN_thresh         = 0.44535
# radio_thresh       = 0.495866031
# # Calibrated and PR-optimised models
# cal_str_thresh     = 0.17790
# cal_AGN_thresh     = 0.34611
# cal_radio_thresh   = 0.24164
# Values obtained with train, test, calibration, and validation sub-sets
# PR-optimised models (with train+test sub-set)
star_thresh        = 0.26115
AGN_thresh         = 0.43475
radio_thresh       = 0.497101925
# Calibrated and PR-optimised models (with calibration sub-set)
cal_str_thresh     = 0.6007345636412931
cal_AGN_thresh     = 0.574165428416621
cal_radio_thresh   = 0.2740401572257431
# High redshift limit
high_z_limit       = 2.0  # 3.6

# Colours and colormaps
cmap_shap          = 'cmr.guppy'  # cmr.pride, cet_CET_R3 cmr.wildfire cmr.guppy
cmap_conf_matr     = 'cet_dimgray_r'
cmap_z_plots       = 'cet_linear_kryw_5_100_c64_r'