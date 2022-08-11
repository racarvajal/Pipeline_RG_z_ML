#!/usr/bin/env python

# File with most used
# variables in this project.
# File paths, file names, etc.

# Paths
cat_path           = '../../Catalogs/'  # relative path to the same directory
plots_path         = 'plots/'
models_path        = 'models/'

# Boolean flags
use_5sigma         = True  # use files with 5-sigma magnitude imputation

# File versions
mqc_version        = '7_4d'  # '7_2'

# Catalogue file names
file_HETDEX        = 'CatWISE2020_VLASS_LOFAR_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_LOLSS_SDSS_DR16_5sigma_imp.h5'  # 6729647 objects (6.7e6)
file_S82           = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_5sigma_imp.h5'
file_COSMOS        = 'CatWISE2020_COSMOS_MILLIQUAS_7_4d_COSMOSVLA3_PS1_GALEX_TGSS_VLASS_XMM_2MASS_ALLWISE_SDSS_DR16_5sigma_imp.h5'
file_S82_Ananna_17 = 'CatWISE2020_S82_VLASS_VLAS82_PS1_GALEX_TGSS_XMM_2MASS_MILLIQUAS_7_4d_ALLWISE_SDSS_DR16_Ananna_17_zsp_5sigma_imp.h5'

# Model file names
# Stacked models
AGN_gal_model      = 'classification_AGN_galaxy_ago_03_2022'
radio_model        = 'classification_radio_detect_ago_02_2022'
full_z_model       = 'regression_z_ago_04_2022'
high_z_model       = 'regression_z_jul_19_2022'
# Calibrated models
cal_AGN_gal_model  = 'cal_classification_AGN_galaxy_ago_03_2022.joblib'
cal_radio_model    = 'cal_classification_radio_detect_ago_02_2022.joblib'

# Seeds
seed               = 42

# Thresholds
# Naive values
naive_AGN_thresh   = 0.5
naive_radio_thresh = 0.5
# PR-optimised
AGN_thresh         = 0.44535
radio_thresh       = 0.500005978
# Calibrated models
cal_AGN_thresh     = 0.38261
cal_radio_thresh   = 0.31594
# High redshift limit
high_z_limit       = 3.6
