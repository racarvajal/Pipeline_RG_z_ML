#!/usr/bin/env python

# Code to create catalogue files from
# HETDEX, Stripe 82, and COSMOS data.
# It converts fluxes to magnitudes,
# imputes missing data and
# creates new features out of
# original quantities (colours,
# ratios, flags, etc.).

import numpy as np
from itertools import combinations
from astropy.table import Table
from astropy import units as u
import pandas as pd
import global_variables as gv

def create_AGN_gal_flags(initial_tab, imputed_df, AGN_types, mqc_version):
    filt_NLAGN                = create_MQC_filter(initial_tab, AGN_types[mqc_version])
    imputed_df['is_str']      = (np.array(initial_tab['spCl'] == 'STAR  ')).astype(int)
    tmp_AGN_0                 = np.array(initial_tab['Z'] > 0)
    tmp_AGN_1                 = np.array((initial_tab['Z'] * 10) % 1 == 0)  # non spec z only
    tmp_AGN_2                 = np.array([(st.startswith('B') | st.startswith('R') |
                                  st.startswith('X') | st.startswith('2')) for st in initial_tab['TYPE']])
    imputed_df['is_SDSS_QSO'] = (np.array(initial_tab['spCl'] == 'QSO   ')).astype(int)
    imputed_df['is_AGN']      = (tmp_AGN_0 & ~(tmp_AGN_1 & tmp_AGN_2) & filt_NLAGN).astype(int)
    imputed_df['is_SDSS_gal'] = (np.array(initial_tab['spCl'] == 'GALAXY')).astype(int)
    imputed_df['is_gal']      = (imputed_df['is_SDSS_gal'] & ~imputed_df['is_AGN']).astype(int)
    return imputed_df

def create_colours(imputed_df, mag_list):
    for mags_pair in combinations(mag_list, 2):
        colour_name             = mag_names_short[mags_pair[0]] + '_' + mag_names_short[mags_pair[1]]
        imputed_df[colour_name] = imputed_df[mags_pair[0]] - imputed_df[mags_pair[1]]
    return imputed_df

def create_ratios(imputed_df):
    imputed_df['r/z']     = imputed_df.loc[:, 'rmag']     / imputed_df.loc[:, 'zmag']
    imputed_df['i/y']     = imputed_df.loc[:, 'imag']     / imputed_df.loc[:, 'ymag']
    imputed_df['w1/w3']   = imputed_df.loc[:, 'W1mproPM'] / imputed_df.loc[:, 'W3mag']
    imputed_df['w1/w4']   = imputed_df.loc[:, 'W1mproPM'] / imputed_df.loc[:, 'W4mag']
    imputed_df['w2/w4']   = imputed_df.loc[:, 'W2mproPM'] / imputed_df.loc[:, 'W4mag']
    imputed_df['J/K']     = imputed_df.loc[:, 'Jmag']     / imputed_df.loc[:, 'Kmag']
    imputed_df['FUV/K']   = imputed_df.loc[:, 'FUVmag']   / imputed_df.loc[:, 'Kmag']
    imputed_df['g/J']     = imputed_df.loc[:, 'gmag']     / imputed_df.loc[:, 'Jmag']
    imputed_df['r/H']     = imputed_df.loc[:, 'rmag']     / imputed_df.loc[:, 'Hmag']
    imputed_df['i/K']     = imputed_df.loc[:, 'imag']     / imputed_df.loc[:, 'Kmag']
    imputed_df['z/W1']    = imputed_df.loc[:, 'zmag']     / imputed_df.loc[:, 'W1mproPM']
    imputed_df['y/W2']    = imputed_df.loc[:, 'ymag']     / imputed_df.loc[:, 'W2mproPM']
    return imputed_df

def fix_dtypes(initial_tab):
    for col in initial_tab.colnames:
        if initial_tab[col].dtype.name == 'float64':
            initial_tab[col] = initial_tab[col].astype(np.float32)
        elif 'float' in initial_tab[col].dtype.name:
            initial_tab[col].fill_value = np.nan
        elif initial_tab[col].dtype.name == 'int64':
            initial_tab[col] = initial_tab[col].astype(np.int32)
        elif 'bytes' in initial_tab[col].dtype.name:
            initial_tab[col] = initial_tab[col].astype(str)
    # Special case
    initial_tab['QPCT'] = initial_tab['QPCT'].astype(np.int32)
    initial_tab['TYPE'] = initial_tab['TYPE'].filled('')
    return initial_tab

def create_band_count(mags_df, magnitude_cols, feat_name):
    band_count_df  = pd.DataFrame()
    na_bool        = 1 - mags_df.loc[:, magnitude_cols].isna().astype(int)
    band_count_df[feat_name] = na_bool.sum(axis=1)
    return band_count_df

def create_MQC_filter(initial_tab, AGN_types):
    filters_array = [np.array(np.char.find(initial_tab['TYPE'].data, AGN_type) != -1) for AGN_type in AGN_types]
    or_in_arrays  = np.bitwise_or.reduce(filters_array)
    return or_in_arrays

def create_X_ray_detect(imputed_df, initial_tab):
    imputed_df['X_ray_detect'] = (np.array(initial_tab['FEP'] > 0) & np.isfinite(initial_tab['FEP'])).astype(int)
    return imputed_df

def create_radio_detect(imputed_df, initial_tab, radio_cols):
    filters_array = [(np.array(initial_tab[radio_col] > 0) & np.isfinite(initial_tab[radio_col])) for radio_col in radio_cols]
    or_in_arrays  = np.bitwise_or.reduce(filters_array)
    imputed_df['radio_detect'] = or_in_arrays.astype(int)
    for radio_col in radio_cols:
        imputed_df[radio_col.split('_')[-1] + '_detect'] = (np.array(initial_tab[radio_col] > 0) &\
                                                           np.isfinite(initial_tab[radio_col])).astype(int)
    return imputed_df

def create_imputation_count(mags_df, magnitude_cols, magnitude_limits, feat_name):
    imputation_count_df            = pd.DataFrame()
    imputation_bool_df             = pd.DataFrame()
    for mag in magnitude_cols:
        imputation_bool_df[mag]    = np.array(mags_df.loc[:, mag] == magnitude_limits[mag]).astype(int)
    imputation_count_df[feat_name] = imputation_bool_df.sum(axis=1)
    return imputation_count_df

file_name_clean_HETDEX_err = gv.fits_HETDEX.replace('.fits', '_err_5sigma_imp.h5')      # h5 file
file_name_clean_S82_err    = gv.fits_S82.replace('.fits', '_err_5sigma_imp.h5')         # h5 file
# file_name_clean_S82_err    = gv.fits_S82_Ananna_17.replace('.fits', '_err_5sigma_imp.h5')  # h5 file, temp line
file_name_clean_COSMOS_err = gv.fits_COSMOS.replace('.fits', '_err_5sigma_imp.h5')      # h5 file

run_HETDEX_flag = True
run_S82_flag    = True
run_COSMOS_flag = False

run_S82_full    = True  # True for use all S82 sources. False for using Ananna+17 sample

save_HETDEX_flag = True
save_S82_flag    = True
save_COSMOS_flag = False

all_vega_cols  = ['W1mproPM', 'W2mproPM', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag', 'Hmag', 'Kmag',\
                    'e_W1mproPM', 'e_W2mproPM', 'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag', 'e_Jmag',\
                    'e_Hmag', 'e_Kmag']
vega_cols      = ['W1mproPM', 'W2mproPM', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag', 'Hmag', 'Kmag']
vega_shift     = {'W1mproPM': 2.699, 'W2mproPM': 3.339, 'W1mag': 2.699, 'W2mag': 3.339, 'W3mag': 5.174,\
                    'W4mag': 6.620, 'Jmag': 0.910, 'Hmag': 1.390, 'Kmag': 1.850}

mag_cols_lim_5sigma = {'W1mproPM': 20.13, 'W2mproPM': 19.81, 'Sint_LOFAR': 17.52, 'Total_flux_VLASS': 15.21,\
                    'TotalFlux_LoLSS': 12.91, 'Stotal_TGSS': 11.18, 'Fint_VLAS82': 17.86,\
                    'Flux_COSMOSVLA3': 21.25, 'W1mag': 19.6, 'W2mag': 19.34, 'W3mag': 16.67,\
                    'W4mag': 14.62, 'gmag': 23.3, 'rmag': 23.2, 'imag': 23.1, 'zmag': 22.3,\
                    'ymag': 21.4, 'FUVmag': 20.0, 'NUVmag': 21.0, 'FEP': 57.9, 'Jmag': 17.45,\
                    'Hmag': 17.24, 'Kmag': 16.59}  # Proper (5-sigma) limits

mag_cols_for_colours = ['gmag', 'rmag', 'imag', 'zmag', 'ymag', 'Jmag', 'Hmag',\
                        'Kmag', 'W1mproPM', 'W2mproPM', 'W3mag', 'W4mag']
mag_names_short      = {'gmag': 'g', 'rmag': 'r', 'imag': 'i', 'zmag': 'z',\
                        'ymag': 'y', 'Jmag': 'J', 'Hmag': 'H', 'Kmag': 'K',\
                        'W1mproPM': 'W1', 'W2mproPM': 'W2', 'W3mag': 'W3', 'W4mag': 'W4'}

for key in mag_cols_lim_5sigma:
    mag_cols_lim_5sigma[key] = np.float32(mag_cols_lim_5sigma[key])

mag_cols_lim        = {'5sigma': mag_cols_lim_5sigma}

AGN_types_list    = {'7_4d': ['Q', 'A', 'B', 'L', 'K', 'N', 'R', 'X', '2']}

if run_HETDEX_flag:
    print('-' * 40)
    print('Working with HETDEX data')
    print('Reading files')
    HETDEX_initial_tab     = Table.read(gv.cat_path + gv.fits_HETDEX, format='fits')

    print('Fixing dtypes')
    HETDEX_initial_tab     = fix_dtypes(HETDEX_initial_tab)

    id_cols = ['objID', 'RA_ICRS', 'DE_ICRS', 'Name', 'RA_MILLI', 
                'DEC_MILLI', 'TYPE', 'Z', 'zsp', 'spCl']
    clean_cat_HETDEX_df = HETDEX_initial_tab[id_cols].to_pandas()

    zero_point_star_equiv  = u.zero_point_flux(3631.1 * u.Jy)  # zero point (AB) to Jansky

    print('Convert fluxes to magnitudes')

    # flx_cols = ['Total_flux_VLASS', 'Sint_LOFAR', 'Stotal_TGSS', 'FEP', 'TotalFlux_LoLSS']
    mJy_cols_HETDEX = [col_name for col_name in HETDEX_initial_tab.colnames if 
                HETDEX_initial_tab[col_name].unit == 'mJy' and not 
                (col_name.startswith('e') or col_name.startswith('E'))]

    for col in mJy_cols_HETDEX:
        HETDEX_initial_tab[col] = HETDEX_initial_tab[col].to(u.mag(u.AB))

    # Transform Vega magnitudes to AB magnitudes
    print('Transforming Vega to AB')
    for col in vega_cols:
        HETDEX_initial_tab[col] += vega_shift[col]

    # Fix units of magnitudes for following steps
    for col in HETDEX_initial_tab.colnames:
        if HETDEX_initial_tab[col].unit == u.mag:
            HETDEX_initial_tab[col].unit = u.mag(u.AB)
            HETDEX_initial_tab[col]      = u.Magnitude(HETDEX_initial_tab[col])

    # Select features to impute
    magnitude_cols = [col_name for col_name in HETDEX_initial_tab.colnames if 
                HETDEX_initial_tab[col_name].unit == u.mag(u.AB) and not 
                (col_name.startswith('e') or col_name.startswith('E') or col_name.endswith('MILLI')) and not
                     col_name.endswith('SDSS')]

    magnitude_cols_non_radio = [mag for mag in magnitude_cols if mag not in mJy_cols_HETDEX]

    mags_HETDEX_df      = HETDEX_initial_tab[magnitude_cols].to_pandas()
    imputed_HETDEX_df   = pd.DataFrame()

    # Create flags for X-ray and radio detection, and AGN classification
    print('Creating flags for X-ray and radio detections')
    # imputed_HETDEX_df = create_X_ray_detect(imputed_HETDEX_df, HETDEX_initial_tab)
    radio_cols_HETDEX = ['Sint_LOFAR']
    imputed_HETDEX_df = create_radio_detect(imputed_HETDEX_df, HETDEX_initial_tab, radio_cols_HETDEX)

    # Select, from MQC, sources that have been classified 
    # as host-dominated NLAGN, AGN, or QSO candidates.
    # For MQC 7.2, that means 'N', 'A', 'q'.
    print('Creating flags for AGN/Galaxy/Star classification')
    imputed_HETDEX_df = create_AGN_gal_flags(HETDEX_initial_tab, imputed_HETDEX_df, AGN_types_list, gv.mqc_version)

    # Remove columns with too high numbe of missing values
    print('Removing columns with high nullity')
    removed_cols   = []
    limit_fraction = 1.00
    for col in magnitude_cols:
        filt_temp = np.isfinite(HETDEX_initial_tab[col])
        removed   = 'NOT REMOVED'
        if np.sum(~filt_temp) > int(np.ceil(limit_fraction * len(HETDEX_initial_tab[col]))):
            removed_cols.append(col)
            magnitude_cols.remove(col)
            removed = 'REMOVED'
            print(f'column: {col}\t -\t n_bad: {np.sum(~filt_temp)}\t{removed}')

    # Create counter of measurements per source (magnitudes)
    print('Creating new features:')
    print('Creating counter of valid measurements')
    band_count_HETDEX_df = create_band_count(mags_HETDEX_df, mag_cols_for_colours, 'band_num')

    # Impute values
    non_imputed_HETDEX_df = imputed_HETDEX_df.copy()
    print('Imputing values')
    
    for col in magnitude_cols:
        imputed_HETDEX_df.loc[:, col] = mags_HETDEX_df.loc[:, col].fillna(np.float32(mag_cols_lim['5sigma'][col]), inplace=False)
        imputed_HETDEX_df.loc[:, col] = imputed_HETDEX_df.loc[:, col].mask(imputed_HETDEX_df.loc[:, col] >\
             mag_cols_lim['5sigma'][col], mag_cols_lim['5sigma'][col], inplace=False)
        non_imputed_HETDEX_df.loc[:, col] = mags_HETDEX_df.loc[:, col]

    # Create derived features
    print('Creating colours')
    imputed_HETDEX_df     = create_colours(imputed_HETDEX_df,     mag_cols_for_colours)
    non_imputed_HETDEX_df = create_colours(non_imputed_HETDEX_df, mag_cols_for_colours)

    print('Not creating magnitude ratios')
    # imputed_HETDEX_df = create_ratios(imputed_HETDEX_df)
    
    # Create counter of imputed measurements per source (magnitudes)
    print('Creating counter of valid measurements')
    imputed_count_HETDEX_df = create_imputation_count(imputed_HETDEX_df, mag_cols_for_colours, mag_cols_lim['5sigma'], 'num_imputed')

    if save_HETDEX_flag:
        print('Joining all tables')
        cat_final_non_imp_HETDEX_df = pd.concat([clean_cat_HETDEX_df, band_count_HETDEX_df, non_imputed_HETDEX_df], axis=1)
        clean_cat_final_HETDEX_df   = pd.concat([clean_cat_HETDEX_df, band_count_HETDEX_df, imputed_count_HETDEX_df, imputed_HETDEX_df], axis=1)
        # save new catalogue to a hdf5 file (.h5)
        print('Saving final table to file')
        cat_final_non_imp_HETDEX_df.to_hdf(gv.cat_path + gv.file_non_imp_HETDEX, key='df')
        clean_cat_final_HETDEX_df.to_hdf(gv.cat_path + gv.file_HETDEX, key='df')
        cat_final_non_imp_HETDEX_df.loc[:, mag_cols_for_colours].to_hdf(gv.preds_path + 'HETDEX_mags_non_imputed.h5', key='df')
        clean_cat_final_HETDEX_df.loc[:, mag_cols_for_colours].to_hdf(gv.preds_path + 'HETDEX_mags_imputed.h5', key='df')

#######

if run_S82_flag:
    print('-' * 40)
    print('Working with Stripe 82 data')
    print('Reading files')
    if run_S82_full:
        S82_initial_tab     = Table.read(gv.cat_path + gv.fits_S82, format='fits')
    if not run_S82_full:
        S82_initial_tab     = Table.read(gv.cat_path + gv.fits_S82_Ananna_17, format='fits')

    print('Fixing dtypes')
    S82_initial_tab     = fix_dtypes(S82_initial_tab)

    if run_S82_full:
        id_cols = ['objID', 'RA_ICRS', 'DE_ICRS', 'Name', 'RA_MILLI', 
                    'DEC_MILLI', 'TYPE', 'Z', 'zsp', 'spCl'] # zsp for Annana+17
    if not run_S82_full:
        id_cols = ['objID', 'RA_ICRS', 'DE_ICRS', 'Name', 'RA_MILLI', 
                'DEC_MILLI', 'TYPE', 'Z', 'zsp'] # zsp for Annana+17
    clean_cat_S82_df = S82_initial_tab[id_cols].to_pandas()

    zero_point_star_equiv  = u.zero_point_flux(3631.1 * u.Jy)  # zero point (AB) to Jansky

    print('Convert fluxes to magnitudes')
    # flx_cols = ['Total_flux_VLASS', 'Sint_LOFAR', 'Stotal_TGSS', 'FEP', 'Fint_VLAS82']
    mJy_cols_S82 = [col_name for col_name in S82_initial_tab.colnames if 
                S82_initial_tab[col_name].unit == 'mJy' and not 
                (col_name.startswith('e') or col_name.startswith('E')) and not 'rms' in col_name]

    for col in mJy_cols_S82:
        S82_initial_tab[col] = S82_initial_tab[col].to(u.mag(u.AB))

    # Transform Vega magnitudes to AB magnitudes
    print('Transforming Vega to AB')
    
    for col in vega_cols:
        S82_initial_tab[col] += vega_shift[col]

    # sys.exit()

    # Fix units of magnitudes for following steps
    for col in S82_initial_tab.colnames:
        if S82_initial_tab[col].unit == u.mag:
            S82_initial_tab[col].unit = u.mag(u.AB)
            S82_initial_tab[col]      = u.Magnitude(S82_initial_tab[col])

    # Select features to impute
    magnitude_cols = [col_name for col_name in S82_initial_tab.colnames if 
                S82_initial_tab[col_name].unit == u.mag(u.AB) and not 
                (col_name.startswith('e') or col_name.startswith('E') or col_name.endswith('MILLI')) and not
                     col_name.endswith('SDSS')]

    magnitude_cols_non_radio = [mag for mag in magnitude_cols if mag not in mJy_cols_S82]

    mags_S82_df      = S82_initial_tab[magnitude_cols].to_pandas()
    imputed_S82_df   = pd.DataFrame()

    # Create flags for X-ray and radio detection, and AGN classification
    print('Creating flags for X-ray and radio detections')
    # imputed_S82_df = create_X_ray_detect(imputed_S82_df, S82_initial_tab)
    radio_cols_S82 = ['Fint_VLAS82']
    imputed_S82_df = create_radio_detect(imputed_S82_df, S82_initial_tab, radio_cols_S82)

    # Select, from MQC, sources that have been classified 
    # as host-dominated NLAGN, AGN, or QSO candidates.
    print('Creating flags for AGN/Galaxy/Star classification')
    imputed_S82_df = create_AGN_gal_flags(S82_initial_tab, imputed_S82_df, AGN_types_list, gv.mqc_version)

    # Remove columns with too high numbe of missing values
    print('Removing columns with high nullity')
    removed_cols   = []
    limit_fraction = 1.00
    for col in magnitude_cols:
        filt_temp = np.isfinite(S82_initial_tab[col])
        removed   = 'NOT REMOVED'
        if np.sum(~filt_temp) > int(np.ceil(limit_fraction * len(S82_initial_tab[col]))):
            removed_cols.append(col)
            magnitude_cols.remove(col)
            removed = 'REMOVED'
            print(f'column: {col}\t -\t n_bad: {np.sum(~filt_temp)}\t{removed}')

    # Create counter of measurements per source (magnitudes)
    print('Creating new features:')
    print('Creating counter of valid measurements')
    band_count_S82_df = create_band_count(mags_S82_df, mag_cols_for_colours, 'band_num')

    # Impute values
    non_imputed_S82_df = imputed_S82_df.copy()
    print('Imputing values')
    
    for col in magnitude_cols:
        imputed_S82_df.loc[:, col] = mags_S82_df.loc[:, col].fillna(np.float32(mag_cols_lim['5sigma'][col]), inplace=False)
        imputed_S82_df.loc[:, col] = imputed_S82_df.loc[:, col].mask(imputed_S82_df.loc[:, col] >\
             mag_cols_lim['5sigma'][col], mag_cols_lim['5sigma'][col], inplace=False)
        non_imputed_S82_df.loc[:, col] = mags_S82_df.loc[:, col]

    # Create derived features
    print('Creating colours')
    imputed_S82_df     = create_colours(imputed_S82_df,     mag_cols_for_colours)
    non_imputed_S82_df = create_colours(non_imputed_S82_df, mag_cols_for_colours)

    print('Not creating magnitude ratios')
    # imputed_S82_df = create_ratios(imputed_S82_df)
    
    # Create counter of imputed measurements per source (magnitudes)
    print('Creating counter of valid measurements')
    imputed_count_S82_df = create_imputation_count(imputed_S82_df, mag_cols_for_colours, mag_cols_lim['5sigma'], 'num_imputed')

    if save_S82_flag:
        print('Joining all tables')
        cat_final_non_imp_S82_df = pd.concat([clean_cat_S82_df, band_count_S82_df, non_imputed_S82_df], axis=1)
        clean_cat_final_S82_df   = pd.concat([clean_cat_S82_df, band_count_S82_df, imputed_count_S82_df, imputed_S82_df], axis=1)
        # save new catalogue to a hdf5 file (.h5)
        print('Saving final table to file')
        if run_S82_full:
            cat_final_non_imp_S82_df.to_hdf(gv.cat_path + gv.file_non_imp_S82, key='df')
            clean_cat_final_S82_df.to_hdf(gv.cat_path + gv.file_S82, key='df')
            cat_final_non_imp_S82_df.loc[:, mag_cols_for_colours].to_hdf(gv.preds_path + 'S82_mags_non_imputed.h5', key='df')
            clean_cat_final_S82_df.loc[:, mag_cols_for_colours].to_hdf(gv.preds_path + 'S82_mags_imputed.h5', key='df')
        if not run_S82_full:
            clean_cat_final_S82_df.to_hdf(gv.cat_path + gv.file_S82_Ananna_17, key='df')

#######
# Run for COSMOS data
if run_COSMOS_flag:
    print('-' * 40)
    print('Working with COSMOS Field data')
    print('Reading files')
    COSMOS_initial_tab     = Table.read(gv.cat_path + gv.fits_COSMOS, format='fits')

    print('Fixing dtypes')
    COSMOS_initial_tab     = fix_dtypes(COSMOS_initial_tab)
    
    for col in ['objID', 'Name', 'TYPE']: # , 'COMMENT']:
        COSMOS_initial_tab[col].fill_value = ''
        COSMOS_initial_tab[col]   = COSMOS_initial_tab[col].astype(np.str)

    id_cols = ['objID', 'RA_ICRS', 'DE_ICRS', 'Name', 'RA_MILLI', 
                'DEC_MILLI', 'TYPE', 'Z', 'zsp', 'spCl'] # , 'COMMENT']
    clean_cat_COSMOS_df = COSMOS_initial_tab[id_cols].filled().to_pandas()
    
    # fix dtypes
    clean_cat_COSMOS_df.loc[:,['objID', 'Name', 'TYPE']] = clean_cat_COSMOS_df[['objID', 'Name', 'TYPE']].applymap(str)

    zero_point_star_equiv  = u.zero_point_flux(3631.1 * u.Jy)  # zero point (AB) to Jansky

    print('Convert fluxes to magnitudes')
    xray_freqs  = {'FEP':            1.51e+18 * u.Hz}
    xray_cols   = ['FEP']
    for col in xray_cols:
        COSMOS_initial_tab[col]       = COSMOS_initial_tab[col] / xray_freqs[col]
        # COSMOS_initial_tab[col].unit /= u.Hz
        COSMOS_initial_tab[col].unit  = u.mW * u.m**-2 * u.Hz**-1
        COSMOS_initial_tab[col]       = COSMOS_initial_tab[col].to(u.mJy)

    # flx_cols = ['Total_flux_VLASS', 'Sint_LOFAR', 'Stotal_TGSS', 'FEP', 'Fint_VLACOSMOS']
    mJy_cols_COSMOS = [col_name for col_name in COSMOS_initial_tab.colnames if 
                (COSMOS_initial_tab[col_name].unit == 'mJy' or COSMOS_initial_tab[col_name].unit == 'uJy') and not 
                (col_name.startswith('e') or col_name.startswith('E'))]

    for col in mJy_cols_COSMOS:
        COSMOS_initial_tab[col] = COSMOS_initial_tab[col].to(u.mag(u.AB))

    # Transform Vega magnitudes to AB magnitudes
    print('Transforming Vega to AB')
    
    for col in vega_cols:
        COSMOS_initial_tab[col] += vega_shift[col]

    # sys.exit()

    # Fix units of magnitudes for following steps
    for col in COSMOS_initial_tab.colnames:
        if COSMOS_initial_tab[col].unit == u.mag:
            COSMOS_initial_tab[col].unit = u.mag(u.AB)
            COSMOS_initial_tab[col]      = u.Magnitude(COSMOS_initial_tab[col])

    # Select features to impute
    magnitude_cols = [col_name for col_name in COSMOS_initial_tab.colnames if 
                COSMOS_initial_tab[col_name].unit == u.mag(u.AB) and not 
                (col_name.startswith('e') or col_name.startswith('E') or col_name.endswith('MILLI')) and not
                     col_name.endswith('SDSS')]

    magnitude_cols_non_radio = [mag for mag in magnitude_cols if mag not in mJy_cols_COSMOS]

    mags_COSMOS_df      = COSMOS_initial_tab[magnitude_cols].to_pandas()
    imputed_COSMOS_df   = pd.DataFrame()

    # Create flags for X-ray and radio detection, and AGN classification
    print('Creating flags for X-ray and radio detections')
    imputed_COSMOS_df = create_X_ray_detect(imputed_COSMOS_df, COSMOS_initial_tab)
    radio_cols_COSMOS = ['Flux_COSMOSVLA3', 'Stotal_TGSS', 'Total_flux_VLASS']
    imputed_COSMOS_df = create_radio_detect(imputed_COSMOS_df, COSMOS_initial_tab, radio_cols_COSMOS)

    # Select, from MQC, sources that have been classified 
    # as host-dominated NLAGN, AGN, or QSO candidates.
    print('Creating flags for AGN/Galaxy/Star classification')
    imputed_COSMOS_df = create_AGN_gal_flags(COSMOS_initial_tab, imputed_COSMOS_df, AGN_types_list, gv.mqc_version)

    # Remove columns with too high numbe of missing values
    print('Removing columns with high nullity')
    removed_cols   = []
    limit_fraction = 1.00
    for col in magnitude_cols:
        filt_temp = np.isfinite(COSMOS_initial_tab[col])
        removed   = 'NOT REMOVED'
        if np.sum(~filt_temp) > int(np.ceil(limit_fraction * len(COSMOS_initial_tab[col]))):
            removed_cols.append(col)
            magnitude_cols.remove(col)
            removed = 'REMOVED'
            print(f'column: {col}\t -\t n_bad: {np.sum(~filt_temp)}\t{removed}')

    # Create counter of measurements per source (magnitudes)
    print('Creating new features:')
    print('Creating counter of valid measurements')
    band_count_COSMOS_df = create_band_count(mags_COSMOS_df, mag_cols_for_colours, 'band_num')

    # Impute values
    non_imputed_COSMOS_df = imputed_COSMOS_df.copy()
    print('Imputing values')
    
    for col in magnitude_cols:
        imputed_COSMOS_df.loc[:, col] = mags_COSMOS_df.loc[:, col].fillna(np.float32(mag_cols_lim['5sigma'][col]), inplace=False)
        imputed_COSMOS_df.loc[:, col] = imputed_COSMOS_df.loc[:, col].mask(imputed_COSMOS_df.loc[:, col] >\
             mag_cols_lim['5sigma'][col], mag_cols_lim['5sigma'][col], inplace=False)
        non_imputed_COSMOS_df.loc[:, col] = mags_COSMOS_df.loc[:, col]

    # Create derived features
    print('Creating colours')
    imputed_COSMOS_df     = create_colours(imputed_COSMOS_df,     mag_cols_for_colours)
    non_imputed_COSMOS_df = create_colours(non_imputed_COSMOS_df, mag_cols_for_colours)

    print('Not creating magnitude ratios')
    # imputed_COSMOS_df = create_ratios(imputed_COSMOS_df)
    
    # Create counter of imputed measurements per source (magnitudes)
    print('Creating counter of valid measurements')
    imputed_count_COSMOS_df = create_imputation_count(imputed_COSMOS_df, mag_cols_for_colours, mag_cols_lim['5sigma'], 'num_imputed')

    if save_COSMOS_flag:
        print('Joining all tables')
        cat_final_non_imp_COSMOS_df = pd.concat([clean_cat_COSMOS_df, band_count_COSMOS_df, non_imputed_COSMOS_df], axis=1)
        clean_cat_final_COSMOS_df   = pd.concat([clean_cat_COSMOS_df, band_count_COSMOS_df, imputed_count_COSMOS_df, imputed_COSMOS_df], axis=1)
        # save new catalogue to a hdf5 file (.h5)
        print('Saving final table to file')
        cat_final_non_imp_COSMOS_df.to_hdf(gv.cat_path + gv.file_non_imp_COSMOS, key='df')
        clean_cat_final_COSMOS_df.to_hdf(gv.cat_path + gv.file_COSMOS, key='df')
