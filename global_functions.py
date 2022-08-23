#!/usr/bin/env python

# File with most used
# functions and derived
# variables.

# Initial imports
import numpy as np
import pandas as pd
import shap
import copy
import sklearn.pipeline as skp
from sklearn.metrics import ConfusionMatrixDisplay
from astropy.visualization import LogStretch, PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from pycaret import classification as pyc
from pycaret import regression as pyr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import global_variables as gv

##########################################
##########################################
# Define additional metrics for regression
def sigma_mad(z_true, z_pred, **kwargs):
    try:
        med = np.nanmedian(np.abs(z_true - z_pred)).astype('float32')
    except:
        med = np.nanmedian(np.abs(z_true - z_pred))
    return 1.48 * med

def sigma_nmad(z_true, z_pred, **kwargs):
    dif  = (z_true - z_pred)
    frac = dif / (1 + z_true).values
    try:
        med  = np.nanmedian(np.abs(frac)).astype('float32')
    except:
        med  = np.nanmedian(np.abs(frac))
    return 1.48 * med

def sigma_z(z_true, z_pred, **kwargs):
    dif = z_true - z_pred
    ssq = np.sum(dif**2)
    try:
        rot = np.sqrt(ssq / len(z_true)).astype('float32')
    except:
        rot = np.sqrt(ssq / len(z_true))
    return rot

def sigma_z_norm(z_true, z_pred, **kwargs):
    dif = (z_true - z_pred) / (1 + z_true)
    ssq = np.sum(dif**2)
    try:
        rot = np.sqrt(ssq / len(z_true)).astype('float32')
    except:
        rot = np.sqrt(ssq / len(z_true))
    return rot

def outlier_frac(z_true, z_pred, **kwargs):
    dif  = np.abs((z_true - z_pred) / (1 + z_true))
    try:
        siz  = np.sum(np.isfinite(dif)).astype('float32')
        num  = np.sum(np.array(dif > 0.15)).astype('float32')
    except:
        siz  = np.sum(np.isfinite(dif))
        num  = np.sum(np.array(dif > 0.15))
    frac = num / siz
    return frac

##########################################
# Obtain classification metrics from confusion matrices
def conf_mat_func(true_class_arr, predicted_class_arr):
    cm = np.array([[np.sum(np.array(true_class_arr == 0) & np.array(predicted_class_arr == 0)),\
                    np.sum(np.array(true_class_arr == 0) & np.array(predicted_class_arr == 1))],\
                   [np.sum(np.array(true_class_arr == 1) & np.array(predicted_class_arr == 0)),\
                    np.sum(np.array(true_class_arr == 1) & np.array(predicted_class_arr == 1))]])
    return cm

def flatten_CM(cm_array):
    try:
        TN, FP, FN, TP = cm_array.flatten().astype('float32')
    except:
        TN, FP, FN, TP = cm_array.flatten()
    return TN, FP, FN, TP

# Matthews correlation coefficient
def MCC_from_CM(cm_array):
    TN, FP, FN, TP = flatten_CM(cm_array)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return MCC

# Accuracy
def ACC_from_CM(cm_array):
    TN, FP, FN, TP = flatten_CM(cm_array)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    return ACC

# F-1 score
def F1_from_CM(cm_array):
    _, FP, FN, TP = flatten_CM(cm_array)
    F1 = 2 * TP / (2 * TP + FP + FN)
    return F1

# Recall
def Recall_from_CM(cm_array):
    TN, FP, FN, TP = flatten_CM(cm_array)
    Recall = TP / (TP + FN)
    return Recall

##########################################
# Methods using Pycaret pipelines
def get_final_column_names(pycaret_pipeline, sample_df, verbose=False):
    if isinstance(pycaret_pipeline, skp.Pipeline):
        for (name, method) in pycaret_pipeline.named_steps.items():
            if method != 'passthrough' and name != 'trained_model':
                if verbose:
                    print(f'Running {name}')
                sample_df = method.transform(sample_df)
        return sample_df.columns.tolist()
    else:
        try:
            for (name, method) in pyr.get_config('prep_pipe').named_steps.items():
                if method != 'passthrough' and name != 'trained_model':
                    if verbose:
                        print(f'Running {name}')
                    sample_df = method.transform(sample_df)
        except:
            for (name, method) in pyc.get_config('prep_pipe').named_steps.items():
                if method != 'passthrough' and name != 'trained_model':
                    if verbose:
                        print(f'Running {name}')
                    sample_df = method.transform(sample_df)
        return sample_df.columns.tolist()

# Feature importance (or mean of) from meta model (or base models)
def get_feature_importances_df(pycaret_pipeline, sample_df, n=10):
    
    final_cols = get_final_column_names(pycaret_pipeline, sample_df)
    
    if isinstance(pycaret_pipeline, skp.Pipeline):
        try:
            variables = pycaret_pipeline["trained_model"].feature_importances_
            
        except:
            variables = np.mean([
                            tree.feature_importances_ for tree in pycaret_pipeline["trained_model"].estimators_
                if hasattr(tree, 'feature_importances_')
                            ], axis=0)
        
        coef_df = pd.DataFrame({'Feature': final_cols, 'Importance': variables})
        sorted_df = (
            coef_df.sort_values(by='Importance', ascending=False)
            .head(n)
            .sort_values(by='Importance', ascending=True).reset_index(drop=True)
        )
    else:
        try:
            variables = pycaret_pipeline.feature_importances_
            
        except:
            variables = np.mean([
                            tree.feature_importances_ for tree in pycaret_pipeline.estimators_
                if hasattr(tree, 'feature_importances_')
                            ], axis=0)
        
        coef_df = pd.DataFrame({'Feature': final_cols, 'Importance': variables})
        sorted_df = (
            coef_df.sort_values(by='Importance', ascending=False)
            .head(n)
            .sort_values(by='Importance', ascending=True).reset_index(drop=True)
        )
    return sorted_df

def get_base_estimators_names(pycaret_pipeline):
    if isinstance(pycaret_pipeline, skp.Pipeline):
        estimators  = pycaret_pipeline['trained_model'].estimators
    else:
        estimators  = pycaret_pipeline.estimators

    estimators_list = [estimator[0] for estimator in estimators]
    return estimators_list

def get_base_estimators_models(pycaret_pipeline):
    if isinstance(pycaret_pipeline, skp.Pipeline):
        estimators_  = pycaret_pipeline['trained_model'].estimators_
    else:
        estimators_  = pycaret_pipeline.estimators_
    return estimators_

# Run data through previous steps of pipeline
def preprocess_data(pycaret_pipeline, data_df, base_models_names, verbose=False):
    processed_data = data_df.loc[:, get_final_column_names(pycaret_pipeline, data_df)].copy()
    processed_idx_data  = processed_data.index
    # processed_cols_data  = processed_data.columns
    processed_cols_data = processed_data.columns.insert(0, base_models_names[0])
    if len(base_models_names) > 1:
        for est_name in base_models_names[1::]:
            processed_cols_data = processed_cols_data.insert(0, est_name)
    if isinstance(pycaret_pipeline, skp.Pipeline):
        prep_steps = pycaret_pipeline.named_steps.items()
    else:
        prep_steps = pyc.get_config('prep_pipe').named_steps.items()

    for (name, method) in prep_steps:
        if method != 'passthrough':  # and name != 'trained_model':
            if verbose:
                print(f'Running {name}')
            processed_data = method.transform(processed_data)
    processed_data_df = pd.DataFrame(processed_data, columns=processed_cols_data, index=processed_idx_data)
    return processed_data_df

# Sorted feature importances
def feat_importances_base_models(base_models_names, base_models, transformed_data_df):
    coef_sorted_base_df = {}
    feat_names = transformed_data_df.columns.drop(base_models_names)
    for model, model_fit in zip(base_models_names, base_models):
        if hasattr(model_fit, 'feature_importances_'):
            coef_base_df = pd.DataFrame({'Feature': feat_names,
                                         'Importance': model_fit.feature_importances_})
            coef_sorted_base_df[model] = (
            coef_base_df.sort_values(by='Importance', ascending=False)
            .head(len(feat_names))
            .sort_values(by='Importance', ascending=False).reset_index(drop=True)
            )
        elif hasattr(model_fit, 'coef_'):
            coef_base_df = pd.DataFrame({'Feature': feat_names,
                                         'Importance': np.abs(model_fit.coef_.ravel()) *\
                                         transformed_data_df.loc[:, feat_names].std(axis=0)})
            coef_sorted_base_df[model] = (
            coef_base_df.sort_values(by='Importance', ascending=False)
            .head(len(feat_names))
            .sort_values(by='Importance', ascending=False).reset_index(drop=True)
            )
    return coef_sorted_base_df

# Sorted feature importances
def feat_importances_meta_model(pycaret_pipeline, transformed_data_df):
    extended_cols_data = transformed_data_df.columns
    if isinstance(pycaret_pipeline, skp.Pipeline):
        if hasattr(pycaret_pipeline.named_steps['trained_model'].final_estimator_, 'feature_importances_'):
            importances_coef = pycaret_pipeline.named_steps['trained_model'].final_estimator_.feature_importances_
        elif hasattr(pycaret_pipeline.named_steps['trained_model'].final_estimator_, 'coef_'):
            importances_coef = np.abs(np.ravel(pycaret_pipeline.named_steps['trained_model'].final_estimator_.coef_)) *\
                                         transformed_data_df.loc[:, extended_cols_data].std(axis=0)
    else:
        if hasattr(pycaret_pipeline.final_estimator_, 'feature_importances_'):
            importances_coef = pycaret_pipeline.final_estimator_.feature_importances_
        elif hasattr(pycaret_pipeline.final_estimator_, 'coef_'):
            importances_coef = np.abs(np.ravel(pycaret_pipeline.final_estimator_.coef_)) *\
                                         transformed_data_df.loc[:, extended_cols_data].std(axis=0)

    coef_meta_df = pd.DataFrame({'Feature': extended_cols_data, 'Importance': importances_coef})
    coef_sorted_meta_df = (
        coef_meta_df.sort_values(by='Importance', ascending=False)
        .head(len(extended_cols_data))
        .sort_values(by='Importance', ascending=False).reset_index(drop=True)
    )
    return coef_sorted_meta_df


##########################################
# Plotting methods

# Path effects for labels and plots.
pe1            = [mpe.Stroke(linewidth=5.0, foreground='black'),
                  mpe.Stroke(foreground='white', alpha=1),
                  mpe.Normal()]
pe2            = [mpe.Stroke(linewidth=3.0, foreground='white'),
                  mpe.Stroke(foreground='white', alpha=1),
                  mpe.Normal()]

# Create class to normalize asymmetric colorscales  
# (from http://chris35wills.github.io/matplotlib_diverging_colorbar/).
class MidpointNormalize(mcolors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# Plot confusion matrix
def plot_conf_mat(confusion_matrix, title, axin, display_labels=['0', '1'], cmap=gv.cmap_conf_matr, show_clb=False, log_stretch=False):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)

    if log_stretch:
        norm = ImageNormalize(stretch=LogStretch())
    if not log_stretch:
        norm = ImageNormalize(stretch=PowerStretch(0.35))

    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp_b = disp.plot(include_values=True, cmap=cm.get_cmap(cmap),\
             ax=axin, xticks_rotation='horizontal', values_format=',')

    for text_val in disp_b.text_.flatten():
        text_val.set_fontsize(28)
    clb = plt.gca().images[-1].colorbar
    clb.ax.tick_params(labelsize=14)
    clb.ax.ticklabel_format(style='sci', scilimits=(0, 0))
    clb.outline.set_linewidth(2.5)
    clb.ax.set_ylabel('Elements in bin', size=14)
    if not show_clb:
        clb.remove()

    disp_b.im_.norm = norm

    axin.xaxis.get_label().set_fontsize(16)
    axin.yaxis.get_label().set_fontsize(16)

    axin.tick_params(axis='both', which='major', labelsize=14)

    plt.setp(axin.spines.values(), linewidth=2.5)
    plt.setp(axin.spines.values(), linewidth=2.5)
    axin.set_title(title, fontsize=16)
    plt.tight_layout()
    return axin

# Plot true and estimated/predicted redshifts
def plot_redshift_compare(true_z, predicted_z, ax_pre, title=None, dpi=10, cmap=gv.cmap_z_plots, show_clb=False, log_stretch=False):
    if log_stretch:
        norm = ImageNormalize(vmin=0., stretch=LogStretch())
    if not log_stretch:
        norm = ImageNormalize(vmin=0., stretch=PowerStretch(0.5))

    filt_pair_z   = np.isfinite(true_z) & np.isfinite(predicted_z)
    max_for_range = np.nanmax([np.nanmax(1 + true_z.loc[filt_pair_z]), np.nanmax(1 + predicted_z.loc[filt_pair_z])])

    dens_1 = ax_pre.scatter_density((1 + true_z.sample(frac=1, random_state=gv.seed)),\
            (1 + predicted_z.sample(frac=1, random_state=gv.seed)),\
            cmap=plt.get_cmap(cmap), zorder=0, dpi=dpi, norm=norm, alpha=0.93)
    
    ax_pre.axline((2., 2.), (3., 3.), ls='--', marker=None, c='Gray', alpha=0.8, lw=3.0, zorder=20)
    ax_pre.axline(xy1=(1., 1.15), xy2=(2., 2.3), ls='-.', marker=None, c='slateblue', alpha=0.6, lw=3.0, zorder=20)
    ax_pre.axline(xy1=(1., 0.85), xy2=(2., 1.7), ls='-.', marker=None, c='slateblue', alpha=0.6, lw=3.0, zorder=20)

    if show_clb:
        clb = plt.colorbar(dens_1, extend='neither', norm=norm, ticks=mtick.MaxNLocator(integer=True))
        clb.ax.tick_params(labelsize=14)
        clb.outline.set_linewidth(2.5)
        clb.ax.set_ylabel('Elements per pixel', size=16, path_effects=pe2)

    # Inset axis with residuals
    axins = inset_axes(ax_pre, width='35%', height='20%', loc=2)
    res_z_z = (predicted_z - true_z) / (1 + true_z)
    axins.hist(res_z_z, histtype='stepfilled', fc='grey', ec='k', bins=50, lw=2.5)
    axins.axvline(x=np.nanpercentile(res_z_z, [15.9]), ls='--', lw=2.5, c='royalblue')
    axins.axvline(x=np.nanpercentile(res_z_z, [84.1]), ls='--', lw=2.5, c='royalblue')
    axins.set_xlabel('$\Delta Z / (1 + Z_{\mathrm{True}})$', fontsize=10)
    axins.tick_params(labelleft=False, labelbottom=True)
    axins.tick_params(which='both', top=True, right=True, direction='in')
    axins.tick_params(axis='both', which='major', labelsize=10)
    axins.tick_params(which='major', length=8, width=1.5)
    axins.tick_params(which='minor', length=4, width=1.5)
    plt.setp(axins.spines.values(), linewidth=2.5)
    plt.setp(axins.spines.values(), linewidth=2.5)
    axins.set_xlim(left=-0.9, right=0.9)
    ##
    ax_pre.set_xlabel('$1 + Z_{\mathrm{True}}$', fontsize=20)
    ax_pre.set_ylabel('$1 + Z_{\mathrm{Predicted}}$', fontsize=20)
    ax_pre.tick_params(which='both', top=True, right=True, direction='in')
    ax_pre.tick_params(axis='both', which='minor', labelsize=14)
    ax_pre.tick_params(which='major', length=8, width=1.5)
    ax_pre.tick_params(which='minor', length=4, width=1.5)
    # ax_pre.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    # ax_pre.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax_pre.xaxis.set_minor_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.yaxis.set_minor_formatter(mtick.ScalarFormatter(useMathText=False))
    plt.setp(ax_pre.spines.values(), linewidth=2.5)
    plt.setp(ax_pre.spines.values(), linewidth=2.5)
    ax_pre.set_xlim(left=1., right=np.ceil(max_for_range))
    ax_pre.set_ylim(bottom=1., top=np.ceil(max_for_range))
    ax_pre.set_title(title)
    plt.tight_layout()
    return ax_pre

# Plot SHAP beeswarm
def plot_shap_beeswarm(pred_type, model_name, shap_values, cmap=gv.cmap_shap, ax_factor=0.75, base_meta=''):
    if np.ndim(shap_values.values) == 2:
        shap.plots.beeswarm(copy.deepcopy(shap_values), log_scale=False, show=False, color_bar=False,
                            color=plt.get_cmap(cmap), max_display=len(shap_values.feature_names), alpha=1.0)
    elif np.ndim(shap_values.values) > 2:
        shap.plots.beeswarm(copy.deepcopy(shap_values)[:, :, 1], log_scale=False, show=False, color_bar=False,
                            color=plt.get_cmap(cmap), max_display=len(shap_values.feature_names), alpha=1.0)
    _, h = plt.gcf().get_size_inches()
    m  = cm.ScalarMappable(cmap=cmap)
    cb = plt.colorbar(m, ticks=[0, 1], aspect=100)
    cb.set_ticklabels([shap.plots._labels.labels['FEATURE_VALUE_LOW'], shap.plots._labels.labels['FEATURE_VALUE_HIGH']])
    cb.set_label(shap.plots._labels.labels["FEATURE_VALUE"], size=16, labelpad=-20)
    cb.ax.tick_params(labelsize=16, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    plt.gca().tick_params('x', labelsize=14)
    plt.gca().xaxis.get_offset_text().set_fontsize(14)
    plt.gca().xaxis.get_offset_text().set_position((0,1))
    plt.gca().tick_params('y', labelsize=20)
    plt.gca().xaxis.label.set_size(20)
    plt.title(f'{pred_type}: {base_meta}-learner - {model_name}', fontsize=16)
    plt.gcf().set_size_inches(h * ax_factor, h * ax_factor *3/2)
    plt.tight_layout()

# Plot SHAP decision
def plot_shap_decision(pred_type, model_name, shap_values, shap_explainer, col_names, ax, link, cmap=gv.cmap_shap, new_base_value=None, base_meta='', xlim=None):
    if np.ndim(shap_values.values) == 2:
        shap.plots.decision(base_value=shap_explainer.expected_value,
                            shap_values=shap_values.values,
                            feature_names=col_names.to_list(),
                            link=link, plot_color=plt.get_cmap(cmap),
                            highlight=None, auto_size_plot=False,
                            show=False, xlim=xlim,
                            feature_display_range=slice(-1, -(len(shap_values.feature_names) +1), -1),
                            new_base_value=new_base_value)
    if np.ndim(shap_values.values) > 2:
        shap.plots.decision(base_value=shap_explainer.expected_value[-1],
                            shap_values=(shap_values.values)[:, :, 1],
                            feature_names=col_names.to_list(),
                            link=link, plot_color=plt.get_cmap(cmap),
                            highlight=None, auto_size_plot=False,
                            show=False, xlim=None,
                            feature_display_range=slice(-1, -(len(shap_values.feature_names) +1), -1),
                            new_base_value=new_base_value)
    ax.tick_params('x', labelsize=14)
    ax.xaxis.get_offset_text().set_fontsize(14)
    #ax1.xaxis.get_offset_text().set_position((0,1))
    ax.tick_params('y', labelsize=20)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.xaxis.label.set_size(20)
    plt.title(f'{pred_type}: {base_meta}-learner - {model_name}', fontsize=16)
    plt.tight_layout()
    return ax