import os
import re
import json
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.stats import nbinom

import utils as ut

cwd_path = os.getcwd()

prevalence_cutoff = 0.001

def plot_panel_homogeneous_thresholds_heatmaps(
        model_region='National', 
        model_opinion='HOT',
        model_hesitancy='RAN',
        target_vaccination=None,
        target_zealot=0.0, 
        path_cwd=os.getcwd(),
        path_relative_source='results/curated',
        cutoff_prevalence = 0.05,
        ):

    path_full_source = os.path.join(path_cwd, path_relative_source)
    list_filenames = ut.collect_filenames(
        path_search=path_full_source, 
        header='global__', 
        string_segments=[model_region, model_opinion, model_hesitancy], 
        extension='.json',
        )

    dict_results = {}

    for i, filename in enumerate(list_filenames):
        fullname = os.path.join(path_cwd, path_relative_source, filename)

        size = int(filename.split('_n')[1].split('_')[0])
        fa = float(filename.split('_fa')[1].split('_')[0])
        th = float(filename.split('_th')[1].split('_')[0])
        rv = float(filename.split('_rv')[1].split('_')[0])
        fz = float(filename.split('_fz')[1].split('_')[0])

        if fz == target_zealot and rv in target_vaccination:
            with open(fullname, 'rb') as input_data:
                dict_output = pk.load(input_data)

            key_pars = (fa, th, rv, fz)

            if key_pars not in dict_results:
                dict_results[key_pars] = {'convinced': [], 'prevalence': [], 'vaccinated': []}

            array_convinced = dict_output['global']['convinced']
            array_prevalence = dict_output['global']['prevalence']
            array_vaccinated = dict_output['global']['vaccinated']

            for idx, prevalence in enumerate(array_prevalence):
                if (prevalence / size) >= cutoff_prevalence:
                    dict_results[key_pars]['convinced'].append(array_convinced[idx] / size)
                    dict_results[key_pars]['prevalence'].append(array_prevalence[idx] / size)
                    dict_results[key_pars]['vaccinated'].append(array_vaccinated[idx] / size)

    dict_stats = {}
    for key_pars, results in dict_results.items():
        dict_stats[key_pars] = {
            'convinced': {
                'mean': np.nanmean(results['convinced']) if results['convinced'] else float('nan'),
                'std': np.nanstd(results['convinced']) if results['convinced'] else float('nan'),
                },
            'prevalence': {
                'mean': np.nanmean(results['prevalence']) if results['prevalence'] else float('nan'),
                'std': np.nanstd(results['prevalence']) if results['prevalence'] else float('nan'),
                },
            'vaccinated': {
                'mean': np.nanmean(results['vaccinated']) if results['vaccinated'] else float('nan'),
                'std': np.nanstd(results['vaccinated']) if results['vaccinated'] else float('nan'),
            }
        }

    fa_values = set()
    th_values = set()
    rv_values = set()

    for key in dict_results.keys():
        fa, th, rv, fz = key
        fa_values.add(fa)
        th_values.add(th)
        rv_values.add(rv)

    array_control_fa = np.array(sorted(fa_values))
    array_control_th = np.array(sorted(th_values))
    array_control_rv = np.array(sorted(rv_values))

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18, 20))
    cmap = 'viridis'

    for v, rv in enumerate(array_control_rv):
        len_th = len(array_control_th)
        len_fa = len(array_control_fa)

        X, Y = np.meshgrid(array_control_th, array_control_fa)
        R = np.zeros((len_fa, len_th))
        V = np.zeros((len_fa, len_th))
        C = np.zeros((len_fa, len_th))

        for i, th in enumerate(array_control_th):
            for j, fa in enumerate(array_control_fa):
                if (th, fa, rv, fz) in dict_stats:

                    R[j][i] = dict_stats[(fa, th, rv, fz)]['prevalence']['mean']
                    V[j][i] = dict_stats[(fa, th, rv, fz)]['vaccinated']['mean']

                    max_change = 1.0 - fa
                    norm_change = (dict_stats[(fa, th, rv, fz)]['convinced']['mean'] - fa) / max_change
                    if fa == 1.0:
                        norm_change = 0.0
                    C[j][i] = norm_change
                else:
                    R[j][i] = np.nan
                    V[j][i] = np.nan
                    C[j][i] = np.nan

        im2 = ax[v, 2].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=0.6)
        im1 = ax[v, 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
        im0 = ax[v, 0].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
        
        contour_levels = [0.0195]
        if v == 2:
            contour_levels = [0.07]
        if v == 3:
            contour_levels = [0.05]
        ax[v, 2].contour(X, Y, R, contour_levels, colors='white', linestyles='dashed')
        #ax[v, 1].contour(X, Y, V, contour_levels, colors='white', linestyles='dashed')
        contour_levels = [0.1]
        ax[v, 0].contour(X, Y, C, levels=contour_levels, colors='white', linestyles='dashed')
        
        cb2 = plt.colorbar(im2, ax=ax[v, 2])
        cb2.ax.tick_params(labelsize=18)
        cb1 = plt.colorbar(im1, ax=ax[v, 1])
        cb1.ax.tick_params(labelsize=18)
        cb0 = plt.colorbar(im0, ax=ax[v, 0])
        cb0.ax.tick_params(labelsize=18)

    ax[0, 2].set_title(r"$r(\infty)$", fontsize=35)
    ax[0, 1].set_title(r"$v(\infty)$", fontsize=35)
    ax[0, 0].set_title(r"$\Delta n_A(\infty)$", fontsize=35)

    ax[0, 0].text(-0.35, 0.15, r'$\alpha=0.001$', fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.35, 0.12, r'$\alpha=0.0025$', fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')
    ax[2, 0].text(-0.35, 0.15, r'$\alpha=0.005$', fontsize=30, transform=ax[2, 0].transAxes, rotation='vertical')
    ax[3, 0].text(-0.35, 0.2, r'$\alpha=0.01$', fontsize=30, transform=ax[3, 0].transAxes, rotation='vertical')
    
    ax[3, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[3, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[3, 2].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[3, 0].set_ylabel(r'$n_A(0)$', fontsize=30)

    ax[0, 0].text(0.04, 0.8, r"a1", transform=ax[0, 0].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 0].text(0.04, 0.8, r"a2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 0].text(0.04, 0.8, r"a3", transform=ax[2, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[3, 0].text(0.04, 0.8, r"a4", transform=ax[3, 0].transAxes, fontsize=30, color='black', weight="bold")
    
    ax[0, 1].text(0.04, 0.8, r"b1", transform=ax[0, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 1].text(0.04, 0.8, r"b2", transform=ax[1, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[2, 1].text(0.04, 0.8, r"b3", transform=ax[2, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[3, 1].text(0.04, 0.8, r"b4", transform=ax[3, 1].transAxes, fontsize=30, color='white', weight="bold")

    ax[0, 2].text(0.04, 0.8, r"c1", transform=ax[0, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 2].text(0.04, 0.8, r"c2", transform=ax[1, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[2, 2].text(0.04, 0.8, r"c3", transform=ax[2, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[3, 2].text(0.04, 0.8, r"c4", transform=ax[3, 2].transAxes, fontsize=30, color='white', weight="bold")

    for axis in ax.flatten():
        axis.tick_params(axis='both', labelsize=18)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    path_full_target = os.path.join(path_cwd, 'figures')
    filename_target = 'threshold_main_homogeneous_heatmap' 
    extension_list = ['pdf', 'png']
    if not os.path.exists(path_full_target):
        os.makedirs(path_full_target)
    for ext in extension_list:
        full_name = os.path.join(path_full_target, filename_target + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_panel_homogeneous_thresholds_sections(
        model_region='National', 
        model_opinion='HOT',
        model_hesitancy='RAN',
        target_vaccination=None,
        target_zealot=0.0, 
        path_cwd=os.getcwd(),
        path_relative_source='results/curated',
        cutoff_prevalence = 0.05,
        ):

    path_full_source = os.path.join(path_cwd, path_relative_source)
    list_filenames = ut.collect_filenames(
        path_search=path_full_source, 
        header='global__', 
        string_segments=[model_region, model_opinion, model_hesitancy], 
        extension='.json',
        )

    dict_results = {}

    for i, filename in enumerate(list_filenames):
        fullname = os.path.join(path_cwd, path_relative_source, filename)

        size = int(filename.split('_n')[1].split('_')[0])
        fa = float(filename.split('_fa')[1].split('_')[0])
        th = float(filename.split('_th')[1].split('_')[0])
        rv = float(filename.split('_rv')[1].split('_')[0])
        fz = float(filename.split('_fz')[1].split('_')[0])

        if fz == target_zealot and rv in target_vaccination:
            with open(fullname, 'rb') as input_data:
                dict_output = pk.load(input_data)

            key_pars = (fa, th, rv, fz)

            if key_pars not in dict_results:
                dict_results[key_pars] = {'convinced': [], 'prevalence': [], 'vaccinated': []}

            array_convinced = dict_output['global']['convinced']
            array_prevalence = dict_output['global']['prevalence']
            array_vaccinated = dict_output['global']['vaccinated']

            for idx, prevalence in enumerate(array_prevalence):
                if (prevalence / size) >= cutoff_prevalence:
                    dict_results[key_pars]['convinced'].append(array_convinced[idx] / size)
                    dict_results[key_pars]['prevalence'].append(array_prevalence[idx] / size)
                    dict_results[key_pars]['vaccinated'].append(array_vaccinated[idx] / size)

    dict_stats = {}
    for key_pars, results in dict_results.items():
        dict_stats[key_pars] = {
            'convinced': {
                'mean': np.nanmean(results['convinced']) if results['convinced'] else float('nan'),
                'std': np.nanstd(results['convinced']) if results['convinced'] else float('nan'),
                },
            'prevalence': {
                'mean': np.nanmean(results['prevalence']) if results['prevalence'] else float('nan'),
                'std': np.nanstd(results['prevalence']) if results['prevalence'] else float('nan'),
                },
            'vaccinated': {
                'mean': np.nanmean(results['vaccinated']) if results['vaccinated'] else float('nan'),
                'std': np.nanstd(results['vaccinated']) if results['vaccinated'] else float('nan'),
            }
        }

    fa_values = set()
    th_values = set()
    rv_values = set()

    for key in dict_results.keys():
        fa, th, rv, fz = key
        fa_values.add(fa)
        th_values.add(th)
        rv_values.add(rv)

    array_control_fa = np.array(sorted(fa_values))
    array_control_th = np.array(sorted(th_values))
    array_control_rv = np.array(sorted(rv_values))

    norm = Normalize(vmin=min(array_control_fa), vmax=max(array_control_fa))
    cmap = cm.viridis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 18))

    for v, rv in enumerate(array_control_rv):
        len_th = len(array_control_th)
        len_fa = len(array_control_fa)

        C = np.zeros((len_fa, len_th))
        R = np.zeros((len_fa, len_th))
        V = np.zeros((len_fa, len_th))

        for j, fa in enumerate(array_control_fa):
            for i, th in enumerate(array_control_th):
                if (th, fa, rv, fz) in dict_stats:

                    R[j][i] = dict_stats[(fa, th, rv, fz)]['prevalence']['mean']
                    V[j][i] = dict_stats[(fa, th, rv, fz)]['vaccinated']['mean']

                    max_change = 1.0 - fa
                    norm_change = (dict_stats[(fa, th, rv, fz)]['convinced']['mean'] - fa) / max_change
                    if fa == 1.0:
                        norm_change = 0.0
                    C[j][i] = norm_change
                else:
                    R[j][i] = np.nan
                    V[j][i] = np.nan
                    C[j][i] = np.nan

            ax[0, v].plot(array_control_th, C[j], color=sm.to_rgba(fa))
            ax[1, v].plot(array_control_th, V[j], color=sm.to_rgba(fa))
            ax[2, v].plot(array_control_th, R[j], color=sm.to_rgba(fa))

        ax[0, v].set_title(r'$\alpha=${0}'.format(rv), fontsize=40)

    for i in range(3):
        subplot_bottom = ax[i, 0].get_position().y0
        subplot_height = ax[i, 0].get_position().height
    
        cbar_ax = fig.add_axes([0.99, subplot_bottom, 0.01, subplot_height])
        #cbar_ax = fig.add_axes([0.99, 0.7 - i*0.3, 0.01, 0.3])
        fig.colorbar(sm, cax=cbar_ax, label=r'$n_A(0)$')

    ax[0, 0].set_ylabel(r"$\Delta n_A(\infty)$", fontsize=35)
    ax[1, 0].set_ylabel(r"$v(\infty)$", fontsize=35)
    ax[2, 0].set_ylabel(r"$r(\infty)$", fontsize=35)

    ax[-1, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[-1, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[-1, 2].set_xlabel(r'$\theta$', fontsize=30)
    ax[-1, 3].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].text(0.04, 0.8, r"a1", transform=ax[0, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 1].text(0.04, 0.8, r"a2", transform=ax[0, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 2].text(0.04, 0.8, r"a3", transform=ax[0, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 3].text(0.04, 0.8, r"a4", transform=ax[0, 3].transAxes, fontsize=30, color='black', weight="bold")
    
    ax[1, 0].text(0.04, 0.8, r"b1", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 1].text(0.04, 0.8, r"b2", transform=ax[1, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 2].text(0.04, 0.8, r"b3", transform=ax[1, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 3].text(0.04, 0.8, r"b4", transform=ax[1, 3].transAxes, fontsize=30, color='black', weight="bold")

    ax[2, 0].text(0.04, 0.8, r"c1", transform=ax[2, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 1].text(0.04, 0.8, r"c2", transform=ax[2, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 2].text(0.04, 0.8, r"c3", transform=ax[2, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 3].text(0.04, 0.8, r"c4", transform=ax[2, 3].transAxes, fontsize=30, color='black', weight="bold")

    for axis in ax.flatten():
        axis.tick_params(axis='both', labelsize=18)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    path_full_target = os.path.join(path_cwd, 'figures')
    filename_target = 'threshold_main_homogeneous_sections' 
    extension_list = ['pdf', 'png']
    if not os.path.exists(path_full_target):
        os.makedirs(path_full_target)
    for ext in extension_list:
        full_name = os.path.join(path_full_target, filename_target + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_panel_survey_thresholds_vaccination_curves(
        model_opinion='DataDriven',
        path_cwd=os.getcwd(),
        path_relative_source='results/curated',
        cutoff_prevalence=0.01,
        ):

    path_full_source = os.path.join(path_cwd, path_relative_source)
    list_filenames = ut.collect_filenames(
        path_search=path_full_source, 
        header='global__', 
        string_segments=[model_opinion], 
        extension='.json',
        )

    dict_results = {}
    
    dict_state_attitude = ut.load_vaccination_data(path_cwd)

    for i, filename in enumerate(list_filenames):
        fullname = os.path.join(path_cwd, path_relative_source, filename)

        model_region = filename.split('ml')[1].split('_')[0]
        size = int(filename.split('_n')[1].split('_')[0])
        rv = float(filename.split('_rv')[1].split('_')[0])
        
        with open(fullname, 'rb') as input_data:
            dict_output = pk.load(input_data)

            if model_region not in dict_results:
                dict_results[model_region] = {}

            if rv not in dict_results[model_region]:
                dict_results[model_region][rv] = {'convinced': [], 'prevalence': [], 'vaccinated': []}
        
            array_convinced = dict_output['global']['convinced']
            array_prevalence = dict_output['global']['prevalence']
            array_vaccinated = dict_output['global']['vaccinated']

            for idx, prevalence in enumerate(array_prevalence):
                if (prevalence / size) >= cutoff_prevalence:
                    dict_results[model_region][rv]['convinced'].append(array_convinced[idx] / size)
                    dict_results[model_region][rv]['prevalence'].append(array_prevalence[idx] / size)
                    dict_results[model_region][rv]['vaccinated'].append(array_vaccinated[idx] / size)

    dict_stats = {}
    for model_region, rvs in dict_results.items():
        dict_stats[model_region] = {}
        for rv, state_results in rvs.items():
            convinced_data = state_results['convinced']
            prevalence_data = state_results['prevalence']
            vaccinated_data = state_results['vaccinated']

            def calc_stats(data):
                if not data: 
                    return {'avg': float('nan'), 'std': float('nan'), 'l95': float('nan'), 'u95': float('nan')}
                mean = np.nanmean(data)
                std = np.nanstd(data)
                n = len(data)
                sem = std / np.sqrt(n)
                l95 = mean - 1.96 * sem
                u95 = mean + 1.96 * sem
                return {'avg': mean, 'std': std, 'l95': l95, 'u95': u95}

            dict_stats[model_region][rv] = {
                'convinced': calc_stats(convinced_data),
                'prevalence': calc_stats(prevalence_data),
                'vaccinated': calc_stats(vaccinated_data),
            }

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))

    vmin = 0.25
    vmax = 0.65
    norm = Normalize(vmin=vmin, vmax=vmax)

    for _, model_region in enumerate(dict_stats.keys()):
        array_rv = sorted(dict_stats[model_region].keys())

        stats_values = [dict_stats[model_region][rv] for rv in array_rv]

        conv_avg = np.array([obs['convinced']['avg'] if 'convinced' in obs and 'avg' in obs['convinced'] else np.nan for obs in list(stats_values)])
        conv_l95 = np.array([obs['convinced']['l95'] if 'convinced' in obs and 'l95' in obs['convinced'] else np.nan for obs in list(stats_values)])
        conv_u95 = np.array([obs['convinced']['u95'] if 'convinced' in obs and 'u95' in obs['convinced'] else np.nan for obs in list(stats_values)])

        vac_shares = dict_state_attitude[model_region]
        if 'Underage' in model_opinion or 'UA' in model_opinion:
            array_population = ut.import_age_distribution(state=model_region, reference=False, year=2019, path=path_cwd)
            fraction_underage = ut.count_fraction_underage(array_population)
            fraction_eligible = 1.0 - fraction_underage
        else:
            fraction_eligible = 1.0

        already = fraction_eligible * vac_shares['already']
        soon = fraction_eligible * vac_shares['soon']
        initial_support = already + soon
        zealots = fraction_eligible * vac_shares['never']

        max_change = 1.0 - (already + soon + zealots)
        delta_con_avg = [(con - initial_support) / max_change for con in conv_avg]
        delta_con_l95 = [(con - initial_support) / max_change for con in conv_l95]
        delta_con_u95 = [(con - initial_support) / max_change for con in conv_u95]

        vacc_avg = np.array([obs['vaccinated']['avg'] if 'vaccinated' in obs and 'avg' in obs['vaccinated'] else np.nan for obs in list(stats_values)])
        vacc_l95 = np.array([obs['vaccinated']['l95'] if 'vaccinated' in obs and 'l95' in obs['vaccinated'] else np.nan for obs in list(stats_values)])
        vacc_u95 = np.array([obs['vaccinated']['u95'] if 'vaccinated' in obs and 'u95' in obs['vaccinated'] else np.nan for obs in list(stats_values)])

        prev_avg = np.array([obs['prevalence']['avg'] if 'prevalence' in obs and 'avg' in obs['prevalence'] else np.nan for obs in list(stats_values)])
        prev_l95 = np.array([obs['prevalence']['l95'] if 'prevalence' in obs and 'l95' in obs['prevalence'] else np.nan for obs in list(stats_values)])
        prev_u95 = np.array([obs['prevalence']['u95'] if 'prevalence' in obs and 'u95' in obs['prevalence'] else np.nan for obs in list(stats_values)])

        color = plt.cm.viridis(norm(initial_support))

        ax[0].scatter(array_rv, delta_con_avg, color=color)
        ax[0].fill_between(array_rv, delta_con_l95, delta_con_u95, color=color, alpha=0.2)
        ax[1].scatter(array_rv, vacc_avg, color=color)
        ax[1].fill_between(array_rv, vacc_l95, vacc_u95, color=color, alpha=0.2)
        ax[2].scatter(array_rv, prev_avg, color=color)
        ax[2].fill_between(array_rv, prev_l95, prev_u95, color=color, alpha=0.2)
        ax[2].axvline(0.09,linestyle='dashed', color='gray')

        state_code = ut.extract_code_from_state(model_region)
        ut.write_annotations_vaccination_curves(state_code, ax, array_rv, vacc_avg, prev_avg)

    cbar_ml = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax[2], orientation='vertical')
    cbar_ml.set_label(r'$v(0)+n_A(0)$', size=25)
    cbar_ml.ax.tick_params(labelsize=15)

    #ax[0].legend(loc='upper right', fontsize=15)
    #ax[1].legend(loc='lower right', fontsize=15)

    ax[0].set_title(r"$\Delta n_A(\infty)$", fontsize=30)
    ax[1].set_title(r"$v(\infty)$", fontsize=30)
    ax[2].set_title(r"$r(\infty)$", fontsize=30)

    ax[0].text(0.04, 0.9, r"a", transform=ax[0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1].text(0.04, 0.9, r"b", transform=ax[1].transAxes, fontsize=30, color='black', weight="bold")
    ax[2].text(0.04, 0.9, r"c", transform=ax[2].transAxes, fontsize=30, color='black', weight="bold")

    ax[0].set_xlabel(r'$\alpha$', fontsize=30)
    ax[1].set_xlabel(r'$\alpha$', fontsize=30)
    ax[2].set_xlabel(r'$\alpha$', fontsize=30)

    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')

    ax[0].set_ylabel(r'Population fraction', fontsize=30)

    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].tick_params(axis='both', labelsize=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[2].tick_params(axis='both', which='major', labelsize=15)
    ax[2].tick_params(axis='both', labelsize=15)

    ax[0].set_ylim(-0.025, 1.0)
    ax[1].set_ylim(-0.025, 1.0)
    ax[2].set_ylim(-0.025, 0.5)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    path_full_target = os.path.join(path_cwd, 'figures')
    filename_target = 'threshold_main_vaccination_curves'
    extension_list = ['pdf', 'png']
    if not os.path.exists(path_full_target):
        os.makedirs(path_full_target)
    for ext in extension_list:
        full_name = os.path.join(path_full_target, filename_target + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    
    plt.show()

def plot_state_age_structure_data(
        model_region, 
        flag_rebuild=False, 
        flag_compare=False, 
        path_cwd='',
        ):
    path_source = path_cwd

    data_pop_a = ut.import_age_distribution(state=model_region, path=path_source)
    data_contact = ut.import_contact_matrix(state=model_region, path=path_source)

    age_array = np.array(range(len(data_pop_a)))
    degree_array = age_array
    
    data_full_new_pop_a = ut.import_age_distribution(state=model_region, reference=False, year=2019, path=path_source)
    data_new_contact = ut.update_contact_matrix(data_contact, old_pop_a=data_pop_a, new_pop_a=data_full_new_pop_a)

    n = np.sum(data_full_new_pop_a)

    data_avg_degree_dist = ut.average_degree_distribution(data_new_contact, data_full_new_pop_a, norm_flag=True)

    data_new_pop_a = ut.import_age_distribution(state=model_region, reference=False, year=2019, norm_flag=True, path=path_source)

    ut.check_contact_reciprocity(data_new_contact, data_new_pop_a)

    fig = plt.figure(figsize=(18, 9))

    if flag_rebuild:
        path_full_source_results = os.path.join(path_source, 'results/curated')
        list_filename_rebuilt = ut.collect_filenames(path_search=path_full_source_results, header='contact_ml', string_segments=[model_region], extension='.json')
        filename_rebuilt = list_filename_rebuilt[0]
        main_keys = ['contact']
        observable_keys = ['age_distribution', 'contact_matrix', 'degree_distribution']
        fullname_rebuilt = os.path.join(path_full_source_results, filename_rebuilt)

        #with open(fullname_rebuilt) as file:
        #    output_rebuilt = json.load(file)
        with open(fullname_rebuilt, 'rb') as input_data:
            output_rebuilt = pk.load(input_data)
    
        rebuilt_pop_a = output_rebuilt[observable_keys[0]]
        rebuilt_contact = np.asarray(output_rebuilt[observable_keys[1]])
        rebuilt_degree_dist = ut.average_degree_distribution(rebuilt_contact, rebuilt_pop_a, norm_flag=True)

        ut.check_contact_reciprocity(rebuilt_contact, rebuilt_pop_a)
        #interlayer_probability_sims = ut.interlayer_probability(rebuilt_contact)
    
        if flag_compare:
            plt.suptitle("Age and contact structure for {0} (data and sims)".format(model_region), fontsize=25)

            ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
            
            max_val = np.max(np.abs(100.0 * (data_new_contact - rebuilt_contact) / data_new_contact))
            cax = ax1.matshow(100.0 * (data_new_contact - rebuilt_contact) / data_new_contact, cmap='coolwarm', vmin=-max_val, vmax=max_val)
            cb = fig.colorbar(cax, ax=ax1)
            cb.ax.tick_params(labelsize=14)
            cb.set_label('Percent error (%) = (Data - Sims)/ Data', fontsize=18)

            ax1.set_title(r'Contact matrix difference', fontsize=20)
            ax1.set_ylabel(r'Age group', fontsize=18)
            ax1.set_xlabel(r'Age group', fontsize=18)

            ax1.text(-0.1, 1.05, f'a', transform=ax1.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

            print("Average degree from data contact matrix = {0}".format(np.sum(data_new_pop_a * np.sum(data_new_contact, axis=1))))
            print("Average degree from multilayer contact matrix = {0}".format(np.sum(rebuilt_pop_a * np.sum(rebuilt_contact, axis=1))))

            bar_width = 0.35
            index = np.arange(len(data_new_pop_a))

            ax2 = plt.subplot2grid((2, 3), (0, 2))
            
            ax2.bar(range(len(data_new_pop_a)), data_new_pop_a, color='teal', label='data')
            ax2.bar(index + bar_width, rebuilt_pop_a, bar_width, color='slateblue', label='sims')
            
            ax2.axvline(np.sum(age_array * data_new_pop_a), color='crimson', linestyle='dashed', alpha=1.0)
            ax2.axvline(np.sum(age_array * rebuilt_pop_a), color='crimson', linestyle='dashed', alpha=1.0)
            
            text_offset = 10.75
            
            mean_age_rebuilt = np.sum(age_array * rebuilt_pop_a)

            ax2.text(mean_age_rebuilt + text_offset, max(max(data_new_pop_a), max(rebuilt_pop_a)) * 0.95,
            r"$\langle a\rangle={0}$".format(int(np.round(mean_age_rebuilt))),
            fontsize=12, color='crimson', ha='center')
        
            ax2.set_title(r'Population', fontsize=20)
            ax2.set_xlabel(r'Age group', fontsize=18)
            ax2.set_ylabel(r'Frequency', fontsize=18)

            ax2.text(-0.1, 1.15, f'b', transform=ax2.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

            ax2.legend(loc='upper left')

            ax3 = plt.subplot2grid((2, 3), (1, 2))
            
            ax3.bar(range(len(data_avg_degree_dist)), data_avg_degree_dist, color='teal', label='data')
            ax3.bar(index + bar_width, rebuilt_degree_dist, bar_width, color='slateblue', label='sims')      
            
            ax3.axvline(np.sum(age_array * data_avg_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)  
            ax3.axvline(np.sum(degree_array * rebuilt_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)

            mean_degree_data = np.sum(age_array * data_avg_degree_dist)
            ax3.text(mean_degree_data + text_offset, max(max(data_avg_degree_dist), max(rebuilt_degree_dist)) * 0.95,
            r"$\langle k\rangle={0:.2f}$".format(mean_degree_data),
            fontsize=12, color='crimson', ha='center')
            
            ax3.set_title(r'Average degree distribution', fontsize=20)
            ax3.set_xlabel(r'Number of contacts', fontsize=18)
            ax3.set_ylabel(r'Frequency', fontsize=18)

            ax3.text(-0.1, 1.15, f'c', transform=ax3.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

            ax3.legend(loc='upper right')

        else:
            plt.suptitle("Age and contact structure for {0} (sims)".format(model_region), fontsize=25)

            ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
            cax = ax1.matshow(rebuilt_contact, cmap='viridis')
            cb = fig.colorbar(cax, ax=ax1)
            cb.ax.tick_params(labelsize=14)
            cb.set_label('average per capita contacts', fontsize=18)

            ax1.set_ylabel(r'Age group', fontsize=18)
            ax1.set_xlabel(r'Age group', fontsize=18)

            ax1.text(-0.1, 1.15, f'a', transform=ax1.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')
    
            print("Average degree from simulated contact matrix = {0}".format(np.mean(rebuilt_contact)))

            bar_width = 0.35
            index = np.arange(len(data_new_pop_a))

            ax2 = plt.subplot2grid((2, 3), (0, 2))
            ax2.bar(index + bar_width, rebuilt_pop_a, bar_width, color='slateblue', label='sims')
            ax2.axvline(np.sum(age_array * rebuilt_pop_a), color='crimson', linestyle='dashed', alpha=1.0)

            text_offset = 10.75
            
            mean_age_rebuilt = np.sum(age_array * rebuilt_pop_a)

            ax2.text(mean_age_rebuilt + text_offset, max(max(data_new_pop_a), max(rebuilt_pop_a)) * 0.95,
            r"$\langle a\rangle={0}$".format(int(np.round(mean_age_rebuilt))),
            fontsize=12, color='crimson', ha='center')
            
            ax2.set_title(r'Population', fontsize=20)
            ax2.set_xlabel(r'Age group', fontsize=18)
            ax2.set_ylabel(r'Frequency', fontsize=18)

            ax2.text(-0.1, 1.15, f'b', transform=ax2.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

            ax3 = plt.subplot2grid((2, 3), (1, 2))
            ax3.bar(index + bar_width, rebuilt_degree_dist, bar_width, color='slateblue', label='sims')      
            
            ax3.axvline(np.sum(age_array * data_avg_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)  
            ax3.axvline(np.sum(degree_array * rebuilt_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)

            mean_degree_data = np.sum(age_array * data_avg_degree_dist)
            ax3.text(mean_degree_data + text_offset, max(max(data_avg_degree_dist), max(rebuilt_degree_dist)) * 0.95,
            r"$\langle k\rangle={0:.2f}$".format(mean_degree_data),
            fontsize=12, color='crimson', ha='center')
            
            ax3.set_title(r'Degree distribution', fontsize=20)
            ax3.set_xlabel(r'Number of contacts', fontsize=18)
            ax3.set_ylabel(r'Frequency', fontsize=18)

            ax3.text(-0.1, 1.15, f'c', transform=ax3.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

    else:
        ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
        cax = ax1.matshow(data_new_contact, cmap='viridis')
        cb = fig.colorbar(cax, ax=ax1)
        cb.ax.tick_params(labelsize=14)
        cb.set_label('average per capita contacts', fontsize=18)

        ax1.set_title('Contact matrix', fontsize=20)
        ax1.set_xlabel(r'Age group', fontsize=18)
        ax1.set_ylabel(r'Age group', fontsize=18)
        ax1.xaxis.set_ticks_position('bottom')

        ax1.text(-0.1, 1.15, f'a', transform=ax1.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

        bar_width = 0.35
        index = np.arange(len(data_new_pop_a))

        ax2 = plt.subplot2grid((2, 3), (0, 2))
        
        ax2.bar(range(len(data_new_pop_a)), data_new_pop_a, color='teal', label='data')
        
        ax2.axvline(np.sum(age_array * data_new_pop_a), color='crimson', linestyle='dashed', alpha=1.0)

        text_offset = 0.75
            
        mean_age_rebuilt = np.sum(age_array * rebuilt_pop_a)

        ax2.text(mean_age_rebuilt + text_offset, max(max(data_new_pop_a), max(rebuilt_pop_a)) * 0.95,
        r"$\langle a\rangle={0}$".format(int(np.round(mean_age_rebuilt))),
        fontsize=12, color='crimson', ha='center')

        ax2.set_title(r'Population', fontsize=20)
        ax2.set_xlabel(r'Age group', fontsize=18)
        ax2.set_ylabel(r'frequency', fontsize=18)

        ax2.text(-0.1, 1.15, f'b', transform=ax2.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

        ax3 = plt.subplot2grid((2, 3), (1, 2))
        
        ax3.bar(range(len(data_avg_degree_dist)), data_avg_degree_dist, color='teal', label='data')
        
        ax3.axvline(np.sum(age_array * data_avg_degree_dist), color='crimson', linestyle='dashed', alpha=1.0) 

        mean_degree_data = np.sum(age_array * data_avg_degree_dist)
        ax3.text(mean_degree_data + text_offset, max(max(data_avg_degree_dist), max(rebuilt_degree_dist)) * 0.95,
        r"$\langle k\rangle={0:.2f}$".format(mean_degree_data),
        fontsize=12, color='crimson', ha='center')
        
        ax3.set_title(r'Degree distribution', fontsize=20)
        ax3.set_xlabel(r'Number of contacts', fontsize=18)
        ax3.set_ylabel(r'Frequency', fontsize=18)

        ax3.text(-0.1, 1.15, f'c', transform=ax3.transAxes, fontsize=25, fontweight='bold', verticalalignment='top')

        plt.suptitle("Age and contact structure for {0} (data)".format(model_region), fontsize=25)

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=8)

    plt.tight_layout()
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(path_source, 'figures')
    base_name = 'contact_' + model_region + '_reb_' + str(flag_rebuild) + '_comp_' + str(flag_compare)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

    plt.show()
    #plt.clf()