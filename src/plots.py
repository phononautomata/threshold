import os
import json
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from scipy.stats import nbinom

import utils as ut

cwd_path = os.getcwd()

prevalence_cutoff = 0.001

def plot_panel_homogeneous_thresholds_heatmaps(
        model_region='National', 
        array_thresholds=None, 
        array_active=None, 
        target_vaccination=None,
        target_zealot=0.0, 
        path_cwd=os.getcwd(),
        path_relative_source='results/curated',
        ):

    path_full_source = os.path.join(path_cwd, path_relative_source)
    list_filenames = ut.collect_pickle_filenames(fullpath=path_full_source, header='global__')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

    for i, epi_filename in enumerate(list_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        acf_flag = float(epi_filename.split('_acf')[1].split('_')[0])
        thr_flag = float(epi_filename.split('_thr')[1].split('_')[0])
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
        zef_flag = float(epi_filename.split('_zef')[1].split('_')[0])

        if zef_flag == target_zealot and var_flag in target_vaccination:
            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)
    
            if var_flag not in go_dict:
                go_dict[var_flag] = {}

            key_tuple = (thr_flag, acf_flag)
            go_dict[var_flag][key_tuple] = stat_output

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18, 20))
    cmap = 'viridis'

    for v, var_flag in enumerate(go_dict.keys()):
        par2_array = sorted(list(set([key[1] for key in go_dict.keys()]))) # ACTIVE FRACTION
        par1_array = sorted(list(set([key[0] for key in go_dict.keys()]))) # THRESHOLD
        par1_len = len(par1_array)
        par2_len = len(par2_array)

        X, Y = np.meshgrid(par1_array, par2_array)
        R = np.zeros((par2_len, par1_len))
        V = np.zeros((par2_len, par1_len))
        C = np.zeros((par2_len, par1_len))

        for i, p1 in enumerate(par1_array):
            for j, p2 in enumerate(par2_array):
                if (p1, p2) in go_dict:

                    R[j][i] = go_dict[(p1, p2)]['prevalence']['avg']
                    V[j][i] = go_dict[(p1, p2)]['vaccinated']['avg']

                    max_change = 1.0 - p2
                    norm_change = (go_dict[(p1, p2)]['convinced']['avg'] - p2) / max_change
                    if p2 == 1.0:
                        norm_change = 0.0
                    C[j][i] = norm_change
                else:
                    R[j][i] = np.nan
                    V[j][i] = np.nan
                    C[j][i] = np.nan

        im2 = ax[v, 2].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=0.6)
        im1 = ax[v, 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
        im0 = ax[v, 0].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
        
        contour_levels = [0.0195]  # Adjust these levels as needed
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

def plot_panel_survey_thresholds_vaccination_curves(
        path_cwd, 
        path_relative='results/curated',
        ):
    target_observable_list = ['convinced', 'prevalence', 'vaccinated']

    vaccination_data = ut.load_vaccination_data(cwd_path)

    header = 'global__'

    path_full_source = os.path.join(path_cwd, path_relative)
    list_filenames = ut.collect_pickle_filenames(fullpath=path_full_source, header=header)

    results_dict = {}

    for i, filename in enumerate(list_filenames):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(path_full_source, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])

        output = ut.load_output(fullname, ['global'], target_observable_list)

        prevalence_s = np.array(output['prevalence']) / n
        failed_outbreaks = ut.extract_failed_outbreaks(prevalence_s, prevalence_cutoff=prevalence_cutoff)

        processed_dict = {}
        for key, val in output.items():
            val = np.array(val) / n
            filt_obs = ut.filter_observable_array(observable_distribution=val, failed_outbreaks=failed_outbreaks)
            if len(filt_obs) > 0:
                stat_obs = ut.stat_observable(filt_obs)
                processed_dict[key] = stat_obs
            else:
                stat_obs = ut.stat_observable(val)
                processed_dict[key] = stat_obs

        if state_string not in results_dict:
            results_dict[state_string] = {}

        if var_value not in results_dict[state_string]:
            results_dict[state_string][var_value] = processed_dict

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))

    vmin = 0.25
    vmax = 0.65
    norm = Normalize(vmin=vmin, vmax=vmax)

    for state_idx, state_key in enumerate(results_dict.keys()):
        var_values = sorted(results_dict[state_key].keys())

        stats_values = [results_dict[state_key][var_key] for var_key in var_values]

        conv_avg_w1 = np.array([obs['convinced']['avg'] if 'convinced' in obs and 'avg' in obs['convinced'] else np.nan for obs in list(stats_values)])
        conv_l95_w1 = np.array([obs['convinced']['l95'] if 'convinced' in obs and 'l95' in obs['convinced'] else np.nan for obs in list(stats_values)])
        conv_u95_w1 = np.array([obs['convinced']['u95'] if 'convinced' in obs and 'u95' in obs['convinced'] else np.nan for obs in list(stats_values)])

        vac_shares = vaccination_data[state_key]
        already = vac_shares['already']
        soon = vac_shares['soon']
        initial_support = already + soon
        zealots = vac_shares['never']

        already = vaccination_data[state_key]['already']
        soon = vaccination_data[state_key]['soon']  

        max_change = 1.0 - (already + soon + zealots)
        delta_con_avg = [(con - initial_support) / max_change for con in conv_avg_w1]
        delta_con_l95 = [(con - initial_support) / max_change for con in conv_l95_w1]
        delta_con_u95 = [(con - initial_support) / max_change for con in conv_u95_w1]

        vacc_avg = np.array([obs['vaccinated']['avg'] if 'vaccinated' in obs and 'avg' in obs['vaccinated'] else np.nan for obs in list(stats_values)])
        vacc_l95 = np.array([obs['vaccinated']['l95'] if 'vaccinated' in obs and 'l95' in obs['vaccinated'] else np.nan for obs in list(stats_values)])
        vacc_u95 = np.array([obs['vaccinated']['u95'] if 'vaccinated' in obs and 'u95' in obs['vaccinated'] else np.nan for obs in list(stats_values)])

        prev_avg_w1 = np.array([obs['prevalence']['avg'] if 'prevalence' in obs and 'avg' in obs['prevalence'] else np.nan for obs in list(stats_values)])
        prev_l95_w1 = np.array([obs['prevalence']['l95'] if 'prevalence' in obs and 'l95' in obs['prevalence'] else np.nan for obs in list(stats_values)])
        prev_u95_w1 = np.array([obs['prevalence']['u95'] if 'prevalence' in obs and 'u95' in obs['prevalence'] else np.nan for obs in list(stats_values)])

        color = plt.cm.viridis(norm(initial_support))

        ax[0].scatter(var_values, delta_con_avg, color=color)
        ax[0].fill_between(var_values, delta_con_l95, delta_con_u95, color=color, alpha=0.2)
        ax[1].scatter(var_values, vacc_avg, color=color)
        ax[1].fill_between(var_values, vacc_l95, vacc_u95, color=color, alpha=0.2)
        ax[2].scatter(var_values, prev_avg_w1, color=color)
        ax[2].fill_between(var_values, prev_l95_w1, prev_u95_w1, color=color, alpha=0.2)
        ax[2].axvline(0.09,linestyle='dashed', color='gray')

        state_code = ut.extract_code_from_state(state_key)
        ut.write_annotations_vaccination_curves(state_code, ax,  var_values, vacc_avg, prev_avg_w1)

    cbar_ml = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax[2], orientation='vertical')
    cbar_ml.set_label(r'$v(0)+n_A(0)$', size=25)
    cbar_ml.ax.tick_params(labelsize=15)

    #ax[0].legend(loc='upper right', fontsize=15)
    #ax[1].legend(loc='lower right', fontsize=15)

    ax[0].set_title(r"$\Delta n_A(\infty)$", fontsize=30)
    ax[1].set_title(r"$v(\infty)$", fontsize=30)
    ax[2].set_title(r"$r(\infty)$", fontsize=30)

    ax[0].text(0.04, 0.9, r"A", transform=ax[0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1].text(0.04, 0.9, r"B", transform=ax[1].transAxes, fontsize=30, color='black', weight="bold")
    ax[2].text(0.04, 0.9, r"C", transform=ax[2].transAxes, fontsize=30, color='black', weight="bold")

    ax[0].set_xlabel(r'$\alpha$', fontsize=30)
    ax[1].set_xlabel(r'$\alpha$', fontsize=30)
    ax[2].set_xlabel(r'$\alpha$', fontsize=30)

    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')

    ax[0].set_ylabel(r'population fraction', fontsize=30)

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
    plt.clf()

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