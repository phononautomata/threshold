import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from scipy.stats import nbinom

import utils as ut

cwd_path = os.getcwd()

lower_path_data = 'data'
lower_path_results = 'results'
lower_path_figures = 'figures'

header_project = 'thr_'
multilayer_exp_string = '3_'

prevalence_cutoff = 0.001

def plot_state_age_structure_data(state_id, rebuild_flag=False, comp_flag=False):
    pop_a = ut.import_age_distribution(state=state_id)
    contact = ut.import_contact_matrix(state=state_id)

    age_array = np.array(range(len(pop_a)))
    degree_array = age_array
    
    full_new_pop_a = ut.import_age_distribution(state=state_id, reference=False, year=2019)
    new_contact = ut.update_contact_matrix(contact, old_pop_a=pop_a, new_pop_a=full_new_pop_a)

    n = np.sum(full_new_pop_a)

    avg_degree_dist = ut.average_degree_distribution(new_contact, full_new_pop_a, norm_flag=True)

    new_pop_a = ut.import_age_distribution(state=state_id, reference=False, year=2019, norm_flag=True)

    ut.check_contact_reciprocity(new_contact, new_pop_a)
    interlayer_probability_data = ut.interlayer_probability(new_contact)

    fig = plt.figure(figsize=(18, 9))

    if rebuild_flag:
        fullpath = os.path.join(cwd_path, 'results')
        rebuild_fullname = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_rebstat_', string_segments=[state_id])[0]
        main_keys = ['rebuild']
        observable_keys = ['mean_age_distribution', 'mean_contact_matrix', 'mean_degree_distribution']
        rebuild_fullname = os.path.join(fullpath, rebuild_fullname)
        rebuild_output = ut.load_output(fullname=rebuild_fullname, main_keys=main_keys, observable_keys=observable_keys)

        rebuilt_pop_a = rebuild_output[observable_keys[0]]
        rebuilt_contact = np.asarray(rebuild_output[observable_keys[1]])
        rebuilt_degree_dist = ut.average_degree_distribution(rebuilt_contact, rebuilt_pop_a, norm_flag=True)

        ut.check_contact_reciprocity(rebuilt_contact, rebuilt_pop_a)
        interlayer_probability_sims = ut.interlayer_probability(rebuilt_contact)
    
        if comp_flag:
            plt.suptitle("Age and contact structure for {0} (data and sims)".format(state_id), fontsize=25)

            ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
            max_val = np.max(np.abs(100.0 * (new_contact - rebuilt_contact) / new_contact))
            cax = ax1.matshow(100.0 * (new_contact - rebuilt_contact) / new_contact, cmap='coolwarm', vmin=-max_val, vmax=max_val)
            cb = fig.colorbar(cax, ax=ax1)
            cb.ax.tick_params(labelsize=14)
            cb.set_label('MPE (%) (data - sims)', fontsize=18)

            print("Average degree from data contact matrix = {0}".format(np.sum(new_pop_a * np.sum(new_contact, axis=1))))
            print("Average degree from simulated contact matrix = {0}".format(np.sum(rebuilt_pop_a * np.sum(rebuilt_contact, axis=1))))

            bar_width = 0.35
            index = np.arange(len(new_pop_a))

            ax2 = plt.subplot2grid((2, 3), (0, 2))
            ax2.bar(range(len(new_pop_a)), new_pop_a, color='teal', label='data')
            ax2.bar(index + bar_width, rebuilt_pop_a, bar_width, color='slateblue', label='sims')
            ax2.axvline(np.sum(age_array * new_pop_a), color='crimson', linestyle='dashed', alpha=1.0)
            ax2.axvline(np.sum(age_array * rebuilt_pop_a), color='crimson', linestyle='dashed', alpha=1.0)
            ax2.set_title('Population', fontsize=20)
            ax2.set_xlabel(r'age group', fontsize=18)
            ax2.set_ylabel(r'frequency', fontsize=18)

            ax2.legend(loc='upper left')

            ax3 = plt.subplot2grid((2, 3), (1, 2))
            ax3.bar(range(len(avg_degree_dist)), avg_degree_dist, color='teal', label='data')
            ax3.bar(index + bar_width, rebuilt_degree_dist, bar_width, color='slateblue', label='sims')      
            ax3.axvline(np.sum(age_array * avg_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)  
            ax3.axvline(np.sum(degree_array * rebuilt_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)
            ax3.set_title(r'Degree distribution', fontsize=20)
            ax3.set_xlabel(r'number of contacts', fontsize=18)
            ax3.set_ylabel(r'frequency', fontsize=18)

            ax3.legend(loc='upper right')

        else:
            plt.suptitle("Age and contact structure for {0} (sims)".format(state_id), fontsize=25)

            ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
            cax = ax1.matshow(rebuilt_contact, cmap='viridis')
            cb = fig.colorbar(cax, ax=ax1)
            cb.ax.tick_params(labelsize=14)
            cb.set_label('average per capita contacts', fontsize=18)

            ax1.set_title('Contact matrix difference', fontsize=20)
            ax1.set_ylabel(r'age group', fontsize=18)
            ax1.set_xlabel(r'age group', fontsize=18)
    
            print("Average degree from simulated contact matrix = {0}".format(np.mean(rebuilt_contact)))

            bar_width = 0.35
            index = np.arange(len(new_pop_a))

            ax2 = plt.subplot2grid((2, 3), (0, 2))
            ax2.bar(index + bar_width, rebuilt_pop_a, bar_width, color='slateblue', label='sims')
            ax2.axvline(np.sum(age_array * rebuilt_pop_a), color='crimson', linestyle='dashed', alpha=1.0)
            ax2.set_title('Population', fontsize=20)
            ax2.set_xlabel(r'age group', fontsize=18)
            ax2.set_ylabel(r'frequency', fontsize=18)

            ax3 = plt.subplot2grid((2, 3), (1, 2))
            ax3.bar(index + bar_width, rebuilt_degree_dist, bar_width, color='slateblue', label='sims')      
            ax3.axvline(np.sum(age_array * avg_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)  
            ax3.axvline(np.sum(degree_array * rebuilt_degree_dist), color='crimson', linestyle='dashed', alpha=1.0)
            ax3.set_title(r'Degree distribution', fontsize=20)
            ax3.set_xlabel(r'number of contacts', fontsize=18)
            ax3.set_ylabel(r'frequency', fontsize=18)

    else:
        ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
        cax = ax1.matshow(new_contact, cmap='viridis')
        cb = fig.colorbar(cax, ax=ax1)
        cb.ax.tick_params(labelsize=14)
        cb.set_label('average per capita contacts', fontsize=18)

        ax1.set_title('Contact matrix', fontsize=20)
        ax1.set_xlabel(r'age group', fontsize=18)
        ax1.set_ylabel(r'age group', fontsize=18)
        ax1.xaxis.set_ticks_position('bottom')

        bar_width = 0.35
        index = np.arange(len(new_pop_a))

        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax2.bar(range(len(new_pop_a)), new_pop_a, color='teal', label='data')
        ax2.axvline(np.sum(age_array * new_pop_a), color='crimson', linestyle='dashed', alpha=1.0)
        ax2.set_title('Population', fontsize=20)
        ax2.set_xlabel(r'age group', fontsize=18)
        ax2.set_ylabel(r'frequency', fontsize=18)

        ax3 = plt.subplot2grid((2, 3), (1, 2))
        ax3.bar(range(len(avg_degree_dist)), avg_degree_dist, color='teal', label='data')
        ax3.axvline(np.sum(age_array * avg_degree_dist), color='crimson', linestyle='dashed', alpha=1.0) 
        ax3.set_title(r'Degree distribution', fontsize=20)
        ax3.set_xlabel(r'number of contacts', fontsize=18)
        ax3.set_ylabel(r'frequency', fontsize=18)

        plt.suptitle("Age and contact structure for {0} (data)".format(state_id), fontsize=25)

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

    full_path = os.path.join(cwd_path, lower_path_figures)
    base_name = 'age_structure_' + state_id + '_sim_' + str(rebuild_flag) + '_comp_' + str(comp_flag)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_all_global_observable(
        lower_path_results=lower_path_results, 
        prevalence_cutoff=prevalence_cutoff,
        target_observable=None,
        target_var=None,
        ):
    
    header = header_project + 'global_' + multilayer_exp_string

    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    results_dict = {}

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])

        if var_value == target_var:
            output = ut.load_output(fullname, ['global'], [target_observable])

            prevalence_s = np.array(output['prevalence']) / n
            failed_outbreaks = ut.extract_failed_outbreaks(prevalence_s, prevalence_cutoff=prevalence_cutoff)

            processed_dict = {}
            for key, val in output.items():
                val = np.array(val) / n
                filt_obs = ut.filter_observable_array(observable_distribution=val, failed_outbreaks=failed_outbreaks)
                stat_obs = ut.stat_observable(filt_obs)
                processed_dict[key] = stat_obs

            if state_string not in results_dict:
                results_dict[state_string] = {}

            if var_value not in results_dict[state_string]:
                results_dict[state_string][var_value] = processed_dict

    states = []
    averages = []
    errors = []

    for state_idx, state_string in enumerate(results_dict):
        processed_output = results_dict[state_string][target_var]
        target_observable_stats = processed_output[target_observable]

        avg_to = target_observable_stats['avg']
        l95_to = target_observable_stats['l95']
        u95_to = target_observable_stats['u95']

        state_code = ut.get_state_code(state_string)

        states.append(state_code)
        averages.append(avg_to)
        errors.append(avg_to - l95_to, u95_to - avg_to)

    combined = list(zip(states, averages, errors))
    combined_sorted = sorted(combined, key=lambda x: x[0])
    states_sorted, averages_sorted, errors_sorted = zip(*combined_sorted)
    
    errors = np.array(errors).T

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    ax.bar(states_sorted, averages_sorted, yerr=errors_sorted, capsize=5)
    ax.set_xlabel('US states')
    ax.set_ylabel('Observable Value')
    ax.set_xticklabels(states, rotation=45)
        
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(cwd_path, lower_path_figures)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'all_global_' + target_observable + '_' + target_var
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_state_attribute_stratified_observable(
        lower_path_results=lower_path_results, 
        prevalence_cutoff=prevalence_cutoff,
        target_attribute=None,
        target_observable=None,
        target_state=None,
        target_var=None,
        age_norm_flag=True,
        ):

    header = header_project + target_attribute + '_' + multilayer_exp_string
    
    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])
    
        if state_string == target_state and var_value == target_var:
            output = ut.load_output(fullname, main_keys=[target_attribute], observable_keys=[target_observable, 'age'])

            prevalence_sa = output['prevalence']
            prevalence_s = np.sum(prevalence_sa, axis=1) / n
            failed_outbreaks = ut.extract_failed_outbreaks(prevalence_s, prevalence_cutoff=prevalence_cutoff)

            filtered_prevalence_s = np.delete(prevalence_s, failed_outbreaks)

            pop_sa = np.array(output['age'])

            if age_norm_flag:
                n = pop_sa
    
            processed_dict = {}
            for key, val in output.items():
                if 'time' not in key and 'when' not in key:
                    val = np.array(val) / n
                    filt_obs = ut.filter_observable_array(observable_distribution=val, failed_outbreaks=failed_outbreaks)
                    stat_obs = ut.stat_stratified_attribute_observable_array(filt_obs)
                else:
                    filt_obs = ut.filter_observable_list(observable_distribution_list=val, failed_outbreaks=failed_outbreaks)
                    stat_obs = ut.stat_stratified_attribute_observable_list(filt_obs)
                processed_dict[key] = stat_obs
            
            norm_pop_sa = pop_sa / np.sum(pop_sa[:], axis=1)[:, np.newaxis]
            stat_obs = ut.stat_stratified_attribute_observable_array(norm_pop_sa)
            processed_dict['pop_age'] = stat_obs

    ngroups = len(processed_dict['prevalence'])
    age_groups = [str(age) for age in range(ngroups)]
    
    distributed_observable_list = ['degree', 'convinced_when', 'infected_when', 'removed_when', 'vaccinated_when']

    fig, ax = plt.subplots(figsize=(10, 6))

    if target_observable in distributed_observable_list:
        dist_avg_per_age = processed_dict[target_observable]['dist_avg_per_age']
        dist_l95_per_age = processed_dict[target_observable]['dist_l95_per_age']
        dist_u95_per_age = processed_dict[target_observable]['dist_u95_per_age']
        error_lower = dist_avg_per_age - dist_l95_per_age
        error_upper = dist_u95_per_age - dist_avg_per_age

        dist_avg_global = processed_dict[target_observable]['dist_avg_global']
        dist_l95_global = processed_dict[target_observable]['dist_l95_global']
        dist_u95_global = processed_dict[target_observable]['dist_u95_global']

        ax.bar(age_groups, dist_avg_per_age, yerr=[error_lower, error_upper], color='royalblue', capsize=5, ecolor='cornflowerblue')

        ax.axhline(dist_avg_global, color='crimson', linestyle='dashed',)
        ax.fill_between(age_groups, dist_l95_global, dist_u95_global, color='crimson', alpha=0.2)
    
    else:
        obs_avg_per_age = np.array([processed_dict[target_observable][age]['avg'] for age in range(ngroups)])
        obs_l95_per_age = np.array([processed_dict[target_observable][age]['l95'] for age in range(ngroups)])
        obs_u95_per_age = np.array([processed_dict[target_observable][age]['u95'] for age in range(ngroups)])
        error_lower = obs_avg_per_age - obs_l95_per_age
        error_upper = obs_u95_per_age - obs_avg_per_age

        ax.bar(age_groups, obs_avg_per_age, yerr=[error_lower, error_upper], color='royalblue', capsize=5, ecolor='cornflowerblue')
        
        if target_observable == 'prevalence' and age_norm_flag:
            global_prevalence = np.mean(filtered_prevalence_s)
            ax.axhline(global_prevalence, color='crimson', linestyle='dashed')
        elif target_observable == 'prevalence' and age_norm_flag == False:
            pop_avg_per_age = np.array([processed_dict['pop_age'][age]['avg'] for age in range(ngroups)])
            pop_l95_per_age = np.array([processed_dict['pop_age'][age]['l95'] for age in range(ngroups)])
            pop_u95_per_age = np.array([processed_dict['pop_age'][age]['avg'] for age in range(ngroups)])
            pop_error_lower = pop_avg_per_age - pop_l95_per_age
            pop_error_upper = pop_u95_per_age - pop_avg_per_age

            ax.bar(age_groups, pop_avg_per_age, yerr=[pop_error_lower, pop_error_upper], color='firebrick', capsize=5, ecolor='salmon', alpha=0.2)
        else:
            pass

    ax.set_xlabel('age group', fontsize=20)
    ax.set_ylabel('{0}'.format(target_observable), fontsize=20)
    ax.set_title('{0} by {1}'.format(target_observable, target_attribute))

    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=6)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(cwd_path, lower_path_figures)
    base_name = 'bar_' + target_state + '_' + target_attribute + '_' + target_observable + '_' + str(target_var)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_state_attribute_stratified_panel(
        lower_path_results=lower_path_results, 
        prevalence_cutoff=prevalence_cutoff,
        target_attribute=None,
        target_observable=None,
        target_state=None,
        target_var=None,
        age_norm_flag=True,):
    header = header_project + target_attribute + '_' + multilayer_exp_string
    
    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])
    
        if state_string == target_state and var_value == target_var:
            output = ut.load_output(fullname, main_keys=[target_attribute], observable_keys=[target_observable, 'age'])

            prevalence_sa = output['prevalence']
            prevalence_s = np.sum(prevalence_sa, axis=1) / n
            failed_outbreaks = ut.extract_failed_outbreaks(prevalence_s, prevalence_cutoff=prevalence_cutoff)

            filtered_prevalence_s = np.delete(prevalence_s, failed_outbreaks)

            pop_sa = np.array(output['age'])

            if age_norm_flag:
                n = pop_sa
    
            processed_dict = {}
            for key, val in output.items():
                if 'time' not in key and 'when' not in key:
                    val = np.array(val) / n
                    filt_obs = ut.filter_observable_array(observable_distribution=val, failed_outbreaks=failed_outbreaks)
                    stat_obs = ut.stat_stratified_attribute_observable_array(filt_obs)
                else:
                    filt_obs = ut.filter_observable_list(observable_distribution_list=val, failed_outbreaks=failed_outbreaks)
                    stat_obs = ut.stat_stratified_attribute_observable_list(filt_obs)
                processed_dict[key] = stat_obs
            
            norm_pop_sa = pop_sa / np.sum(pop_sa[:], axis=1)[:, np.newaxis]
            stat_obs = ut.stat_stratified_attribute_observable_array(norm_pop_sa)
            processed_dict['pop_age'] = stat_obs

    ngroups = len(processed_dict['prevalence'])
    age_groups = [str(age) for age in range(ngroups)]

    distributed_observable_list = ['degree', 'convinced_when', 'infected_when', 'removed_when', 'vaccinated_when']

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 6))

    if target_observable in distributed_observable_list:
        dist_avg_per_age = processed_dict[target_observable]['dist_avg_per_age']
        dist_l95_per_age = processed_dict[target_observable]['dist_l95_per_age']
        dist_u95_per_age = processed_dict[target_observable]['dist_u95_per_age']
        error_lower = dist_avg_per_age - dist_l95_per_age
        error_upper = dist_u95_per_age - dist_avg_per_age

        dist_avg_global = processed_dict[target_observable]['dist_avg_global']
        dist_l95_global = processed_dict[target_observable]['dist_l95_global']
        dist_u95_global = processed_dict[target_observable]['dist_u95_global']

        ax[0, 0].plot(age_groups, dist_avg_per_age, yerr=[error_lower, error_upper], color='royalblue', capsize=5, ecolor='cornflowerblue')

        ax[0, 0].axhline(dist_avg_global, color='crimson', linestyle='dashed',)
        ax[0, 0].fill_between(age_groups, dist_l95_global, dist_u95_global, color='crimson', alpha=0.2)
    
    else:
        obs_avg_per_age = np.array([processed_dict[target_observable][age]['avg'] for age in range(ngroups)])
        obs_l95_per_age = np.array([processed_dict[target_observable][age]['l95'] for age in range(ngroups)])
        obs_u95_per_age = np.array([processed_dict[target_observable][age]['u95'] for age in range(ngroups)])
        error_lower = obs_avg_per_age - obs_l95_per_age
        error_upper = obs_u95_per_age - obs_avg_per_age

        ax[0,].bar(age_groups, obs_avg_per_age, yerr=[error_lower, error_upper], color='royalblue', capsize=5, ecolor='cornflowerblue')
        
        if target_observable == 'prevalence' and age_norm_flag:
            global_prevalence = np.mean(filtered_prevalence_s)
            ax.axhline(global_prevalence, color='crimson', linestyle='dashed')
        elif target_observable == 'prevalence' and age_norm_flag == False:
            pop_avg_per_age = np.array([processed_dict['pop_age'][age]['avg'] for age in range(ngroups)])
            pop_l95_per_age = np.array([processed_dict['pop_age'][age]['l95'] for age in range(ngroups)])
            pop_u95_per_age = np.array([processed_dict['pop_age'][age]['avg'] for age in range(ngroups)])
            pop_error_lower = pop_avg_per_age - pop_l95_per_age
            pop_error_upper = pop_u95_per_age - pop_avg_per_age

            ax.bar(age_groups, pop_avg_per_age, yerr=[pop_error_lower, pop_error_upper], color='firebrick', capsize=5, ecolor='salmon', alpha=0.2)
        else:
            pass

    ax.set_xlabel('age group', fontsize=20)
    ax.set_ylabel('{0}'.format(target_observable), fontsize=20)
    ax.set_title('{0} by {1}'.format(target_observable, target_attribute))

    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=6)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(cwd_path, lower_path_figures)
    base_name = 'bar_' + target_state + '_' + target_attribute + '_' + target_observable + '_' + str(target_var)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_all_scatter_observable(
        lower_path_results=lower_path_results, 
        prevalence_cutoff=prevalence_cutoff,
        target_factor=None,
        target_observable=None,
        target_var=None,
        ):
    
    vaccination_data = ut.load_vaccination_data(cwd_path)

    header = header_project + 'global_' + multilayer_exp_string

    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    results_dict = {}

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])

        if var_value == target_var:
            output = ut.load_output(fullname, ['global'], [target_observable])

            prevalence_s = np.array(output['prevalence']) / n
            failed_outbreaks = ut.extract_failed_outbreaks(prevalence_s, prevalence_cutoff=prevalence_cutoff)

            processed_dict = {}
            for key, val in output.items():
                val = np.array(val) / n
                filt_obs = ut.filter_observable_array(observable_distribution=val, failed_outbreaks=failed_outbreaks)
                stat_obs = ut.stat_observable(filt_obs)
                processed_dict[key] = stat_obs

            if state_string not in results_dict:
                results_dict[state_string] = {}

            if var_value not in results_dict[state_string]:
                results_dict[state_string][var_value] = processed_dict

    scatter_dict = {}

    for state_idx, state_string in enumerate(results_dict.keys()):
        vaccination_shares = vaccination_data[state_string]

        already = vaccination_shares['already']
        soon = vaccination_shares['soon']
        someone = vaccination_shares['someone']
        majority = vaccination_shares['majority']
        never = vaccination_shares['never']
        initial_support = already + soon
        hesitant = 1.0 - initial_support - never - someone

        scatter_dict = {}
        scatter_dict[state_string] = {}

        for v, var_value in enumerate(results_dict[state_string].keys()):
            stat_dict = {
                'avg': results_dict[state_string][var_value][target_observable]['avg'],
                'l95': results_dict[state_string][var_value][target_observable]['l95'],
                'u95': results_dict[state_string][var_value][target_observable]['u95'],
                'already': already,
                'soon': someone,
                'majority': majority,
                'never': never,
                'initial_support': initial_support,
                'hesitant': hesitant,
                }
            scatter_dict[state_string][var_value] = stat_dict

    scatter_dict = {state_string: scatter_dict[state_string][target_var] for state_string in scatter_dict.keys()}

    x_values = []
    y_values = []
    for state, state_dict in scatter_dict.items():
        x_values.append(state_dict[target_factor])
        y_values.append(state_dict['avg'])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x_values, y_values)

    ax.set_xlabel(target_factor)
    ax.set_ylabel(target_observable)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(cwd_path, lower_path_figures)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'all_scatter_' + target_var + '_' + target_observable + '_' + target_factor
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_all_vaccination_curves(lower_path_results, prevalence_cutoff=prevalence_cutoff):
    target_observable_list = ['convinced', 'prevalence', 'vaccinated']

    vaccination_data = ut.load_vaccination_data(cwd_path)

    header = header_project + 'global_' + multilayer_exp_string

    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    results_dict = {}

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

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

    full_path = os.path.join(cwd_path, lower_path_figures)
    base_name = 'all_vac_curves'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_all_scatter_panel(lower_path_results, target_observable=['prevalence'], prevalence_cutoff=prevalence_cutoff, target_var=[0]):
    vaccination_data = ut.load_vaccination_data(cwd_path)

    header = header_project + 'global_' + multilayer_exp_string

    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    results_dict = {}

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])

        if var_value in target_var:
            output = ut.load_output(fullname, ['global'], target_observable)

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

            if var_value not in results_dict:
                results_dict[var_value] = {}

            if state_string not in results_dict[var_value]:
                results_dict[var_value][state_string] = processed_dict

    sorted_keys = sorted(results_dict.keys())
    results_dict = dict([(key, results_dict[key]) for key in sorted_keys])

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(25, 15))

    for v, var_key in enumerate(results_dict.keys()):
        already_list = []
        soon_list = []
        someone_list = []
        most_list = []
        never_list = []
        initial_support_list = []
        hesitant_list = []
        
        con_avg_list = []
        dco_avg_list = []
        vacc_avg_list = []
        prw1_avg_list = []

        for state_idx, state_key in enumerate(results_dict[var_key].keys()):
            vac_shares = vaccination_data[state_key]

            already = vac_shares['already']
            soon = vac_shares['soon']
            someone = vac_shares['someone']
            most = vac_shares['majority']
            never = vac_shares['never']
            initial_support = already + soon
            hesitant = 1.0 - initial_support - never - someone

            if state_key != 'Alaska':
                already_list.append(already)
                soon_list.append(soon)
                someone_list.append(someone)
                most_list.append(most)
                never_list.append(never)
                initial_support_list.append(initial_support)
                hesitant_list.append(hesitant)

                outcome = results_dict[var_key][state_key]
                con_avg = outcome['convinced']['avg']
                dco_avg = (con_avg - initial_support) / (1.0 - initial_support)
                prw1_avg = outcome['prevalence']['avg']
                vacc_avg = outcome['vaccinated']['avg']

            if state_key != 'Alaska':
                con_avg_list.append(con_avg)
                dco_avg_list.append(dco_avg)
                prw1_avg_list.append(prw1_avg)
                vacc_avg_list.append(vacc_avg)

        scatter_plots = []
        regressions = []
        for i, lst in enumerate([already_list, soon_list, someone_list, most_list, never_list]):
            scatter, reg = ut.plot_regression(ax, lst, prw1_avg_list, v, i)
            scatter_plots.append(scatter)
            regressions.append(reg)
    
    #ax[0].legend(loc='upper right', fontsize=15)
    #ax[1].legend(loc='lower right', fontsize=15)   

    ax[0, 0].text(-0.4, 0.35, r"$\alpha={0}$".format(target_var[0]), 
               transform=ax[0, 0].transAxes, fontsize=20, color='black', weight="bold", rotation=90)
    ax[1, 0].text(-0.4, 0.35, r"$\alpha={0}$".format(0.005), 
                   transform=ax[1, 0].transAxes, fontsize=20, color='black', weight="bold", rotation=90)
    ax[2, 0].text(-0.4, 0.35, r"$\alpha={0}$".format(target_var[2]), 
                   transform=ax[2, 0].transAxes, fontsize=20, color='black', weight="bold", rotation=90)

    ax[0, 0].text(0.8, 0.8, r"A1", transform=ax[0, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 1].text(0.8, 0.8, r"B1", transform=ax[0, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 2].text(0.8, 0.8, r"C1", transform=ax[0, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 3].text(0.8, 0.8, r"D1", transform=ax[0, 3].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 4].text(0.8, 0.1, r"E1", transform=ax[0, 4].transAxes, fontsize=30, color='black', weight="bold")
    
    ax[1, 0].text(0.8, 0.8, r"A2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 1].text(0.8, 0.8, r"B2", transform=ax[1, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 2].text(0.8, 0.8, r"C2", transform=ax[1, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 3].text(0.8, 0.8, r"D2", transform=ax[1, 3].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 4].text(0.8, 0.1, r"E2", transform=ax[1, 4].transAxes, fontsize=30, color='black', weight="bold")
    
    ax[2, 0].text(0.8, 0.8, r"A3", transform=ax[2, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 1].text(0.8, 0.8, r"B3", transform=ax[2, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 2].text(0.8, 0.8, r"C3", transform=ax[2, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 3].text(0.8, 0.8, r"D3", transform=ax[2, 3].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 4].text(0.8, 0.1, r"E3", transform=ax[2, 4].transAxes, fontsize=30, color='black', weight="bold")

    ax[0, 0].xaxis.set_label_position('top')
    ax[0, 1].xaxis.set_label_position('top')
    ax[0, 2].xaxis.set_label_position('top')
    ax[0, 3].xaxis.set_label_position('top')
    ax[0, 4].xaxis.set_label_position('top')

    ax[0, 0].set_xlabel(r'already, $v(0)$', fontsize=30)
    ax[0, 1].set_xlabel(r'soon, $n_A(0)$', fontsize=30)
    ax[0, 2].set_xlabel(r'someone', fontsize=30)
    ax[0, 3].set_xlabel(r'most', fontsize=30)
    ax[0, 4].set_xlabel(r'never, $n_Z$', fontsize=30)

    ax[0, 0].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$r(\infty)$', fontsize=30)

    for axs in ax.flatten():
        #axs.set_xlim(0.0, 1.0)
        #axs.set_ylim(0.0, 1.0)
        axs.tick_params(axis='both', labelsize=16)
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(cwd_path, lower_path_figures)
    base_name = 'all_scatter_panel_' + target_observable[0] + '_var_' + str(target_var)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_state_cluster( 
        prevalence_cutoff=prevalence_cutoff,
        target_attribute=None,
        target_cluster=None,
        target_state=None,
        target_var=None,
        age_norm_flag=True,
        ):
    header = header_project + target_attribute + '_' + multilayer_exp_string
    
    fullpath = os.path.join(cwd_path, lower_path_results)
    filenames_epi = ut.collect_pickle_filenames(fullpath=fullpath, header=header)

    for i, filename in enumerate(filenames_epi):
        print("Loop {0}. Filename: {1}".format(i + 1, filename))

        fullname = os.path.join(cwd_path, lower_path_results, filename)

        state_string = filename.split('_3_')[1].split('_')[0]
        var_value = float(filename.split('_var')[1].split('_')[0])

        before, after = ut.find_nth_occurrence(filename, '_n', 2)
        n = int(after.split('_')[0])
    
        if state_string == target_state and var_value == target_var:
            output = ut.load_output(fullname, main_keys=[target_attribute], observable_keys=[target_cluster, 'age'])

            prevalence_sa = output['prevalence']
            prevalence_s = np.sum(prevalence_sa, axis=1) / n
            failed_outbreaks = ut.extract_failed_outbreaks(prevalence_s, prevalence_cutoff=prevalence_cutoff)

            filtered_prevalence_s = np.delete(prevalence_s, failed_outbreaks)

            pop_sa = np.array(output['age'])

            if age_norm_flag:
                n = pop_sa
    
            processed_dict = {}
            for key, val in output.items():
                if 'time' not in key and 'when' not in key:
                    val = np.array(val) / n
                    filt_obs = ut.filter_observable_array(observable_distribution=val, failed_outbreaks=failed_outbreaks)
                    stat_obs = ut.stat_stratified_attribute_observable_array(filt_obs)
                else:
                    filt_obs = ut.filter_observable_list(observable_distribution_list=val, failed_outbreaks=failed_outbreaks)
                    stat_obs = ut.stat_stratified_attribute_observable_list(filt_obs)
                processed_dict[key] = stat_obs
            
            norm_pop_sa = pop_sa / np.sum(pop_sa[:], axis=1)[:, np.newaxis]
            stat_obs = ut.stat_stratified_attribute_observable_array(norm_pop_sa)
            processed_dict['pop_age'] = stat_obs

    # Specify the number of bins and density parameter
    num_bins = 30
    density = True

    # Create a new figure and axis objects
    fig, axs = plt.subplots(4, 2, figsize=(12, 8))

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    full_path = os.path.join(cwd_path, lower_path_figures)
    base_name = 'cluster_' + target_cluster + '_' + target_state + '_' + str(target_var)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

#plot_state_age_structure_data(state_id='Massachusetts', rebuild_flag=False, comp_flag=True)
#plot_all_global_observable(lower_path_results='results', prevalence_cutoff=prevalence_cutoff, target_observable='prevalence', target_var=0, )
#plot_state_attribute_stratified_observable(lower_path_results='results', prevalence_cutoff=prevalence_cutoff, target_attribute='age', target_observable='prevalence', target_state='Massachusetts', target_var=0.005, age_norm_flag=False)
#plot_all_vaccination_curves(lower_path_results='results', prevalence_cutoff=prevalence_cutoff)
plot_all_scatter_panel(lower_path_results='results', target_observable=['prevalence', 'convinced', 'vaccinated'], prevalence_cutoff=prevalence_cutoff, target_var=[0.001, 0.005012, 0.01])