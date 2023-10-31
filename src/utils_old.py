import os
import pickle as pk
import numpy as np
import pandas as pd

import analysis as an

path = os.getcwd()

def rename_files():
    # Define the folder path and the old and new name patterns
    lower_path = 'results'
    input_dir = os.path.join(path, lower_path)
    import re

    # Define the old and new substrings to replace
    old_substrings = {
        'thr': 'thr_mc',
        'modeler': 'modelErdosRenyi',
        'modelcomplete': 'modelComplete',
        'paf': 'acf',
        'der': 'rer'
    }
    new_substrings = {
        'thr_mc': 'thr_mc',
        'modelErdosRenyi': 'modelErdosRenyi',
        'modelComplete': 'modelComplete',
        'acf': 'acf',
        'rer': 'rer'
    }

    # Loop over the files in the input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.pickle'):
            continue
        # Replace the old substrings with the new ones
        new_filename = filename
        for old_substring, new_substring in zip(old_substrings, 
                                                new_substrings):
            pattern = re.compile(old_substring)
            new_filename = pattern.sub(new_substrings[new_substring], 
                                       new_filename)
    
        # Rename the file
        old_path = os.path.join(input_dir, filename)
        new_path = os.path.join(input_dir, new_filename)
        os.rename(old_path, new_path)

def get_full_label(abbreviature):

    labels_dict = {}
    labels_dict['thr'] = 'threshold'
    labels_dict['acf'] = 'active_fraction'
    labels_dict['zef'] = 'zealot_fraction'
    labels_dict['rer'] = 'removal_rate'
    labels_dict['var'] = 'vaccination_rate'

    return labels_dict[abbreviature]

def dict_to_string(dictionary):
    return '_'.join([f"{key}{value}" for key, value in dictionary.items()])

def build_header_string(header_dictionary):
    return '_'.join([f"{value}" for key, value in header_dictionary.items()])

def build_header_parameters_dictionary(
        project_id,
        model_id,
        attribute_id,
        exp_id,
        usa_id=None
):
    hpars = {}
    hpars['project'] = project_id
    hpars['model'] = model_id
    hpars['attribute'] = attribute_id
    hpars['exp_id'] = exp_id
    if usa_id != None:
        hpars['usa_id'] = usa_id
    return hpars

def build_network_parameters_dictionary(
        model, 
        n, 
        k=None, 
        p=None, 
        m=None, 
        k_min=None,
        k_max=None, 
        gamma=None,
):
    npars = {}
    npars['model'] = model
    npars['n'] = n
    if model == 'Complete':
        pass
    elif model == 'Regular':
        npars['k'] = k
    elif model == 'ErdosRenyi' or model == 'er':
        npars['p'] = p
    elif model == 'BarabasiAlbert' or model == 'ba':
        npars['m'] = m
    elif model == 'ScaleFree' or model == 'sf':
        npars['k_min'] = k_min
        npars['k_max'] = k_max
        npars['gamma'] = gamma
    elif model == 'WattsStrogatz' or model == 'ws' or model == 'sw':
        npars['k'] = k
        npars['p'] = p
    else:
        pass
    return npars

def build_opinion_parameters_dictionary(
        active_fraction=0, 
        threshold=0, 
        zealot_fraction=0
):
    opars = {}
    opars['thr'] = threshold
    opars['acf'] = active_fraction
    opars['zef'] = zealot_fraction
    return opars

def build_epidemic_parameters_dictionary(
        seeds=0, 
        alpha=0.0, 
        beta=0.0, 
        gamma=0.0, 
        r0=0.0,
):
    epars = {}
    epars['var'] = alpha
    epars['r0'] = r0
    epars['rer'] = gamma
    #epars['seeds'] = seeds
    return epars

def build_algorithm_parameters_dictionary(
        model_id, 
        nsims_net=0, 
        nsims_dyn=0, 
        t_max=0,
):
    apars = {}
    apars['tmax'] = t_max
    if model_id == 'mc':
        apars['nsn'] = nsims_net
        apars['nsd'] = nsims_dyn
    return apars

def build_plot_parameters_dictionary(
        control1=None, 
        control2=None, 
        observable=None,
):
    ppars = {}
    ppars['control1'] = control1
    ppars['control2'] = control2
    ppars['observable'] = observable
    return ppars

def build_parameter_dictionary(
        project_id, 
        dynmod_id, 
        attribute_id, 
        exp_id, 
        model, 
        n, 
        p=None, 
        m=None, 
        n_A=0, 
        theta=0, 
        n_Z=0, 
        seeds=0, 
        alpha=0.0, 
        beta=0.0, 
        gamma=0.0, 
        r0=0.0, 
        nsims_net=0, 
        nsims_dyn=0, 
        t_max=0, 
        control1=None, 
        control2=None,
        observable=None,
        usa_id=None,
    ):

    pars = {}
    pars['header'] = \
        build_header_parameters_dictionary(
        project_id, 
        dynmod_id, 
        attribute_id, 
        exp_id, 
        usa_id
    )
    pars['network'] = build_network_parameters_dictionary(model, n, p=p, m=m)
    pars['opinion'] = build_opinion_parameters_dictionary(n_A, theta, n_Z)
    pars['epidemic'] \
        = build_epidemic_parameters_dictionary(seeds, alpha, beta, gamma, r0)
    pars['algorithm'] \
        = build_algorithm_parameters_dictionary(
        dynmod_id, 
        nsims_net, 
        nsims_dyn, 
        t_max
    )
    pars['plot'] \
        = build_plot_parameters_dictionary(control1, control2, observable)

    return pars

def build_file_name(pars, exclude_keys=[], collection=False, plot=False):
    # Get header
    header_dict = pars['header']
    header_string = build_header_string(header_dict)
    
    # Get network parameters if applies
    if pars['header']['model'] == 'mc':
        npars_dict = pars['network']
        network_string = dict_to_string(npars_dict)
    else:
        npars_dict = {}
        npars_dict['model'] = pars['network']['model']
        npars_dict['n'] = pars['network']['n']
        npars_dict['k_avg'] = pars['network']['k_avg']
        network_string = dict_to_string(npars_dict)
    
    # Get opinion model parameters
    opars_dict = {}
    opars_dict['thr'] = pars['opinion']['thr']
    opars_dict['acf'] = pars['opinion']['acf']
    opars_dict['zef'] = pars['opinion']['zef']
    for key in exclude_keys:
        if key in opars_dict:
            del opars_dict[key]
    opinion_string = dict_to_string(opars_dict)
    
    # Get epidemic model parameters
    epars_dict = {} 
    epars_dict['var'] = pars['epidemic']['var']
    epars_dict['r0'] = pars['epidemic']['r0']
    epars_dict['rer'] = pars['epidemic']['rer']
    for key in exclude_keys:
        if key in epars_dict:
            del epars_dict[key]
    epidemic_string = dict_to_string(epars_dict)
    
    # Get algorithm parameters
    if pars['header']['model'] == 'mc':
        apars_dict = pars['algorithm']
        algorithm_string = dict_to_string(apars_dict)
    else:
        algorithm_string = 'tmax' + str(pars['algorithm']['tmax'])
    
    # Get plot parameters
    if (pars['header']['exp_id'] == 1 and collection):
        ppars_dict = {}
        if plot:
            ppars_dict['observable'] = pars['plot']['observable']
        ppars_dict['control1'] = pars['plot']['control1']
        ppars_dict['control2'] = pars['plot']['control2']
        for key in exclude_keys:
            if key in ppars_dict:
                del ppars_dict[key]
        plot_string = dict_to_string(ppars_dict)

        strings = [header_string, 
                   plot_string, network_string, 
                   opinion_string, epidemic_string, 
                   algorithm_string]
    elif (pars['header']['exp_id'] == 2 and collection):
        strings = [header_string, 
                   network_string, opinion_string, 
                   epidemic_string, algorithm_string]
    else:
        strings = [header_string, network_string, 
                   opinion_string, epidemic_string, 
                   algorithm_string]
    file_name = strings[0]
    for s in strings[1:]:
        file_name += f"_{s}"

    return file_name

def write_file_name(
        pars, 
        exclude_category=None, 
        exclude_keys=None,
):
    flat_dict = {}
    
    for category, sub_dict in pars.items():
        if exclude_category and category == exclude_category:
            continue
        for key, value in sub_dict.items():
            if exclude_keys and key in exclude_keys:
                continue
            flat_dict[key] = value
    # Create a list of key-value pairs in the form 'key_value'
    key_value_pairs = [f"{key}{value}" for key, value in flat_dict.items()]
    # Join the key-value pairs together with an underscore separator
    #header = project_id + '_' + attribute_id + '_' + exp_id + '_'
    file_name = "_".join(key_value_pairs)
    
    return file_name

def heatmap_criteria(data, pars):
    if data['pars']['network']['n'] != pars['network']['n']:
        return False
    if data['pars']['network']['model'] != pars['network']['model']:
        return False
    if data['pars']['epidemic']['r0'] != pars['epidemic']['r0']:
        return False
    if data['pars']['epidemic']['infection_decay'] != pars['epidemic']['rer']:
        return False
    if data['pars']['epidemic']['vaccination_rate'] != pars['epidemic']['var']:
        return False
    if data['pars']['algorithm']['nsims_net'] != pars['algorithm']['nsn']:
        return False
    if data['pars']['algorithm']['nsims_dyn'] != pars['algorithm']['nsd']:
        return False
    
    return True

def collect_pickles_for_mc_heatmap(pars):
    lower_path = 'results'
    input_path = os.path.join(path, lower_path)
    header = build_header_string(pars['header'])
    exclude_str = 'control'
    file_paths = [os.path.join(input_path, f) 
                  for f in os.listdir(input_path) 
                  if f.startswith(header) 
                  and f.endswith('.pickle') 
                  and exclude_str not in f]
    
    results_dict = {}
    
    for file_path in file_paths:
        with open(file_path, 'rb') as pickle_file:
            data = pk.load(pickle_file)
        if heatmap_criteria(data, pars):
            con1_label = get_full_label(pars['plot']['control1'])
            con2_label = get_full_label(pars['plot']['control2'])
            par1 = data['pars']['opinion'][con1_label]
            par2 = data['pars']['opinion'][con2_label]
            if (par1, par2) not in results_dict:
                results_dict[(par1, par2)] = {}
            results_dict[(par1, par2)]['prevalence'] \
                = data['global']['prevalence']
            results_dict[(par1, par2)]['peak_incidence'] \
                = data['global']['peak_incidence']
            results_dict[(par1, par2)]['peak_time'] \
                = data['global']['peak_time']
            results_dict[(par1, par2)]['vaccinated'] \
                = data['global']['vaccinated']
            results_dict[(par1, par2)]['convinced'] \
                = data['global']['convinced']
            results_dict[(par1, par2)]['end_time'] \
                = data['global']['end_time']

    # Build output file name
    con1_id = pars['plot']['control1']
    con2_id = pars['plot']['control2']
    file_name = build_file_name(pars, exclude_keys=[con1_id, con2_id], collection=True)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    pk.dump(results_dict, open(full_path, "wb"))

def us_scatter_collection_criteria(data, pars):
    if 'pars' not in data.keys():
        return False
    if data['pars']['network']['n'] != pars['network']['n']:
        return False
    if data['pars']['network']['model'] != pars['network']['model']:
        return False
    if data['pars']['epidemic']['r0'] != pars['epidemic']['r0']:
        return False
    if data['pars']['epidemic']['infection_decay'] != pars['epidemic']['rer']:
        return False
    if data['pars']['epidemic']['vaccination_rate'] != pars['epidemic']['var']:
        return False
    if data['pars']['algorithm']['nsims_net'] != pars['algorithm']['nsn']:
        return False
    if data['pars']['algorithm']['nsims_dyn'] != pars['algorithm']['nsd']:
        return False

    return True

def us_vaccination_collection_criteria(data, pars):
    if 'pars' not in data.keys():
        return False
    if data['pars']['network']['n'] != pars['network']['n']:
        return False
    if data['pars']['network']['model'] != pars['network']['model']:
        return False
    if data['pars']['epidemic']['r0'] != pars['epidemic']['r0']:
        return False
    if data['pars']['epidemic']['infection_decay'] != pars['epidemic']['rer']:
        return False
    if data['pars']['algorithm']['nsims_net'] != pars['algorithm']['nsn']:
        return False
    if data['pars']['algorithm']['nsims_dyn'] != pars['algorithm']['nsd']:
        return False

    return True

def collect_pickles_for_us_states_scatter(pars):
    lower_path = 'results'
    input_path = os.path.join(path, lower_path)
    header = build_header_string(pars['header'])
    exclude_str = 'collection'
    file_paths = [os.path.join(input_path, f) 
                  for f in os.listdir(input_path) 
                  if f.startswith(header) 
                  and f.endswith('.pickle') 
                  and exclude_str not in f]
    
    results_dict = {}
    
    for file_path in file_paths:
        with open(file_path, 'rb') as pickle_file:
            data = pk.load(pickle_file)

            if us_scatter_collection_criteria(data, pars):

                usa_id = data['pars']['vaccination']['state']
                results_dict[usa_id] = {'w1': {}, 'w2': {}} 

                if pars['header']['attribute'] == 'global':
            
                    results_dict[usa_id]['w1']['prevalence'] = data['global_w1']['prevalence']
                    results_dict[usa_id]['w1']['peak_incidence'] = data['global_w1']['peak_incidence']
                    results_dict[usa_id]['w1']['peak_time'] = data['global_w1']['peak_time']
                    results_dict[usa_id]['w1']['vaccinated'] = data['global_w1']['vaccinated']
                    results_dict[usa_id]['w1']['convinced'] = data['global_w1']['convinced']
                    results_dict[usa_id]['w1']['end_time'] = data['global_w1']['end_time']

                    results_dict[usa_id]['w2']['prevalence'] = data['global_w2']['prevalence']
                    results_dict[usa_id]['w2']['peak_incidence'] = data['global_w2']['peak_incidence']
                    results_dict[usa_id]['w2']['peak_time'] = data['global_w2']['peak_time']
                    results_dict[usa_id]['w2']['vaccinated'] = data['global_w2']['vaccinated']
                    results_dict[usa_id]['w2']['convinced'] = data['global_w2']['convinced']
                    results_dict[usa_id]['w2']['end_time'] = data['global_w2']['end_time']
            
                elif pars['header']['attribute'] == 'time':

                    results_dict[usa_id]['w1']['time'] = data['time_w1']['t_array']
                    results_dict[usa_id]['w1']['as'] = data['time_w1']['as_pop_st']
                    results_dict[usa_id]['w1']['hs'] = data['time_w1']['hs_pop_st']
                    results_dict[usa_id]['w1']['ai'] = data['time_w1']['ai_pop_st']
                    results_dict[usa_id]['w1']['hi'] = data['time_w1']['hi_pop_st']
                    results_dict[usa_id]['w1']['ar'] = data['time_w1']['ar_pop_st']
                    results_dict[usa_id]['w1']['hr'] = data['time_w1']['hr_pop_st']
                    results_dict[usa_id]['w1']['av'] = data['time_w1']['av_pop_st']

                    results_dict[usa_id]['w2']['time'] = data['time_w2']['t_array']
                    results_dict[usa_id]['w2']['as'] = data['time_w2']['as_pop_st']
                    results_dict[usa_id]['w2']['hs'] = data['time_w2']['hs_pop_st']
                    results_dict[usa_id]['w2']['ai'] = data['time_w2']['ai_pop_st']
                    results_dict[usa_id]['w2']['hi'] = data['time_w2']['hi_pop_st']
                    results_dict[usa_id]['w2']['ar'] = data['time_w2']['ar_pop_st']
                    results_dict[usa_id]['w2']['hr'] = data['time_w2']['hr_pop_st']
                    results_dict[usa_id]['w2']['av'] = data['time_w2']['av_pop_st']

    # Build output file name
    file_name = build_file_name(pars, collection=True)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    pk.dump(results_dict, open(full_path, "wb"))


def collect_pickles_for_us_states_vaccination(pars):  
    lower_path = 'results'
    input_path = os.path.join(path, lower_path)
    header = build_header_string(pars['header'])
    exclude_str = 'control'
    file_paths = [os.path.join(input_path, f) 
                  for f in os.listdir(input_path) 
                  if f.startswith(header) 
                  and f.endswith('.pickle') 
                  and exclude_str not in f]
    
    results_dict = {}
    
    for file_path in file_paths:
        with open(file_path, 'rb') as pickle_file:
            data = pk.load(pickle_file)

            if us_vaccination_collection_criteria(data, pars):

                usa_id = data['pars']['vaccination']['state']
                alpha = data['pars']['epidemic']['vaccination_rate']

                if usa_id not in results_dict:
                    results_dict[usa_id] = {}

                if alpha not in results_dict[usa_id]:
                    results_dict[usa_id][alpha] = {'w1': {}, 'w2': {}}

                if pars['header']['attribute'] == 'global':
            
                    results_dict[usa_id][alpha]['w1']['prevalence'] = data['global_w1']['prevalence']
                    results_dict[usa_id][alpha]['w1']['peak_incidence'] = data['global_w1']['peak_incidence']
                    results_dict[usa_id][alpha]['w1']['peak_time'] = data['global_w1']['peak_time']
                    results_dict[usa_id][alpha]['w1']['vaccinated'] = data['global_w1']['vaccinated']
                    results_dict[usa_id][alpha]['w1']['convinced'] = data['global_w1']['convinced']
                    results_dict[usa_id][alpha]['w1']['end_time'] = data['global_w1']['end_time']

                    results_dict[usa_id][alpha]['w2']['prevalence'] = data['global_w2']['prevalence']
                    results_dict[usa_id][alpha]['w2']['peak_incidence'] = data['global_w2']['peak_incidence']
                    results_dict[usa_id][alpha]['w2']['peak_time'] = data['global_w2']['peak_time']
                    results_dict[usa_id][alpha]['w2']['vaccinated'] = data['global_w2']['vaccinated']
                    results_dict[usa_id][alpha]['w2']['convinced'] = data['global_w2']['convinced']
                    results_dict[usa_id][alpha]['w2']['end_time'] = data['global_w2']['end_time']

    # Build output file name
    file_name = build_file_name(pars, exclude_keys=['var'], collection=True)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    pk.dump(results_dict, open(full_path, "wb"))

def get_state_code(state_name):
    state_dict = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "DistrictOfColumbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "NewHampshire": "NH",
        "NewJersey": "NJ",
        "NewMexico": "NM",
        "NewYork": "NY",
        "NorthCarolina": "NC",
        "NorthDakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "RhodeIsland": "RI",
        "SouthCarolina": "SC",
        "SouthDakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "WestVirginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "National": "US"
    }
    return state_dict.get(state_name, "Invalid state name")

import json

def read_vaccination_data():
    lower_path = 'data'
    full_path = os.path.join(path, lower_path, "vaccination_data.json")
    with open(full_path, 'r') as f:
        data = json.load(f)

    results = {}
    for state, values in data.items():
        results[state] = {
            "already": values[0],
            "soon": values[1],
            "someone": values[2],
            "majority": values[3],
            "never": values[4]
        }

    return results

def compute_active_fraction(state_thresholds):
    return state_thresholds['already']

def compute_active_soon_fraction(state_thresholds):
    return state_thresholds['already'] + state_thresholds['soon']

def compute_active_soon_fast_fraction(state_thresholds):
    return state_thresholds['already'] \
        + state_thresholds['soon'] \
            + state_thresholds['someone']

def compute_average_theshold(state_thresholds, pars):
    n = pars['network']['n']
    p = pars['network']['p']
    k_avg = p * (n - 1)
    theta_someone = 1.0 / k_avg
    theta_majority = 0.5

    return theta_someone * state_thresholds['someone'] \
        + theta_majority * state_thresholds['majority']

def build_scatter_results_dataframe(pars):
    # Load the vaccination data
    vaccination_results = read_vaccination_data()

    # Extract the vaccination data into numpy arrays
    nentries = len(vaccination_results.keys())
    state_code = [get_state_code(state) for state in vaccination_results.keys()]
    already = np.zeros(nentries)
    soon = np.zeros(nentries)
    someone = np.zeros(nentries)
    majority = np.zeros(nentries)
    never = np.zeros(nentries)
    for state, i in zip(vaccination_results.keys(), range(nentries)):
        already[i] = vaccination_results[state]['already']
        soon[i] = vaccination_results[state]['soon']
        someone[i] = vaccination_results[state]['someone']
        majority[i] = vaccination_results[state]['majority']
        never[i] = vaccination_results[state]['never']
    
    # Check if the unified pickle exists, and generate it if not
    base_name = build_file_name(pars, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        collect_pickles_for_us_states_scatter(pars)

    # Load results
    state_raw_dict = pk.load(open(full_name, "rb"))

    # Collect filtered results
    N = pars['network']['n']
    prevalence_cutoff = 0.025
    state_stats_dict = {}
    avg_pre_w1_list = []
    l95_pre_w1_list = []
    u95_pre_w1_list = []
    avg_con_w1_list = []
    l95_con_w1_list = []
    u95_con_w1_list = []
    avg_vac_w1_list = []
    l95_vac_w1_list = []
    u95_vac_w1_list = []
    avg_ent_w1_list = []
    l95_ent_w1_list = []
    u95_ent_w1_list = []
    avg_pre_w2_list = []
    l95_pre_w2_list = []
    u95_pre_w2_list = []
    avg_con_w2_list = []
    l95_con_w2_list = []
    u95_con_w2_list = []
    avg_vac_w2_list = []
    l95_vac_w2_list = []
    u95_vac_w2_list = []
    avg_ent_w2_list = []
    l95_ent_w2_list = []
    u95_ent_w2_list = []
    state_code_list = []

    for state in vaccination_results.keys():
        print("{0}".format(state))

        state_code_list.append(get_state_code(state))
        pre_w1_dist = np.array(state_raw_dict[state]['w1']['prevalence']) / N
        con_w1_dist = np.array(state_raw_dict[state]['w1']['convinced']) / N
        vac_w1_dist = np.array(state_raw_dict[state]['w1']['vaccinated']) / N
        ent_w1_dist = np.array(state_raw_dict[state]['w1']['end_time'])

        fil_pre_w1_dist = pre_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_pre_w1_dist.size == 0:
            fil_pre_w1_dist = pre_w1_dist
        fil_con_w1_dist = con_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_con_w1_dist.size == 0:
            fil_con_w1_dist = con_w1_dist
        fil_vac_w1_dist = vac_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_vac_w1_dist.size == 0:
            fil_vac_w1_dist = vac_w1_dist
        fil_ent_w1_dist = ent_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_ent_w1_dist.size == 0:
            fil_ent_w1_dist = ent_w1_dist

        pre_w1_stat_dict = an.compute_distribution_statistics(fil_pre_w1_dist)
        con_w1_stat_dict = an.compute_distribution_statistics(fil_con_w1_dist)
        vac_w1_stat_dict = an.compute_distribution_statistics(fil_vac_w1_dist)
        ent_w1_stat_dict = an.compute_distribution_statistics(fil_ent_w1_dist)
        
        state_stats_dict = {state: {'w1': {}, 'w2': {}}}
        state_stats_dict[state]['w1']['prevalence'] = pre_w1_stat_dict
        state_stats_dict[state]['w1']['convinced'] = con_w1_stat_dict
        state_stats_dict[state]['w1']['vaccinated'] = vac_w1_stat_dict
        state_stats_dict[state]['w1']['end_time'] = ent_w1_stat_dict

        avg_pre_w1_list.append(pre_w1_stat_dict['avg'])
        l95_pre_w1_list.append(pre_w1_stat_dict['l95'])
        u95_pre_w1_list.append(pre_w1_stat_dict['u95'])
        avg_con_w1_list.append(con_w1_stat_dict['avg'])
        l95_con_w1_list.append(con_w1_stat_dict['l95'])
        u95_con_w1_list.append(con_w1_stat_dict['u95'])
        avg_vac_w1_list.append(vac_w1_stat_dict['avg'])
        l95_vac_w1_list.append(vac_w1_stat_dict['l95'])
        u95_vac_w1_list.append(vac_w1_stat_dict['u95'])
        avg_ent_w1_list.append(ent_w1_stat_dict['avg'])
        l95_ent_w1_list.append(ent_w1_stat_dict['l95'])
        u95_ent_w1_list.append(ent_w1_stat_dict['u95'])

        pre_w2_dist = np.array(state_raw_dict[state]['w2']['prevalence']) / N
        con_w2_dist = np.array(state_raw_dict[state]['w2']['convinced']) / N
        vac_w2_dist = np.array(state_raw_dict[state]['w2']['vaccinated']) / N
        ent_w2_dist = np.array(state_raw_dict[state]['w2']['end_time'])

        fil_pre_w2_dist = pre_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_pre_w2_dist.size == 0:
            fil_pre_w2_dist = pre_w2_dist
        fil_con_w2_dist = con_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_con_w2_dist.size == 0:
            fil_con_w2_dist = con_w2_dist
        fil_vac_w2_dist = vac_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_vac_w2_dist.size == 0:
            fil_vac_w2_dist = vac_w2_dist
        fil_ent_w2_dist = ent_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_ent_w2_dist.size == 0:
            fil_ent_w2_dist = ent_w2_dist

        pre_w2_stat_dict = an.compute_distribution_statistics(fil_pre_w2_dist)
        con_w2_stat_dict = an.compute_distribution_statistics(fil_con_w2_dist)
        vac_w2_stat_dict = an.compute_distribution_statistics(fil_vac_w2_dist)
        ent_w2_stat_dict = an.compute_distribution_statistics(fil_ent_w2_dist)

        state_stats_dict[state]['w2']['prevalence'] = pre_w2_stat_dict
        state_stats_dict[state]['w2']['convinced'] = con_w2_stat_dict
        state_stats_dict[state]['w2']['vaccinated'] = vac_w2_stat_dict
        state_stats_dict[state]['w2']['end_time'] = ent_w2_stat_dict

        avg_pre_w2_list.append(pre_w2_stat_dict['avg'])
        l95_pre_w2_list.append(pre_w2_stat_dict['l95'])
        u95_pre_w2_list.append(pre_w2_stat_dict['u95'])
        avg_con_w2_list.append(con_w2_stat_dict['avg'])
        l95_con_w2_list.append(con_w2_stat_dict['l95'])
        u95_con_w2_list.append(con_w2_stat_dict['u95'])
        avg_vac_w2_list.append(vac_w2_stat_dict['avg'])
        l95_vac_w2_list.append(vac_w2_stat_dict['l95'])
        u95_vac_w2_list.append(vac_w2_stat_dict['u95'])
        avg_ent_w2_list.append(ent_w2_stat_dict['avg'])
        l95_ent_w2_list.append(ent_w2_stat_dict['l95'])
        u95_ent_w2_list.append(ent_w2_stat_dict['u95'])    

    # Combine the vaccination data and simulation data into a pandas DataFrame
    data = pd.DataFrame({
        'state_code': state_code,
        'already': already,
        'soon': soon,
        'someone': someone,
        'majority': majority,
        'never': never,
        'avg_pre_w1': avg_pre_w1_list,
        'l95_pre_w1': l95_pre_w1_list,
        'u95_pre_w1': u95_pre_w1_list,
        'avg_con_w1': avg_con_w1_list,
        'l95_con_w1': l95_con_w1_list,
        'u95_con_w1': u95_con_w1_list,
        'avg_vac_w1': avg_vac_w1_list,
        'l95_vac_w1': l95_vac_w1_list,
        'u95_vac_w1': u95_vac_w1_list,
        'avg_pre_w2': avg_pre_w2_list,
        'l95_pre_w2': l95_pre_w2_list,
        'u95_pre_w2': u95_pre_w2_list,
        'avg_con_w2': avg_con_w2_list,
        'l95_con_w2': l95_con_w2_list,
        'u95_con_w2': u95_con_w2_list,
        'avg_vac_w2': avg_vac_w2_list,
        'l95_vac_w2': l95_vac_w2_list,
        'u95_vac_w2': u95_vac_w2_list
    })

    # Save the data as a CSV file
    data.to_csv('vaccination_and_simulation_data.csv', index=False)