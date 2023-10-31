import os
import re
import json
import subprocess
import numpy as np
import pandas as pd
import pickle as pk
from collections import Counter

import analysis as an


def read_data_from_files(folders, file_name, extension):
    for folder in folders:
        file_path = os.path.join(folder, file_name + extension)
        if os.path.exists(file_path):
            if extension == ".json":
                with open(file_path) as json_file:
                    data = json.load(json_file)
                    return data
            elif extension == ".pickle":
                with open(file_path, "rb") as pickle_file:
                    data = pk.load(pickle_file)
                    return data
    return None

def build_path(folders):
    return os.path.join(*folders)

def build_full_path(folders, file_name, extension):
    full_path = os.path.join(*folders, file_name + extension)
    return full_path

def read_json_file(fullname):
    if not fullname.endswith('.json'):
        fullname += '.json'
    with open(fullname) as json_file:
        data = json.load(json_file)
        return data

def read_pickle_file(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as pickle_file:
        data = pk.load(pickle_file)
    return data

def build_dict_from_config_file(full_path):
    dictionary = read_json_file(full_path)
    return dictionary

def build_network_dict_from_config_file(full_path):
    network_dict = read_json_file(full_path)
    return network_dict
    
def build_opinion_dict_from_config_file(full_path):
    opinion_dict = read_json_file(full_path)
    return opinion_dict

def build_epidemic_dict_from_config_file(full_path):
    epidemic_dict = read_json_file(full_path)
    return epidemic_dict

def build_vaccination_dict_from_config_file(full_path):
    vaccination_dict = read_json_file(full_path)
    return vaccination_dict

def build_algorithm_dict_from_config_file(full_path):
    algorithm_dict = read_json_file(full_path)
    return algorithm_dict

def convert_dict_to_seamless_strings(input_dict):
    seamless_strings = []

    for key, value in input_dict.items():
        seamless_string = str(key) + str(value)
        seamless_strings.append(seamless_string)

    return seamless_strings

def dict_to_string(dictionary):
    return '_'.join([f"{key}{value}" for key, value in dictionary.items()])

def find_files_with_string(string, folder_path):
    file_paths = []
    for file_name in os.listdir(folder_path):
        if string in file_name:
            full_path = os.path.join(folder_path, file_name)
            file_paths.append(full_path)
    return file_paths

def trim_file_extension(file_string):
    extensions = ['.pickle', '.pdf', '.png', '.txt', '.csv']
    
    for ext in extensions:
        if file_string.endswith(ext):
            file_string = file_string[:-(len(ext))]
            break
    
    return file_string

def trim_file_path(file_path):
    sequences = ['global', 'clusters', 'csp', 'cd', 'agents', 'asp', 'ad', 'time']
    
    for sequence in sequences:
        if sequence in file_path:
            start_index = file_path.index(sequence)
            file_path = file_path[start_index:]
            break
    
    return file_path

def generate_filename(params):
    # Define the mapping of keys to substrings
    key_mapping = {
        'net': {'er': 'neter', 'ba': 'netbar'},
        'k': lambda val: f'k{val}',
        'acf': lambda val: f'acf{val}',
        'thr': lambda val: f'thr{val:.2f}',
        'ze': lambda val: f'zef{val:.1f}',
        'r': lambda val: f'r{val:.1f}',
        'rer': lambda val: f'rer{val:.1f}',
        'var': lambda val: f'var{val:.3f}',
        'nsd': lambda val: f'nsd{val}',
        'nsn': lambda val: f'nsn{val}',
        'tmax': lambda val: f'tmax{val}',
    }

    # Define default values for keys
    default_values = {
        'net': 'er',
        'k': 10,
        'acf': 0,
        'thr': 0.45,
        'ze': 0.1,
        'r': 1.5,
        'rer': 0.2,
        'var': 0.005,
        'nsd': 25,
        'nsn': 25,
        'tmax': 500,
    }

    # Initialize an empty list to store the substrings
    substrings = []

    # Loop through the parameter dictionary and construct substrings
    for key in key_mapping.keys():
        value = params.get(key, default_values.get(key))
        if key in key_mapping:
            if isinstance(key_mapping[key], dict):
                # Handle keys with a predefined mapping
                if value in key_mapping[key]:
                    substrings.append(key_mapping[key][value])
            elif callable(key_mapping[key]):
                # Handle keys with a custom formatting function
                substrings.append(key_mapping[key](value))

    # Convert the substrings to strings and concatenate them
    filename = '_'.join(map(str, substrings)) + '.pickle'

    return filename

def build_header_string(header_dictionary):
    return '_'.join([f"{value}" for key, value in header_dictionary.items()])

def build_network_dictionary(
        NetworkModel,
        network_size,
        k=None, 
        p=None, 
        m=None, 
        k_min=None,
        k_max=None, 
        gamma=None,
    ):

    npars = {}
    npars['model'] = NetworkModel
    npars['n'] = network_size
    if NetworkModel == 'Complete':
        pass
    elif NetworkModel == 'Regular':
        npars['k'] = k
    elif NetworkModel == 'ErdosRenyi' or NetworkModel == 'er':
        npars['p'] = p
    elif NetworkModel == 'BarabasiAlbert' or NetworkModel == 'ba':
        npars['m'] = m
    elif NetworkModel == 'ScaleFree' or NetworkModel == 'sf':
        npars['k_min'] = k_min
        npars['k_max'] = k_max
        npars['gamma'] = gamma
    elif NetworkModel == 'WattsStrogatz' or NetworkModel == 'ws' or NetworkModel == 'sw':
        npars['k'] = k
        npars['p'] = p
    else:
        pass
    return npars

def build_opinion_dictionary(
        active_fraction=0.0,
        threshold=0,
        zealot_fraction=0,
    ):
    opars = {}
    opars['acf'] = active_fraction
    opars['thr'] = threshold
    opars['zef'] = zealot_fraction
    return opars

def build_epidemic_dictionary(
        alpha=0.0, 
        gamma=0.0,
        r0=0.0,
):
    epars = {}
    epars['r0'] = r0
    epars['rer'] = gamma
    epars['var'] = alpha
    return epars

def build_algorithm_parameters_dictionary(
        nsims_dyn=0,
        nsims_net=0,  
        t_max=0,
):
    apars = {}
    apars['nsd'] = nsims_dyn
    apars['nsn'] = nsims_net
    apars['tmax'] = t_max
    return apars

def purge_dict(dictionary, *keys_to_remove):
    keys_to_remove = set(keys_to_remove)
    return {key: value for key, value in dictionary.items() if key not in keys_to_remove}

def purge_string(string, *str_to_remove):
    result = ""
    remove_next = False

    for i in range(len(string)):
        if string[i] == '_':
            remove_next = False
            result += string[i]
        elif remove_next:
            continue
        elif string[i:i + len(str_to_remove[0])] == str_to_remove[0]:
            remove_next = True
            continue
        else:
            result += string[i]

    return result

def collect_pickle_filenames(fullpath, header, string_segments=None):
    # Get the list of files in the directory
    file_list = os.listdir(fullpath)

    # Filter files based on header and string segments
    result = []
    for file_name in file_list:
        if file_name.startswith(header) and (string_segments is None or all(segment in file_name for segment in string_segments)):
            result.append(file_name)

    return result

def collect_pickle_filenames_by_exclusion(fullpath, header, string_segment):
    # Get the list of files in the directory
    file_list = os.listdir(fullpath)

    # Filter files based on header and string segment
    result = []
    for file_name in file_list:
        if file_name.startswith(header) and (string_segment is None or not any(segment in file_name for segment in string_segment)):
            result.append(file_name)

    return result

def collect_pickle_filenames_by_inclusion(fullpath, header, segment_list):
    # Get the list of files in the directory
    file_list = os.listdir(fullpath)

    # Filter files based on header and segment list
    result = []
    for file_name in file_list:
        if file_name.startswith(header):
            included_all_segments = all(segment in file_name for segment in segment_list)
            if included_all_segments:
                result.append(file_name)

    return result

def extract_parameters_from_string(input_string, substrings_to_find):
    extracted_values = {}

    for substring in substrings_to_find:
        if substring in input_string:
            value_start_index = input_string.index(substring) + len(substring)
            value = input_string[value_start_index:].split('_', 1)[0]
            extracted_values[substring] = value

    return extracted_values

def filter_global_output(global_output_dict, n, prevalence_cutoff):

    norm_prev_dist = np.array(global_output_dict['prevalence_dist']) / n
    norm_vacc_dist = np.array(global_output_dict['vaccinated_dist']) / n
    #norm_vape_dist = np.array(global_output_dict['vaccinated_at_peak_dist']) / n
    norm_conv_dist = np.array(global_output_dict['convinced_dist']) / n
    #norm_cope_dist = np.array(global_output_dict['convinced_at_peak_dist']) / n
    norm_pein_dist = np.array(global_output_dict['peak_incidence_dist']) / n
    norm_peti_dist = np.array(global_output_dict['time_to_peak_dist'])
    norm_enti_dist = np.array(global_output_dict['time_to_end_dist'])
    
    failed_outbreaks = np.where(norm_prev_dist < prevalence_cutoff)[0]
    
    filt_norm_prev_dist = np.delete(norm_prev_dist, failed_outbreaks)
    filt_norm_vacc_dist = np.delete(norm_vacc_dist, failed_outbreaks)
    #filt_norm_vape_dist = np.delete(norm_vape_dist, failed_outbreaks)
    filt_norm_conv_dist = np.delete(norm_conv_dist, failed_outbreaks)
    #filt_norm_cope_dist = np.delete(norm_cope_dist, failed_outbreaks)
    filt_norm_pein_dist = np.delete(norm_pein_dist, failed_outbreaks)
    filt_norm_peti_dist = np.delete(norm_peti_dist, failed_outbreaks)
    filt_norm_enti_dist = np.delete(norm_enti_dist, failed_outbreaks)

    if len(filt_norm_prev_dist) == 0:
        filt_norm_prev_dist = norm_prev_dist
    if len(filt_norm_vacc_dist) == 0:
        filt_norm_vacc_dist = norm_vacc_dist
    #if len(filt_norm_vape_dist) == 0:
    #    filt_norm_vape_dist = norm_vape_dist
    if len(filt_norm_conv_dist) == 0:
        filt_norm_conv_dist = norm_conv_dist
    #if len(filt_norm_cope_dist) == 0:
    #    filt_norm_cope_dist = norm_cope_dist
    if len(filt_norm_pein_dist) == 0:
        filt_norm_pein_dist = norm_pein_dist
    if len(filt_norm_peti_dist) == 0:
        filt_norm_peti_dist = norm_peti_dist
    if len(filt_norm_enti_dist) == 0:
        filt_norm_enti_dist = norm_enti_dist

    output = {}
    output['prevalence_dist'] = norm_prev_dist
    output['vaccinated_dist'] = norm_vacc_dist
    #output['vaccinated_at_peak_dist'] = norm_vape_dist
    output['convinced_dist'] = norm_conv_dist
    #output['convinced_at_peak_dist'] = norm_cope_dist
    output['peak_incidence_dist'] = norm_pein_dist
    output['time_to_peak_dist'] = norm_peti_dist
    output['time_to_end_dist'] = norm_enti_dist

    return output

def load_raw_agent_output(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        agent_dict = pk.load(input_data)

    output_dict = {}

    output_dict['convinced_when'] = agent_dict['convinced_when']
    output_dict['degree'] = agent_dict['degree']
    output_dict['final_act_sus'] = agent_dict['final_active_susceptible']
    output_dict['final_prevalence'] = agent_dict['final_prevalence']
    output_dict['final_vaccinated'] = agent_dict['final_vaccinated']
    output_dict['id'] = agent_dict['id']
    output_dict['infected_by'] = agent_dict['infected_by']
    output_dict['infected_when'] = agent_dict['infected_when']
    output_dict['initial_active_susceptible'] = agent_dict['initial_active_susceptible']
    output_dict['initial_vaccinated'] = agent_dict['initial_vaccinated']
    output_dict['removed_when'] = agent_dict['removed_when']
    output_dict['status'] = agent_dict['status']
    output_dict['vaccinated_when'] = agent_dict['vaccinated_when']
    output_dict['zealots'] = agent_dict['zealots']

    return output_dict

def load_raw_cluster_output(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        cluster_dict = pk.load(input_data)

    output_dict = {}

    output_dict['ai'] = cluster_dict['ai_cluster']
    output_dict['ar'] = cluster_dict['ar_cluster']
    output_dict['as'] = cluster_dict['as_cluster']
    output_dict['av'] = cluster_dict['av_cluster']
    output_dict['hi'] = cluster_dict['hi_cluster']
    output_dict['hr'] = cluster_dict['hr_cluster']
    output_dict['hs'] = cluster_dict['hs_cluster']
    output_dict['hv'] = cluster_dict['hv_cluster']
    output_dict['ze'] = cluster_dict['ze_cluster']

    return output_dict

def load_global_output(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        global_dict = pk.load(input_data)['global']

    output_dict = {}

    output_dict['prevalence_dist'] = global_dict['prevalence']
    output_dict['peak_incidence_dist'] = global_dict['peak_incidence']
    output_dict['time_to_peak_dist'] = global_dict['time_to_peak']
    output_dict['vaccinated_dist'] = global_dict['vaccinated']
    #output_dict['vaccinated_at_peak_dist'] = global_dict['vaccinated_at_peak']
    output_dict['convinced_dist'] = global_dict['convinced']
    #output_dict['convinced_at_peak_dist'] = global_dict['convinced_at_peak']
    output_dict['time_to_end_dist'] = global_dict['time_to_end']

    return output_dict


def load_us_output(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        global_dict = pk.load(input_data)

    output_dict = {'w1': {}, 'w2': {}}

    output_dict['w1']['prevalence_dist'] = global_dict['global_w1']['prevalence']
    output_dict['w1']['peak_incidence_dist'] = global_dict['global_w1']['peak_incidence']
    output_dict['w1']['time_to_peak_dist'] = global_dict['global_w1']['time_to_peak']
    output_dict['w1']['vaccinated_dist'] = global_dict['global_w1']['vaccinated']
    output_dict['w1']['vaccinated_at_peak_dist'] = global_dict['global_w1']['vaccinated_at_peak']
    output_dict['w1']['convinced_dist'] = global_dict['global_w1']['convinced']
    output_dict['w1']['convinced_at_peak_dist'] = global_dict['global_w1']['convinced_at_peak']
    output_dict['w1']['time_to_end_dist'] = global_dict['global_w1']['time_to_end']

    output_dict['w2']['prevalence_dist'] = global_dict['global_w2']['prevalence']
    output_dict['w2']['peak_incidence_dist'] = global_dict['global_w2']['peak_incidence']
    output_dict['w2']['time_to_peak_dist'] = global_dict['global_w2']['time_to_peak']
    output_dict['w2']['vaccinated_dist'] = global_dict['global_w2']['vaccinated']
    output_dict['w2']['vaccinated_at_peak_dist'] = global_dict['global_w2']['vaccinated_at_peak']
    output_dict['w2']['convinced_dist'] = global_dict['global_w2']['convinced']
    output_dict['w2']['convinced_at_peak_dist'] = global_dict['global_w2']['convinced_at_peak']
    output_dict['w2']['time_to_end_dist'] = global_dict['global_w2']['time_to_end']

    return output_dict


def stat_global_output(global_output_dict):

    prev_dist = global_output_dict['prevalence_dist']
    vacc_dist = global_output_dict['vaccinated_dist']
    #vape_dist = global_output_dict['vaccinated_at_peak_dist']
    conv_dist = global_output_dict['convinced_dist']
    #cope_dist = global_output_dict['convinced_at_peak_dist']
    pein_dist = global_output_dict['peak_incidence_dist']
    peti_dist = global_output_dict['time_to_peak_dist']
    enti_dist = global_output_dict['time_to_end_dist']

    prev_stats = an.compute_distribution_statistics(prev_dist)
    vacc_stats = an.compute_distribution_statistics(vacc_dist)
    #vape_stats = an.compute_distribution_statistics(vape_dist)
    conv_stats = an.compute_distribution_statistics(conv_dist)
    #cope_stats = an.compute_distribution_statistics(cope_dist)
    pein_stats = an.compute_distribution_statistics(pein_dist)
    peti_stats = an.compute_distribution_statistics(peti_dist)
    enti_stats = an.compute_distribution_statistics(enti_dist)

    output = {}
    output['prevalence'] = prev_stats
    output['vaccinated'] = vacc_stats
    #output['vaccinated_at_peak'] = vape_stats
    output['convinced'] = conv_stats
    #output['convinced_at_peak'] = cope_stats
    output['peak_incidence'] = pein_stats
    output['time_to_peak'] = peti_stats
    output['time_to_end'] = enti_stats

    return output

def modify_json_file(file_path, key, value):
    with open(file_path, 'r') as file:
        data = json.load(file)
        data[key] = value

    with open(file_path, 'w') as file:
        json.dump(data, file)

def call_rust_file(file_path):
    command = ['cargo', 'run', '-r']
    subprocess.run(command, cwd=file_path)

def save_to_pickle(object, fullpath):
    with open(fullpath, 'wb') as f:
        pk.dump(object, f)

def open_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.pickle'):
        with open(file_path, 'rb') as f:
            df = pk.load(f)
    else:
        raise ValueError("Unsupported file extension. Only .csv and .pickle files are supported.")

    return df

def collect_agent_times_by_degree(
        ids, 
        degree, 
        convinced_when, 
        infected_when, 
        removed_when, 
        vaccinated_when,
        ):
    nsims = len(ids)
    nagents = len(ids[0])

    times_k_dict = {}

    for sim in range(nsims):
        for i in range(nagents):
            agent_degree = degree[sim][i]
            
            # Create a sub-dictionary for the degree if it doesn't exist
            if agent_degree not in times_k_dict:
                times_k_dict[agent_degree] = {
                    'convinced': [],
                    'infected': [],
                    'removed': [],
                    'vaccinated': []
                }

            times_k_dict[agent_degree]['convinced'].append(convinced_when[sim][i])
            times_k_dict[agent_degree]['infected'].append(infected_when[sim][i])
            times_k_dict[agent_degree]['removed'].append(removed_when[sim][i])
            times_k_dict[agent_degree]['vaccinated'].append(vaccinated_when[sim][i])

    return times_k_dict

def stat_agent_times_by_degree(times_k_dict):
    DUMMY_VALUE = 9999999

    stat_times_k_dict = {}

    for degree, events in times_k_dict.items():
        stat_times_k_dict[degree] = {}

        for event_type, event_times in events.items():
            event_times = np.asarray(event_times)
            event_times = event_times[event_times!=DUMMY_VALUE]
            
            size = len(event_times)
            mean = np.nanmean(event_times)
            std = np.nanstd(event_times)
            z = 1.96
            l95 = mean - z * std / np.sqrt(size)
            u95 = mean + z * std / np.sqrt(size)

            stat_times_k_dict[degree][event_type] = {
                'avg': mean,
                'std': std,
                'l95': l95,
                'u95': u95,
            }

    return stat_times_k_dict

def stat_agent_times_by_event_degree(times_k_dict):
    DUMMY_VALUE = 9999999

    stat_times_event_k_dict = {event_type: {} for event_type in times_k_dict[next(iter(times_k_dict))]}  # Initialize for event types

    for degree, events in times_k_dict.items():
        for event_type, event_times in events.items():
            event_times = np.asarray(event_times)
            event_times = event_times[event_times != DUMMY_VALUE]

            size = len(event_times)
            mean = np.nanmean(event_times)
            std = np.nanstd(event_times)
            z = 1.96
            l95 = mean - z * std / np.sqrt(size)
            u95 = mean + z * std / np.sqrt(size)

            if degree not in stat_times_event_k_dict[event_type]:
                stat_times_event_k_dict[event_type][degree] = {}

            stat_times_event_k_dict[event_type][degree] = {
                'avg': mean,
                'std': std,
                'l95': l95,
                'u95': u95,
            }

    return stat_times_event_k_dict

def collect_agent_neighborhood_by_degree(
        ids, 
        status, 
        degree, 
        final_act_sus, 
        final_prevalence, 
        final_vaccinated, 
        initial_active_susceptible, 
        zealots,
        ):
    nsims = len(ids)
    nagents = len(ids[0])
    
    neigh_k_dict = {}

    for sim in range(nsims):
        for i in range(nagents):
            agent_degree = degree[sim][i]
            
            # Create a sub-dictionary for the degree if it doesn't exist
            if agent_degree not in neigh_k_dict:
                neigh_k_dict[agent_degree] = {
                    'final_act_sus': [],
                    'final_prevalence': [],
                    'final_vaccinated': [],
                    'initial_active_sus': [],
                    'zealots': [],
                    #'status': [],
                }

            # Append data for the corresponding event
            neigh_k_dict[agent_degree]['final_act_sus'].append(final_act_sus[sim][i])
            neigh_k_dict[agent_degree]['final_prevalence'].append(final_prevalence[sim][i])
            neigh_k_dict[agent_degree]['final_vaccinated'].append(final_vaccinated[sim][i])
            neigh_k_dict[agent_degree]['initial_active_sus'].append(initial_active_susceptible[sim][i])
            #neigh_k_dict[agent_degree]['status'].append(status[sim][i])

    return neigh_k_dict

def stat_agent_neighborhood_by_obs_degree(neigh_k_dict):
    DUMMY_VALUE = 9999999

    stat_neigh_obs_k_dict = {event_type: {} for event_type in neigh_k_dict[next(iter(neigh_k_dict))]}  # Initialize for event types

    for degree, events in neigh_k_dict.items():
        for event_type, event_times in events.items():
            event_times = np.asarray(event_times)
            event_times = event_times[event_times != DUMMY_VALUE]

            size = len(event_times)
            mean = np.nanmean(event_times)
            std = np.nanstd(event_times)
            z = 1.96
            l95 = mean - z * std / np.sqrt(size)
            u95 = mean + z * std / np.sqrt(size)

            if degree not in stat_neigh_obs_k_dict[event_type]:
                stat_neigh_obs_k_dict[event_type][degree] = {}

            stat_neigh_obs_k_dict[event_type][degree] = {
                'avg': mean,
                'std': std,
                'l95': l95,
                'u95': u95,
            }

    return stat_neigh_obs_k_dict

def build_agent_dataframe(
        ids, 
        status, 
        degree, 
        final_act_sus, 
        final_prevalence, 
        final_vaccinated, 
        initial_active_sus, 
        zealots, 
        sim_filter=None
    ):

    nsims = len(ids)
    nagents = len(ids[0])

    sim_list = []
    id_list = []
    degree_list = []
    status_list = []
    final_act_sus_list = []
    final_prevalence_list = []
    final_vaccinated_list = []
    initial_active_susceptible_list = []
    zealots_list = []

    status_int_map = {'ActSus': 0, 'HesSus': 1, 'ActRem': 2, 'HesRem': 2, 'ActVac': 3}

    for sim in range(nsims):
        # Apply the filter if sim_filter is specified and sim is not in sim_filter
        if sim_filter is not None and sim not in sim_filter:
            continue

        for agent in range(nagents):
            # Extract data from your arrays
            sim_list.append(sim)
            id_list.append(ids[sim][agent])
            sta = status_int_map[status[sim][agent]]
            deg = degree[sim][agent]
            fas = final_act_sus[sim][agent] / deg
            fpr = final_prevalence[sim][agent] / deg
            fva = final_vaccinated[sim][agent] / deg
            ias = initial_active_sus[sim][agent] / deg
            zea = zealots[sim][agent] / deg
            
            degree_list.append(deg)
            status_list.append(sta)
            final_act_sus_list.append(fas)
            final_prevalence_list.append(fpr)
            final_vaccinated_list.append(fva)
            initial_active_susceptible_list.append(ias)
            zealots_list.append(zea)

    # Create a dictionary with column names as keys and lists as values
    data = {
        'sim': sim_list,
        'id': id_list,
        'degree': degree_list,
        'status': status_list,
        'final_act_sus': final_act_sus_list,
        'final_prevalence': final_prevalence_list,
        'final_vaccinated': final_vaccinated_list,
        'initial_active_susceptible': initial_active_susceptible_list,
        'zealots': zealots_list
    }

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    return df

def read_vaccination_data(cwd_path):
    lower_path = 'data'
    full_path = os.path.join(cwd_path, lower_path, "vaccination_data.json")
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

def compute_average_theshold(state_thresholds, k_avg):

    share_someone = state_thresholds['someone']
    share_majority = state_thresholds['majority']
    share_never = state_thresholds['never']
    share_total = share_someone + share_majority + share_never

    theta_someone = 1.0 / k_avg
    theta_majority = 0.5
    theta_zealot = 1.0000000001

    return (theta_someone * share_someone \
        + theta_majority * share_majority \
        + theta_zealot * share_never) / share_total


def import_age_distribution(path, country, state):
    """Import age distribution of a territory from a csv file and store it into 
    an array.
    
    Original data from https://github.com/mobs-lab/mixing-patterns

    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state

    Returns
    -------
    - : np.array
        population distribution by age class
    
    """

    lower_path = 'data/'
    file_name = country + '_subnational_' + state + '_age_distribution_85' + '.csv'
    full_name = os.path.join(path, lower_path, file_name)

    age_df = pd.read_csv(full_name, header=None)
    #print(np.sum(age_df.values.T[1]))
    
    pop_a = np.zeros(len(age_df.values.T[1]), dtype=float)
    pop_a = age_df.values.T[1]

    return pop_a
    

def import_updated_age_distribution(path, country, state):
    """Import updated age distribution of a territory from a xlsl file and
    store it into an array.
    
    Data from 2019 census includes a category of 84 y.o. and then 85+. Data
    from 2018 seems to include just a category 84+. To adapt it, the last
    two age classes of 2019 are merged into one of 84+.
    
    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state
        
    Returns
    -------
    - : np.array
        population distribution by age class

    """

    lower_path = 'data/'
    file_name = '2019_' + country + '_subnational_' + state + '_age_distribution_85' + '.xlsx'
    full_name = os.path.join(path, lower_path, file_name)

    full_age_df = pd.read_excel(full_name)

    age_df = full_age_df['Unnamed: 34'][5:90].values
    merge_older = full_age_df['Unnamed: 34'][89] + full_age_df['Unnamed: 34'][90]
    age_df[-1] = merge_older
    
    pop_a = np.zeros(len(age_df), dtype=float)
    for a in range(len(age_df)):
        pop_a[a] = age_df[a]

    return pop_a

def import_contact_matrix(path, country, state):
    """Import contact matrix between age classes of a territory from a csv file 
    and store into an array.
    
    Original data from https://github.com/mobs-lab/mixing-patterns
    
    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state
        
    Returns
    -------
    - : np.array
        measures the average number of contacts for an individual of age i with 
        all of their contacts of age j.
    
    """

    lower_path = 'data/'
    file_name = country + '_subnational_' + state + '_M_overall_contact_matrix_85' + '.csv'
    full_name = os.path.join(path, lower_path, file_name)
    
    contact_df = pd.read_csv(full_name, header=None)

    return contact_df.values


def update_contact_matrix(contact, old_pop_a, new_pop_a):
    """Update contact matrices for every state from the 2005 version obtained
    in Mistry et al. (2020) to a more recent version (2019 census). 

    Updating proceeds by the methods developed in Arregui et al. (2018)
    https://doi.org/10.1371/journal.pcbi.1006638
    In particular, method 2, density correction is employed. End result comes
    from Eq. (5).
    
    The method can be applied to every year.

    Parameters
    ----------
    contact : np.array
        measures the average number of contacts for an individual of age i with 
        all of their contacts of age j.
    old_pop_a : np.array
        population's distribution by age
    new_pop_a : np.array
        updated population's distribution by age

    Returns
    -------
    new_contact : np.array
        updated contact matrix

    """

    N_old = np.sum(old_pop_a)
    N_new = np.sum(new_pop_a)
    A = len(old_pop_a)
    new_contact = np.zeros((A, A), dtype=float)

    for i in range(A):

        for j in range(A):
            
            old_fraction = N_old / old_pop_a[j]
            new_fraction = new_pop_a[j] / N_new
            factor = old_fraction * new_fraction

            new_contact[i][j] = contact[i][j] * factor

    return new_contact

def compute_average_contact_number(pop_a, contact_matrix):

    n = np.sum(pop_a)
    k_avg = np.sum(pop_a * np.sum(contact_matrix, axis=1)) / n
    return k_avg

def export_average_contact_number_by_state(path):

    lower_path = 'data/'

    country = 'United_States'

    state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'Delaware', 'District_of_Columbia', 
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New_Hampshire',  
              'New_Jersey', 'New_Mexico', 'New_York', 'North_Carolina', 
              'North_Dakota', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'Rhode_Island', 'South_Carolina', 'South_Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 
              'Washington', 'West_Virginia', 'Wisconsin', 'Wyoming', 'United_States']

    formatted_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'Delaware', 'DistrictofColumbia', 
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'NewHampshire',  
              'NewJersey', 'NewMexico', 'NewYork', 'NorthCarolina', 
              'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 'SouthDakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 
              'Washington', 'WestVirginia', 'Wisconsin', 'Wyoming', 'National']
    
    state_degree_dict = {}

    for i, state in enumerate(state_list):

        old_pop_a = import_age_distribution(path=path, country=country, state=state)

        pop_a = import_updated_age_distribution(path=path, country=country, state=state)

        contact = import_contact_matrix(path=path, country=country, state=state)

        contact = update_contact_matrix(contact=contact, old_pop_a=old_pop_a, new_pop_a=pop_a)

        k_avg = compute_average_contact_number(pop_a=pop_a, contact_matrix=contact)
        
        state_label = formatted_list[i]

        state_degree_dict[state_label] = k_avg

        print("state: {0}, k_avg={1}".format(state, k_avg))

    filename = 'average_contacts.json'
    fullname = os.path.join(path, lower_path, filename)

    with open(fullname, "w") as file:
        json.dump(state_degree_dict, file)





    

